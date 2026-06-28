"""Cleanup ("branch-and-relabel") DAgger collection: parallel workers + NVENC.

The replacement for the deprecated α-mixing DAgger
(``sim.scripts.collect_dagger_parallel`` / ``sim.dagger.collect_dagger_round``),
which regressed BC by stitching expert/policy actions *inside* π0's action
chunk → incoherent ("Frankenstein") flow-matching targets. See CLAUDE.md §4 /
README.md §3 for the full diagnosis.

Cleanup DAgger decouples *generating on-policy states* from *recording the
label*, so no recorded chunk ever straddles two actors. Per episode:

1. **Scout (NOT recorded).** Roll the policy open-loop (``act_queued``, one
   action per env step — the exact deployment path) from ``env.reset(seed)``.
   Its only job is to find where the policy gets into trouble; the trajectory is
   thrown away.
2. **Anchor (gate).** A **near-violation in the weave phase** —
   ``grasped`` AND ``env.current_clearance() < gate_margin`` AND the cube is not
   yet at the goal (``> dropoff_radius``) — *triggers* an anchor, but the snapshot
   we keep is the **chunk-planning boundary**, not the danger frame. π0 executes an
   action chunk open-loop, so a near-violation is the consequence of the chunk the
   policy planned at the last queue-refill (≤ chunk-length steps earlier); the only
   on-policy state a chunked policy can be corrected at is the one it *planned
   from*. So the scout snapshots each grasped weave-phase planning boundary
   (``len(action_queue) == 0`` → re-infer) and, on a near-violation, branches from
   the most recent one. Rising-edge trigger + ``cooldown_steps`` + per-episode
   ``max_anchors`` cap so one episode can't over-sample.
3. **Branch / relabel.** ``env.restore(boundary_snap)`` then run the **pure
   scripted expert to completion** (``run_expert_episode(restore_state=snap)``),
   recorded as one normal episode — the *exact* ``collect_demos`` path, except the
   expert first re-roots its BFS carry corridor at the restored cube position
   (``replan_carry_from_current``) so it weaves *forward*, never back toward the
   spawn. Every recorded episode is 100 % expert → coherent chunks, zero handoffs;
   its first frame is ``(on-policy planning state → coherent safe expert chunk)``,
   the correct DAgger relabel for a chunked policy.

The recorder is created with ``track_actor=False`` (every recorded frame is the
expert → no ``privileged.actor`` field), so the cleanup dataset has the **same
schema as the BC set** and aggregates with it with NO stripping step — unlike the
deprecated collector (see the dagger-aggregate-strip-actor memo).

Scouts are embarrassingly parallel across seeds, so this spawns ``--n-workers``
processes, each scouting a *contiguous, disjoint* block of seeds into its own
shard and encoding camera videos with a hardware encoder (``--vcodec auto`` →
``h264_nvenc`` on NVIDIA). When all workers finish, ``aggregate_datasets`` merges
the shards into one dataset at ``--root`` (videos copied, not re-encoded) and the
shards are deleted.

⚠️ VRAM: each worker holds its own π0 copy (~18 GB measured) on the single GPU.
On the 96 GB box 4 workers is comfortably safe (~72 GB + overhead, within the
75–85 GB target); 5+ crowds it. Default is 4.

ALWAYS run headless EGL (see CLAUDE.md §1):

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \
        -m sim.scripts.collect_dagger_cleanup \
        --repo-id local/safe-cube-cleanup --root data/safe_cube_cleanup \
        --checkpoint outputs/safe_pi0_bc/checkpoints/last/pretrained_model \
        --dataset-repo-id local/safe-cube-mixed --dataset-root data/safe_cube_v5 \
        --n-red-cubes 8 --max-steps 500 --n-episodes 400 --seed 2000 \
        --gate-margin 0.03 --max-anchors 3 --n-workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

TASK_DESCRIPTION = (
    "pick up the blue cube, carry it low across the table avoiding every red cube, "
    "and place it on the green goal patch"
)


def scout_for_anchors(
    *,
    env,
    policy,
    task: str,
    seed: int,
    max_steps: int,
    gate_margin: float,
    max_anchors: int,
    cooldown_steps: int,
    dropoff_radius: float,
    anchor_mode: str = "weave",
    place_tol: float = 0.015,
    place_approach_radius: float = 0.10,
) -> list[dict]:
    """Roll the policy open-loop from ``env.reset(seed)`` and return snapshots to
    branch the expert from. The rollout is NOT recorded.

    Two anchor *gates*, selected by ``anchor_mode`` (``"weave"`` | ``"place"`` |
    ``"both"``). Both branch from the **chunk-PLANNING boundary, not the trigger
    frame**: π0 executes an action chunk open-loop, so a bad outcome is the
    *consequence* of the chunk planned at the last queue-refill (≤ chunk-length
    steps earlier). The only on-policy state a chunked policy can be corrected at
    is the one it *planned from* (``len(action_queue) == 0`` → re-infer), so each
    gate caches the most recent qualifying planning boundary and, on a rising-edge
    trigger, branches from it. Relabeling the trigger frame, or the policy's
    per-step path, would teach recovery-at-replan at best and reintroduce
    incoherent ("Frankenstein") chunks at worst.

    * **weave** — a *near-violation*: grasped, ee within ``gate_margin`` of a red
      surface (``env.current_clearance()``), cube not yet at the goal (XY-to-goal
      ``> dropoff_radius``, which excludes pickup and dropoff). Branches from the
      most recent grasped, *far-from-goal* (``> dropoff_radius``) boundary → teaches
      "weave clear". This is the original cleanup gate.
    * **place** — a *placement attempt*: grasped and the cube has come within
      ``place_approach_radius`` of the goal (but not already centered within
      ``place_tol``). Branches from the most recent grasped planning boundary —
      captured at *every* grasped re-plan, NOT distance-gated, because the placing
      chunk that lands off-center is often planned from outside
      ``place_approach_radius`` (a 50-step chunk covers a lot of ground), so gating
      the capture would routinely leave no boundary to branch from. Teaches
      "approach → centered drop": it targets the open-loop place undershoot (the
      policy commits the placing chunk and releases before the reach completes,
      landing consistently off-center), and the closed-loop expert re-centers to
      within ``descend2_reach_tol`` before releasing, so the branch is a correct
      centered-place relabel.

    Rising-edge trigger + shared ``cooldown_steps`` + per-scout ``max_anchors`` cap
    so one episode can't over-sample; ``last_anchored`` dedups re-triggers mapping
    to the same boundary (weave and place boundaries are distinct snapshots).
    """
    obs, info = env.reset(seed=seed)
    policy.reset()
    goal_xy = np.asarray(info["goal_pos"], dtype=np.float64)[:2]
    do_weave = anchor_mode in ("weave", "both")
    do_place = anchor_mode in ("place", "both")

    snaps: list[dict] = []
    prev_near = prev_off = False
    cooldown = 0
    steps = 0
    # Latest planning boundary each gate branches from (the state the offending
    # chunk was planned at): weave = grasped & far from goal; place = grasped &
    # near goal. last_anchored dedups re-triggers that map to the same boundary.
    last_boundary: dict | None = None
    last_place_boundary: dict | None = None
    last_anchored: dict | None = None
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps and len(snaps) < max_anchors:
        # Queue empty → this act_queued call re-infers a fresh chunk FROM the
        # current state. That state is what the policy plans from; capture it
        # (cheap — snapshot() copies arrays, no render) per the active gate(s).
        if len(policy.policy._action_queue) == 0 and env._blue_grasped():
            d_goal_b = float(np.linalg.norm(env.blue_cube_position()[:2] - goal_xy))
            if do_weave and d_goal_b > dropoff_radius:
                last_boundary = env.snapshot()
            if do_place:
                # Capture EVERY grasped planning boundary (snapshot is cheap — array
                # copies, no render). The placing chunk that lands off-center is
                # often planned from *outside* place_approach_radius (a 50-step chunk
                # covers a lot of ground), so distance-gating the capture would
                # routinely leave no boundary to branch from. The near-goal gate is
                # on the TRIGGER below; here we just keep the latest chunk origin.
                last_place_boundary = env.snapshot()

        # act_queued pops one action from the open-loop chunk — the
        # deployment-distribution states we want to scout.
        action = policy.act_queued(obs, task)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1

        grasped = bool(info["grasped"])
        cube_xy = np.asarray(info["blue_cube_pos"], dtype=np.float64)[:2]
        d_goal = float(np.linalg.norm(cube_xy - goal_xy))
        at_goal = d_goal <= dropoff_radius

        # weave near-violation (rising edge)
        near = do_weave and grasped and not at_goal and env.current_clearance() < gate_margin
        if (near and not prev_near and cooldown <= 0
                and last_boundary is not None and last_boundary is not last_anchored):
            snaps.append(last_boundary)
            last_anchored = last_boundary
            cooldown = cooldown_steps
        prev_near = near

        # placement off-center (rising edge): grasped, within the approach band
        # of the goal but still off-center (> place_tol). The band's upper bound is
        # place_approach_radius (not dropoff_radius) so it fires on the descent
        # wherever the policy commits to its off-center drop, not only in the last
        # few cm — the grasped cube often doesn't dwell inside dropoff_radius before
        # releasing.
        off = do_place and grasped and place_tol < d_goal <= place_approach_radius
        if (off and not prev_off and cooldown <= 0
                and last_place_boundary is not None and last_place_boundary is not last_anchored):
            snaps.append(last_place_boundary)
            last_anchored = last_place_boundary
            cooldown = cooldown_steps
        prev_off = off

        if cooldown > 0:
            cooldown -= 1
    return snaps


def _collect_block(payload: dict) -> dict:
    """Worker: scout seeds ``[ep_start, ep_end)`` and branch each anchor into one
    shard dataset.

    Heavy imports happen here so the parent stays light and (with the 'spawn'
    start method) each worker initializes its own EGL/MuJoCo + CUDA + NVENC
    context — including a private ~18 GB π0 copy via PolicyRollout."""
    from lerobot.configs.video import VideoEncoderConfig
    from sim.configs import EnvConfig, ExpertConfig, SceneConfig
    from sim.dagger import PolicyRollout
    from sim.env import SafeCubeEnv
    from sim.expert import ScriptedExpert
    from sim.recorder import EpisodeRecorder
    from sim.rollout import run_expert_episode

    wid = payload["wid"]
    base_seed = payload["base_seed"]
    ep_start, ep_end = payload["ep_start"], payload["ep_end"]
    task = payload["task"]
    max_steps = payload["max_steps"]
    branch_cap = payload["branch_cap"]
    successes_only = payload["successes_only"]
    min_branch_frames = payload["min_branch_frames"]
    target_branches = payload.get("target_branches")  # per-worker saved-branch cap

    scene = SceneConfig(
        n_red_cubes=payload["n_red"],
        path_clearance_weight=payload["path_clearance_weight"],
        path_clearance_pref=payload["path_clearance_pref"],
        path_wall_field_sides=payload["path_wall_field_sides"],
    )
    # terminate_on_red_contact=False: the scout must keep rolling past finger
    # grazes to surface multiple near-violations, and a branch (which starts from
    # a near-violation) shouldn't die on the first graze — branch quality is
    # judged by the discard filter below, not by early termination.
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=max_steps,
                                seed=base_seed + ep_start,
                                terminate_on_red_contact=False))
    obs, _ = env.reset(seed=base_seed + ep_start)
    state_dim = obs["state"].shape[0]

    # track_actor=False → no privileged.actor field → SAME schema as the BC set,
    # so this aggregates with it with no stripping (see module docstring).
    rec = EpisodeRecorder.create(
        repo_id=payload["repo_id"], root=payload["root"], n_red_cubes=payload["n_red"],
        image_size=scene.image_size, action_dim=env.action_dim, state_dim=state_dim,
        fps=env.cfg.fps, use_videos=True,
        camera_encoder=VideoEncoderConfig(vcodec=payload["vcodec"]),
        track_actor=False,
    )
    expert = ScriptedExpert(env=env, cfg=ExpertConfig())
    grip_open = ExpertConfig().grip_open

    # One π0 copy per worker (~18 GB VRAM). Normalization stats + feature shapes
    # come from the dataset the checkpoint was trained on (the BC set), which is
    # NOT where we write the new cleanup demos (--repo-id / --root).
    policy = PolicyRollout(
        checkpoint=payload["checkpoint"],
        dataset_repo_id=payload["dataset_repo_id"],
        dataset_root=payload["dataset_root"],
        device="cuda",
    )

    scouts = anchors = saved = discarded = success = lfail = 0
    for ep in range(ep_start, ep_end):
        seed = base_seed + ep
        try:
            snaps = scout_for_anchors(
                env=env, policy=policy, task=task, seed=seed, max_steps=max_steps,
                gate_margin=payload["gate_margin"], max_anchors=payload["max_anchors"],
                cooldown_steps=payload["cooldown_steps"],
                dropoff_radius=payload["dropoff_radius"],
                anchor_mode=payload["anchor_mode"],
                place_tol=payload["place_tol"],
                place_approach_radius=payload["place_approach_radius"],
            )
        except RuntimeError:
            # sample_layout exhausted its attempts for this seed — skip it.
            lfail += 1
            continue
        scouts += 1
        anchors += len(snaps)

        for snap in snaps:
            # Branch: restore the planning boundary the offending chunk came from
            # and run the PURE expert to completion (it re-roots its BFS corridor at
            # the restored cube position inside run_expert_episode). Optionally cap
            # how far past the anchor it runs (default: run to --max-steps).
            # restore() resets _stats.steps to 0, so swapping max_episode_steps
            # truncates the branch at branch_cap.
            env.cfg.max_episode_steps = branch_cap if branch_cap is not None else max_steps
            try:
                stats = run_expert_episode(
                    env=env, expert=expert, rec=rec, task=task,
                    seed=seed, grip_open=grip_open,
                    choose_executed=None, restore_state=snap,
                )
            finally:
                env.cfg.max_episode_steps = max_steps
            # Keep only CLEAN expert successes so the cleanup set matches the BC
            # set's distribution (the BC collector drops fly-overs and terminates
            # on red contact under --successes-only): drop fly-overs (poison for
            # the stay-low task) and finger-clip red contacts always, and drop
            # non-successes under --successes-only.  Also drop degenerate-short
            # branches (a placement anchor restored right at the goal can finish in
            # a handful of frames): they carry no useful training signal AND some HW
            # encoders (h264_nvenc) fail on ultra-short clips inside save_episode().
            too_short = rec.frames_in_episode < min_branch_frames
            clean = not stats["fly_over"] and not stats["red_contact"]
            if clean and not too_short and (stats["success"] or not successes_only):
                rec.save_episode()
                saved += 1
                success += int(stats["success"])
            else:
                rec.discard_episode()
                discarded += 1
        # Stop scouting once this worker has saved its share of the branch target.
        # The seed block is sized generously (see main) so the target is reached
        # before the block is exhausted; remaining seeds in the block go unused.
        if target_branches is not None and saved >= target_branches:
            break
    env.close()
    rec.finalize()
    return dict(wid=wid, scouts=scouts, anchors=anchors, saved=saved,
                discarded=discarded, success=success, lfail=lfail,
                repo_id=payload["repo_id"], root=payload["root"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="dataset repo id for the NEW cleanup demos")
    p.add_argument("--root", type=str, required=True, help="dataset root for the NEW cleanup demos")
    p.add_argument("--checkpoint", required=True,
                   help="path to a trained `pretrained_model` dir to scout with (the BC policy)")
    p.add_argument("--dataset-repo-id", required=True,
                   help="dataset the checkpoint was trained on — supplies normalization "
                        "stats / feature shapes to PolicyRollout (the BC set, not --repo-id)")
    p.add_argument("--dataset-root", type=str, default=None,
                   help="root of --dataset-repo-id on disk (needed for local/* datasets)")
    p.add_argument("--n-episodes", type=int, default=400,
                   help="scout-episode budget (upper bound). With --target-branches this is "
                        "just a generous cap; each worker stops early once it hits its branch share")
    p.add_argument("--target-branches", type=int, default=None,
                   help="stop once this many CLEAN branches are saved across all workers "
                        "(each worker targets target/n_workers, then stops scouting). "
                        "Set --n-episodes high enough that the seed block isn't exhausted first")
    p.add_argument("--n-red-cubes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0,
                   help="scout N uses seed+N; keep ≥ collection N so layouts stay held out")
    p.add_argument("--max-steps", type=int, default=500)
    # ── Gate / anchor knobs ──────────────────────────────────────────────
    p.add_argument("--anchor-mode", choices=("weave", "place", "both"), default="weave",
                   help="which gate fires: 'weave' (near-red during the carry — the "
                        "original cleanup gate), 'place' (off-center drop at the goal), "
                        "or 'both'. 'place' targets the open-loop placement undershoot")
    p.add_argument("--gate-margin", type=float, default=0.025,
                   help="[weave] near-violation clearance threshold (m): anchor when the "
                        "grasped ee comes within this of a red surface during the weave")
    p.add_argument("--place-tol", type=float, default=0.015,
                   help="[place] inner guard (m): don't anchor if the grasped cube is already "
                        "within this of the goal center (already a centered drop)")
    p.add_argument("--place-approach-radius", type=float, default=0.10,
                   help="[place] trigger radius (m): anchor when the grasped cube first comes "
                        "within this of the goal (a placement attempt). The branch point is "
                        "the most recent grasped planning boundary — captured regardless of "
                        "distance, since the placing chunk often originates farther out")
    p.add_argument("--max-anchors", type=int, default=3,
                   help="per-scout cap on anchors (branches), so one episode can't over-sample")
    p.add_argument("--cooldown-steps", type=int, default=20,
                   help="env steps to suppress re-anchoring after an anchor fires")
    p.add_argument("--dropoff-radius", type=float, default=0.05,
                   help="cube-to-goal XY distance (m) under which the cube counts as 'at the "
                        "goal' → no anchor (excludes the dropoff from the weave gate)")
    p.add_argument("--branch-cap", type=int, default=None,
                   help="cap a branch's length (env steps past the anchor); default = run to "
                        "completion (--max-steps)")
    p.add_argument("--min-branch-frames", type=int, default=8,
                   help="discard branches shorter than this many frames before saving. Guards "
                        "against degenerate-short placement anchors (restored right at the goal) "
                        "that carry no signal and can crash HW encoders on ultra-short clips. "
                        "Weave branches are long, so the default never bites them")
    p.add_argument("--successes-only", action=argparse.BooleanOptionalAction, default=True,
                   help="keep only branches that reach the goal (default on); "
                        "--no-successes-only keeps failed branches too")
    # ── Parallelism / encoding ───────────────────────────────────────────
    p.add_argument("--n-workers", type=int, default=4,
                   help="parallel workers; EACH holds a ~18 GB π0 copy → 4 comfortably safe on the "
                        "96 GB box (~72 GB + overhead), 5+ crowds the 75–85 GB target")
    p.add_argument("--vcodec", type=str, default="auto",
                   help="'auto' → first available HW encoder (h264_nvenc on NVIDIA), "
                        "else 'libsvtav1' software. Pass an explicit codec to force it. "
                        "NOTE: --anchor-mode place branches are SHORT (~30-100 frames); "
                        "h264_nvenc can fail to emit a file on the shortest clips — use "
                        "'libsvtav1' for place mode.")
    p.add_argument("--keep-shards", action="store_true",
                   help="don't delete the per-worker shard datasets after merge")
    # ── A* corridor planner (used by the expert in each branch) ─────────────
    p.add_argument("--path-clearance-weight", type=float, default=1.0,
                   help="A* soft-clearance weight λ: 0 = plain BFS; >0 = cost-field A* "
                        "that bows corridors toward gap centers (λ≥3 routes around the field)")
    p.add_argument("--path-clearance-pref", type=float, default=0.04,
                   help="extra standoff (m) beyond the hard clearance radius the planner prefers")
    p.add_argument("--path-wall-field-sides", action="store_true",
                   help="hard-block lateral margins so A* weaves through the field (use with high λ)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Keep BLAS from oversubscribing: many workers × few threads ≈ core count.
    # Set before workers (spawn) import numpy so they inherit it.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")

    if args.n_workers >= 5:
        print(f"⚠️  --n-workers={args.n_workers}: each worker holds a ~18 GB π0 copy; "
              f"5+ pushes past the 75–85 GB target on the 96 GB box. ≤4 is the safe range.", flush=True)

    K = max(1, args.n_workers)
    N = args.n_episodes
    # Contiguous, disjoint blocks → seeds base..base+N-1 each scouted once.
    bounds = [round(i * N / K) for i in range(K + 1)]
    root = Path(args.root)
    # Branch-count target: split evenly across workers (ceil so the sum ≥ target).
    per_worker_target = (
        -(-args.target_branches // K) if args.target_branches is not None else None
    )

    payloads = []
    for i in range(K):
        if bounds[i] == bounds[i + 1]:
            continue  # more workers than episodes
        payloads.append(dict(
            wid=i, ep_start=bounds[i], ep_end=bounds[i + 1], base_seed=args.seed,
            repo_id=f"{args.repo_id}-part{i}", root=f"{root}_part{i}",
            checkpoint=args.checkpoint, dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root, task=TASK_DESCRIPTION,
            n_red=args.n_red_cubes, max_steps=args.max_steps,
            gate_margin=args.gate_margin, max_anchors=args.max_anchors,
            cooldown_steps=args.cooldown_steps, dropoff_radius=args.dropoff_radius,
            anchor_mode=args.anchor_mode, place_tol=args.place_tol,
            place_approach_radius=args.place_approach_radius,
            branch_cap=args.branch_cap, successes_only=args.successes_only,
            min_branch_frames=args.min_branch_frames,
            target_branches=per_worker_target,
            vcodec=args.vcodec,
            path_clearance_weight=args.path_clearance_weight,
            path_clearance_pref=args.path_clearance_pref,
            path_wall_field_sides=args.path_wall_field_sides,
        ))

    # Fresh shard dirs.
    for pl in payloads:
        shutil.rmtree(pl["root"], ignore_errors=True)

    target_msg = (f"target={args.target_branches} branches ({per_worker_target}/worker), "
                  f"≤{N // K} scouts/worker" if per_worker_target is not None
                  else f"~{N // K} scouts/worker")
    gate_msg = (f"gate_margin={args.gate_margin}" if args.anchor_mode == "weave"
                else f"place_tol={args.place_tol},approach_r={args.place_approach_radius}"
                if args.anchor_mode == "place"
                else f"gate_margin={args.gate_margin},place_tol={args.place_tol}")
    print(f"Launching {len(payloads)} workers × {target_msg}  "
          f"(anchor_mode={args.anchor_mode}, {gate_msg}, max_anchors={args.max_anchors}, "
          f"vcodec={args.vcodec}, n_red={args.n_red_cubes}, "
          f"successes_only={args.successes_only})\n"
          f"  policy={args.checkpoint}\n"
          f"  stats from={args.dataset_repo_id} (root={args.dataset_root})", flush=True)
    t0 = time.time()
    # ProcessPoolExecutor workers are NON-daemonic, so each may spawn its own
    # child (LeRobot's save_episode launches an encoder subprocess) — a plain
    # multiprocessing.Pool would assert "daemonic processes are not allowed to
    # have children". 'spawn' keeps each worker's EGL/MuJoCo/CUDA/NVENC context clean.
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as ex:
        results = list(ex.map(_collect_block, payloads))
    dt = time.time() - t0

    scouts = sum(r["scouts"] for r in results)
    anchors = sum(r["anchors"] for r in results)
    saved = sum(r["saved"] for r in results)
    discarded = sum(r["discarded"] for r in results)
    success = sum(r["success"] for r in results)
    lfail = sum(r["lfail"] for r in results)
    rate = saved / dt if dt > 0 else 0.0
    print(f"\nWorkers done in {dt/60:.1f} min  ({rate:.2f} saved-ep/s aggregate)", flush=True)
    print(f"  scouts={scouts}  anchors={anchors}  layout_fail={lfail}", flush=True)
    print(f"  branches: saved={saved}  discarded={discarded}  "
          f"(saved successes={success})", flush=True)

    # Merge shards that actually saved episodes.
    from lerobot.datasets.aggregate import aggregate_datasets

    merge = [r for r in results if r["saved"] > 0]
    if not merge:
        raise RuntimeError(
            "No shard saved any branch — no anchors fired (try a larger --gate-margin "
            "or --max-anchors), or every branch was discarded.")
    shutil.rmtree(root, ignore_errors=True)
    print(f"\nMerging {len(merge)} shards → {root} ...", flush=True)
    aggregate_datasets(
        repo_ids=[r["repo_id"] for r in merge],
        aggr_repo_id=args.repo_id,
        roots=[Path(r["root"]) for r in merge],
        aggr_root=root,
    )

    if not args.keep_shards:
        for r in results:
            shutil.rmtree(r["root"], ignore_errors=True)
        print("Removed shard datasets.", flush=True)

    print(f"\nDone. {saved} cleanup episodes → {root}", flush=True)


if __name__ == "__main__":
    main()
