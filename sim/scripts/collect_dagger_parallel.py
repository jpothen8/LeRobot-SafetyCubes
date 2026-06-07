"""Accelerated DAgger collection: parallel workers + GPU (NVENC) video encoding.

The parallel cousin of ``sim.scripts.collect_demos_parallel``, but instead of
rolling the *pure* scripted expert it loads a trained checkpoint into **each
worker** and runs the DAgger mixing loop: at every step it executes the expert
with probability ``--alpha`` else the policy's queued action, while **always
recording the expert action** as the label (the DAgger invariant — see
``sim.dagger.collect_dagger_round``). The policy-induced states are exactly the
off-distribution / near-violation data the safety term needs.

DAgger episodes are embarrassingly parallel across seeds (independent layouts),
so this spawns ``--n-workers`` processes, each rolling a *contiguous, disjoint*
block of episode seeds into its own shard dataset and encoding camera videos
with a hardware encoder (``--vcodec auto`` → ``h264_nvenc`` on NVIDIA). When all
workers finish, ``aggregate_datasets`` merges the shards into one dataset at
``--root`` (videos copied, not re-encoded) and the shards are deleted. Seeds
0..N-1 are covered exactly once, so eval seeds ≥ N stay held out.

This is a drop-in for *only* the collection phase of ``sim.scripts.dagger``;
retraining still runs serially afterward (shell out to ``train_safe_pi0`` as in
the serial orchestrator). Unlike the pure-demo collector this does NOT drop
fly-overs: the recorded label is the (carry-low) expert, so a policy-induced
fly-over state is corrective data, not poison — matching ``collect_dagger_round``.

⚠️ VRAM: each worker holds its own π0 copy (~18 GB measured) on the single GPU.
On the 96 GB box 4 workers is comfortably safe (~72 GB + per-worker overhead,
well within the 75–85 GB target); 5+ starts crowding it. Default is 4. The demo
collector's default of 16 would OOM instantly.

ALWAYS run headless EGL (see CLAUDE.md §1):

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \
        -m sim.scripts.collect_dagger_parallel \
        --repo-id local/safe-cube-dagger --root data/safe_cube_dagger \
        --checkpoint outputs/safe_pi0_bc/checkpoints/last/pretrained_model \
        --dataset-repo-id local/safe-cube-mixed --dataset-root data/safe_cube_v5 \
        --alpha 0.5 --n-red-cubes 8 --max-steps 500 \
        --n-episodes 400 --seed 2000 --n-workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

TASK_DESCRIPTION = (
    "pick up the blue cube, carry it low across the table avoiding every red cube, "
    "and place it on the green goal patch"
)


def _collect_block(payload: dict) -> dict:
    """Worker: roll DAgger episodes [ep_start, ep_end) into one shard dataset.

    Heavy imports happen here so the parent stays light and (with the 'spawn'
    start method) each worker initializes its own EGL/MuJoCo + CUDA + NVENC
    context — including a private ~22 GB π0 copy via PolicyRollout."""
    import numpy as np

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
    alpha = payload["alpha"]
    task = payload["task"]

    scene = SceneConfig(n_red_cubes=payload["n_red"])
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=payload["max_steps"],
                                seed=base_seed + ep_start))
    obs, _ = env.reset(seed=base_seed + ep_start)
    state_dim = obs["state"].shape[0]

    rec = EpisodeRecorder.create(
        repo_id=payload["repo_id"], root=payload["root"], n_red_cubes=payload["n_red"],
        image_size=scene.image_size, action_dim=env.action_dim, state_dim=state_dim,
        fps=env.cfg.fps, use_videos=True,
        camera_encoder=VideoEncoderConfig(vcodec=payload["vcodec"]),
    )
    expert = ScriptedExpert(env=env, cfg=ExpertConfig())
    grip_open = ExpertConfig().grip_open

    # One π0 copy per worker (~18 GB VRAM measured). Normalization stats + feature shapes
    # come from the dataset the checkpoint was trained on (the BC set), which is
    # NOT necessarily where we write the new relabels (--repo-id / --root).
    policy = PolicyRollout(
        checkpoint=payload["checkpoint"],
        dataset_repo_id=payload["dataset_repo_id"],
        dataset_root=payload["dataset_root"],
        device="cuda",
    )

    # Per-worker RNG for the expert/policy mixing coin flips (disjoint, deterministic).
    rng = np.random.default_rng(base_seed + ep_start)

    def choose_executed(expert_action, obs, info):
        # DAgger: execute the expert w.p. alpha, else the policy rolled the SAME
        # (queued) way it is deployed, so we visit the deployment distribution.
        # The label recorded by run_expert_episode is always the expert action.
        if rng.random() < alpha:
            return expert_action
        return policy.act_queued(obs, task)

    saved = discarded = success = red_contact = flyover = lfail = 0
    for ep in range(ep_start, ep_end):
        policy.reset()  # reset the action queue once per episode (queued rollout)
        try:
            stats = run_expert_episode(
                env=env, expert=expert, rec=rec, task=task,
                seed=base_seed + ep, grip_open=grip_open,
                choose_executed=choose_executed,
            )
        except RuntimeError:
            # sample_layout exhausted its attempts for this seed — skip it.
            lfail += 1
            continue
        success += int(stats["success"])
        red_contact += int(stats.get("red_contact", False))
        flyover += int(stats.get("fly_over", False))
        # Keep fly-overs (unlike the pure-demo collector): the recorded label is
        # the carry-low expert, so a policy-induced fly-over state is exactly the
        # corrective data the safety term wants. Drop only failures under
        # --successes-only — matching sim.dagger.collect_dagger_round.
        if payload["successes_only"] and not stats["success"]:
            rec.discard_episode()
            discarded += 1
        else:
            rec.save_episode()
            saved += 1
    env.close()
    rec.finalize()
    return dict(wid=wid, saved=saved, discarded=discarded, success=success,
                red_contact=red_contact, flyover=flyover, lfail=lfail,
                repo_id=payload["repo_id"], root=payload["root"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="dataset repo id for the NEW relabels")
    p.add_argument("--root", type=str, required=True, help="dataset root for the NEW relabels")
    p.add_argument("--checkpoint", required=True,
                   help="path to a trained `pretrained_model` dir to roll out (DAgger policy)")
    p.add_argument("--dataset-repo-id", required=True,
                   help="dataset the checkpoint was trained on — supplies normalization "
                        "stats / feature shapes to PolicyRollout (often the BC set, not --repo-id)")
    p.add_argument("--dataset-root", type=str, default=None,
                   help="root of --dataset-repo-id on disk (needed for local/* datasets)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="expert-mixing probability: execute expert w.p. alpha, else the policy")
    p.add_argument("--n-episodes", type=int, default=400)
    p.add_argument("--n-red-cubes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0,
                   help="episode N uses seed+N; keep ≥ collection N so layouts stay held out")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--successes-only", action="store_true",
                   help="discard failed episodes (off by default: DAgger wants the failures)")
    p.add_argument("--n-workers", type=int, default=4,
                   help="parallel workers; EACH holds a ~18 GB π0 copy → 4 comfortably safe on the "
                        "96 GB box (~72 GB + overhead), 5+ crowds the 75–85 GB target")
    p.add_argument("--vcodec", type=str, default="auto",
                   help="'auto' → first available HW encoder (h264_nvenc on NVIDIA), "
                        "else 'libsvtav1' software. Pass an explicit codec to force it.")
    p.add_argument("--keep-shards", action="store_true",
                   help="don't delete the per-worker shard datasets after merge")
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
    # Contiguous, disjoint blocks → seeds base..base+N-1 each used once.
    bounds = [round(i * N / K) for i in range(K + 1)]
    root = Path(args.root)

    payloads = []
    for i in range(K):
        if bounds[i] == bounds[i + 1]:
            continue  # more workers than episodes
        payloads.append(dict(
            wid=i, ep_start=bounds[i], ep_end=bounds[i + 1], base_seed=args.seed,
            repo_id=f"{args.repo_id}-part{i}", root=f"{root}_part{i}",
            checkpoint=args.checkpoint, dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root, alpha=args.alpha, task=TASK_DESCRIPTION,
            n_red=args.n_red_cubes, max_steps=args.max_steps,
            successes_only=args.successes_only, vcodec=args.vcodec,
        ))

    # Fresh shard dirs.
    for pl in payloads:
        shutil.rmtree(pl["root"], ignore_errors=True)

    print(f"Launching {len(payloads)} workers × ~{N // K} episodes  "
          f"(alpha={args.alpha}, vcodec={args.vcodec}, n_red={args.n_red_cubes}, "
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

    saved = sum(r["saved"] for r in results)
    discarded = sum(r["discarded"] for r in results)
    success = sum(r["success"] for r in results)
    red_contact = sum(r["red_contact"] for r in results)
    flyover = sum(r["flyover"] for r in results)
    lfail = sum(r["lfail"] for r in results)
    attempted = saved + discarded
    rate = saved / dt if dt > 0 else 0.0
    print(f"\nWorkers done in {dt/60:.1f} min  ({rate:.2f} saved-ep/s aggregate)", flush=True)
    print(f"  saved={saved}  discarded={discarded}  layout_fail={lfail}", flush=True)
    if attempted:
        print(f"  rollout stats: success={success}/{attempted}  "
              f"red_contact={red_contact}/{attempted}  fly_over={flyover}/{attempted}", flush=True)

    # Merge shards that actually saved episodes.
    from lerobot.datasets.aggregate import aggregate_datasets

    merge = [r for r in results if r["saved"] > 0]
    if not merge:
        raise RuntimeError("No shard saved any episodes — nothing to merge.")
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

    print(f"\nDone. {saved} episodes → {root}", flush=True)


if __name__ == "__main__":
    main()
