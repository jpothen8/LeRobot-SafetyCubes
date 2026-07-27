"""Accelerated demo collection: parallel workers + GPU (NVENC) video encoding.

Collection is embarrassingly parallel across episodes (each seed is independent),
so this spawns ``--n-workers`` processes, each rolling the scripted expert over a
*contiguous, disjoint* block of episode seeds into its own shard dataset, encoding
the camera videos with a hardware encoder (``--vcodec auto`` → ``h264_nvenc`` on
NVIDIA). When all workers finish, ``aggregate_datasets`` merges the shards into one
dataset at ``--root`` (videos are copied, not re-encoded) and the shards are
deleted. Seeds 0..N-1 are covered exactly once, so eval seeds ≥ N stay held out.

Same discard policy as ``collect_demos``: fly-overs are dropped unconditionally; on
``--successes-only`` failures are dropped too.

ALWAYS run headless EGL (see CLAUDE.md):

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \
        -m sim.scripts.collect_demos_parallel \
        --repo-id local/safe-cube-mixed --root data/safe_cube_v6 \
        --successes-only --n-red-cubes 8 --max-steps 500 \
        --n-episodes 1600 --seed 0 --n-workers 16
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
    """Worker: roll episodes [ep_start, ep_end) into one shard dataset.

    Heavy imports happen here so the parent stays light and (with the 'spawn'
    start method) each worker initializes its own EGL/MuJoCo + NVENC context."""
    import numpy as np

    from lerobot.configs.video import VideoEncoderConfig
    from sim.configs import EnvConfig, ExpertConfig, SceneConfig
    from sim.env import SafeCubeEnv
    from sim.expert import ScriptedExpert
    from sim.recorder import EpisodeRecorder
    from sim.rollout import run_expert_episode

    wid = payload["wid"]
    base_seed = payload["base_seed"]
    ep_start, ep_end = payload["ep_start"], payload["ep_end"]

    scene = SceneConfig(
        n_red_cubes=payload["n_red"],
        path_clearance_weight=payload["path_clearance_weight"],
        path_clearance_pref=payload["path_clearance_pref"],
        path_wall_field_sides=payload["path_wall_field_sides"],
    )
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

    saved = discarded = flyover = lfail = 0
    for ep in range(ep_start, ep_end):
        try:
            stats = run_expert_episode(
                env=env, expert=expert, rec=rec, task=TASK_DESCRIPTION,
                seed=base_seed + ep, grip_open=grip_open,
            )
        except RuntimeError:
            # sample_layout exhausted its attempts for this seed — skip it.
            lfail += 1
            continue
        fell_over = bool(stats.get("fly_over", False))
        if fell_over or (payload["successes_only"] and not stats["success"]):
            rec.discard_episode()
            discarded += 1
            flyover += int(fell_over)
        else:
            rec.save_episode()
            saved += 1
    env.close()
    rec.finalize()
    return dict(wid=wid, saved=saved, discarded=discarded, flyover=flyover, lfail=lfail,
                repo_id=payload["repo_id"], root=payload["root"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--n-episodes", type=int, default=1600)
    p.add_argument("--n-red-cubes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--successes-only", action="store_true")
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--vcodec", type=str, default="auto",
                   help="'auto' → first available HW encoder (h264_nvenc on NVIDIA), "
                        "else 'libsvtav1' software. Pass an explicit codec to force it.")
    p.add_argument("--keep-shards", action="store_true",
                   help="don't delete the per-worker shard datasets after merge")
    # ── A* corridor planner ──────────────────────────────────────────────────
    p.add_argument("--path-clearance-weight", type=float, default=1.0,
                   help="A* soft-clearance weight λ: 0=BFS, >0=cost-field A*")
    p.add_argument("--path-clearance-pref", type=float, default=0.04,
                   help="extra standoff (m) beyond the hard clearance radius")
    p.add_argument("--path-wall-field-sides", action="store_true",
                   help="hard-block lateral margins to force through-field weaving")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Keep BLAS from oversubscribing: many workers × few threads ≈ core count.
    # Set before workers (spawn) import numpy so they inherit it.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")

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
            n_red=args.n_red_cubes, max_steps=args.max_steps,
            successes_only=args.successes_only, vcodec=args.vcodec,
            path_clearance_weight=args.path_clearance_weight,
            path_clearance_pref=args.path_clearance_pref,
            path_wall_field_sides=args.path_wall_field_sides,
        ))

    # Fresh shard dirs.
    for pl in payloads:
        shutil.rmtree(pl["root"], ignore_errors=True)

    print(f"Launching {len(payloads)} workers × ~{N // K} episodes  "
          f"(vcodec={args.vcodec}, n_red={args.n_red_cubes}, successes_only={args.successes_only})",
          flush=True)
    t0 = time.time()
    # ProcessPoolExecutor workers are NON-daemonic, so each may spawn its own
    # child (LeRobot's save_episode launches an encoder subprocess) — a plain
    # multiprocessing.Pool would assert "daemonic processes are not allowed to
    # have children". 'spawn' keeps each worker's EGL/MuJoCo/NVENC context clean.
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as ex:
        results = list(ex.map(_collect_block, payloads))
    dt = time.time() - t0

    saved = sum(r["saved"] for r in results)
    discarded = sum(r["discarded"] for r in results)
    flyover = sum(r["flyover"] for r in results)
    lfail = sum(r["lfail"] for r in results)
    rate = saved / dt if dt > 0 else 0.0
    print(f"\nWorkers done in {dt/60:.1f} min  ({rate:.2f} saved-ep/s aggregate)", flush=True)
    print(f"  saved={saved}  discarded={discarded} (fly_over={flyover})  layout_fail={lfail}", flush=True)

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
