"""Benchmark a trained safe_pi0 checkpoint over a shard of held-out seeds.

Headless (no video) closed-loop eval — same ``PolicyRollout.act_queued`` path as
``record_policy_rollout.py`` (always action-chunked, per CLAUDE.md §3), but
reports the full stat set including ``fly_over`` (which ``sim.evaluate.evaluate``
does not aggregate) and writes per-episode rows to a jsonl file so parallel
shards can be combined afterward. Intended to be launched N-way by a driver
script across disjoint seed ranges (one process per GPU-resident policy copy).

Example (one shard of a 4-way 1000-episode benchmark)::

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \\
        -m sim.scripts.benchmark_policy \\
        --checkpoint outputs/safe_pi0_bc_v7/checkpoints/last/pretrained_model \\
        --dataset-repo-id local/safe-cube-mixed --dataset-root data/safe_cube_v7 \\
        --seed 10000 --n-episodes 250 --out outputs/benchmark_bcv7_shard0.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sim.configs import EnvConfig, SceneConfig
from sim.dagger import PolicyRollout
from sim.env import SafeCubeEnv
from sim.scripts.collect_demos import TASK_DESCRIPTION


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="path to <output_dir>/checkpoints/<step>/pretrained_model")
    p.add_argument("--dataset-repo-id", required=True,
                   help="dataset the policy was trained on (feature shapes + norm stats)")
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--task", type=str, default=TASK_DESCRIPTION)
    p.add_argument("--n-episodes", type=int, required=True, help="episodes in THIS shard")
    p.add_argument("--seed", type=int, required=True,
                   help="base seed for this shard; episode i uses seed + i")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--n-red-cubes", type=int, default=SceneConfig().n_red_cubes)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=Path, default=None, help="jsonl of per-episode stats")
    return p.parse_args()


def _summary_line(rows: list[dict], n: int) -> str:
    successes = sum(r["success"] for r in rows)
    contacts = sum(r["red_contact"] for r in rows)
    ceilings = sum(r["ceiling_violation"] for r in rows)
    fly_overs = sum(r["fly_over"] for r in rows)
    drops = sum(r["blue_dropped"] for r in rows)
    timeouts = sum(
        1 for r in rows
        if not r["success"] and not r["red_contact"]
        and not r["ceiling_violation"] and not r["blue_dropped"]
    )
    mean_steps = sum(r["steps"] for r in rows) / n
    mean_clear = sum(r["min_clearance"] for r in rows) / n
    return (
        f"episodes={n}/{n}  SUCCESS={successes}/{n} ({successes / n:.1%})  "
        f"red_contact={contacts}/{n} ({contacts / n:.1%})  ceiling={ceilings}/{n}  "
        f"fly_over={fly_overs}/{n}  blue_drop={drops}/{n}  timeout={timeouts}/{n}  "
        f"mean_steps={mean_steps:.1f}  mean_min_clear={mean_clear:.3f}m"
    )


def main() -> None:
    args = parse_args()
    scene = SceneConfig(n_red_cubes=args.n_red_cubes)
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=args.max_steps, seed=args.seed))
    policy = PolicyRollout(
        checkpoint=args.checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        device=args.device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True) if args.out else None
    out_f = args.out.open("w") if args.out else None

    rows: list[dict] = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        policy.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy.act_queued(obs, args.task)
            obs, _, terminated, truncated, info = env.step(action)
        stats = info["stats"]
        rows.append(stats)
        if out_f is not None:
            out_f.write(json.dumps({"seed": args.seed + ep, **stats}) + "\n")
            out_f.flush()

        n = ep + 1
        elapsed = time.time() - t0
        eta_min = elapsed / n * (args.n_episodes - n) / 60
        print(f"[{n}/{args.n_episodes}  {elapsed / n:.1f}s/ep  ETA {eta_min:.0f}m] "
              + _summary_line(rows, n), flush=True)

    print("\n=== FINAL ===")
    print(_summary_line(rows, args.n_episodes))
    if out_f is not None:
        out_f.close()
    env.close()


if __name__ == "__main__":
    main()
