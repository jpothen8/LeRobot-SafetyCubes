"""Collect demonstration episodes by rolling the scripted expert.

Example:

    uv run python -m sim.scripts.collect_demos \
        --repo-id local/safe-cube-demos \
        --root data/safe_cube_demos \
        --n-episodes 50

Writes a LeRobotDataset. Failures (red contact, ceiling violation, blue drop)
are *kept* — they're the negative class the safety loss needs. Use the
`--successes-only` flag if you want the clean BC warm-start set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.configs import EnvConfig, ExpertConfig, SceneConfig
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert
from sim.recorder import EpisodeRecorder
from sim.rollout import run_expert_episode

TASK_DESCRIPTION = (
    "pick up the blue cube, carry it low across the table avoiding every red cube, "
    "and place it on the green goal patch"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--mjcf", type=str, default=SceneConfig().mjcf_path,
                   help="path to SO-101 MJCF (see sim/assets/README.md)")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--n-red-cubes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--successes-only", action="store_true",
                   help="discard episodes that fail (red contact / drop / timeout)")
    p.add_argument("--use-videos", action="store_true", default=True)
    # ── A* corridor planner ──────────────────────────────────────────────────
    p.add_argument("--path-clearance-weight", type=float, default=1.0,
                   help="A* soft-clearance weight λ: 0 = plain BFS; >0 = cost-field A* that "
                        "bows corridors toward the center of free gaps. λ≈1 threads gaps; "
                        "λ≥3 tends to route around the field instead of through it.")
    p.add_argument("--path-clearance-pref", type=float, default=SceneConfig().path_clearance_pref,
                   help="extra standoff (m) beyond the hard clearance radius the A* planner "
                        "tries to keep (soft preference, not a hard constraint)")
    p.add_argument("--path-wall-field-sides", action="store_true",
                   help="hard-block lateral margins so A* weaves through the field rather "
                        "than routing around it (use with high --path-clearance-weight)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scene = SceneConfig(
        mjcf_path=args.mjcf,
        n_red_cubes=args.n_red_cubes,
        path_clearance_weight=args.path_clearance_weight,
        path_clearance_pref=args.path_clearance_pref,
        path_wall_field_sides=args.path_wall_field_sides,
    )
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=args.max_steps, seed=args.seed))
    # Reset once to lock in action_dim / state_dim.
    obs, info = env.reset(seed=args.seed)
    state_dim = obs["state"].shape[0]

    rec = EpisodeRecorder.create(
        repo_id=args.repo_id,
        root=args.root,
        n_red_cubes=args.n_red_cubes,
        image_size=scene.image_size,
        action_dim=env.action_dim,
        state_dim=state_dim,
        fps=env.cfg.fps,
        use_videos=args.use_videos,
    )

    expert = ScriptedExpert(env=env, cfg=ExpertConfig())
    grip_open = ExpertConfig().grip_open

    saved = 0
    discarded = 0
    discarded_flyover = 0
    for ep in range(args.n_episodes):
        # Shared expert path (choose_executed=None -> execute the expert: this IS
        # the demonstration). DAgger relabeling uses the exact same loop.
        stats = run_expert_episode(
            env=env, expert=expert, rec=rec, task=TASK_DESCRIPTION,
            seed=args.seed + ep, grip_open=grip_open,
        )
        # Always drop fly-overs (a demo of going *over* an obstacle is poison for
        # the stay-low task); drop failures too under --successes-only. fly_over
        # is belt-and-suspenders — at the low carry height it should never fire.
        fell_over = bool(stats.get("fly_over", False))
        if fell_over or (args.successes_only and not stats["success"]):
            rec.discard_episode()
            discarded += 1
            discarded_flyover += int(fell_over)
        else:
            n = rec.save_episode()
            saved += 1
            print(f"[ep {ep:03d}] frames={n} stats={stats}")
    env.close()
    rec.finalize()
    print(f"\nDone. saved={saved}  discarded={discarded} (fly_over={discarded_flyover})  "
          f"-> {args.root or '$HF_LEROBOT_HOME'}")


if __name__ == "__main__":
    main()
