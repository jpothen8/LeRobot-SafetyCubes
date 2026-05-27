"""Interactive viewer for the SafeCubeEnv — sanity-check the scene without
recording. Drives the scripted expert (or holds still) so you can see the
arm move.

Example:

    uv run python -m sim.scripts.view_env --mjcf path/to/so_arm101.xml
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from sim.configs import EnvConfig, ExpertConfig, SceneConfig
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", type=str, default=SceneConfig().mjcf_path)
    p.add_argument("--n-red-cubes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["expert", "still"], default="expert")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = SafeCubeEnv(EnvConfig(
        scene=SceneConfig(mjcf_path=args.mjcf, n_red_cubes=args.n_red_cubes),
        seed=args.seed,
    ))
    obs, info = env.reset(seed=args.seed)
    expert = ScriptedExpert(env=env, cfg=ExpertConfig()) if args.mode == "expert" else None

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        last_print = time.time()
        while viewer.is_running():
            if expert is not None and not expert.done():
                action = expert.act(info)
            else:
                # Hold current joint positions; gripper open.
                action = list(env.joint_positions()) + [0.0]
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            if time.time() - last_print > 1.0:
                print(info["stats"])
                last_print = time.time()
            if terminated or truncated or (expert is not None and expert.done()):
                print("--- episode end:", info["stats"])
                obs, info = env.reset(seed=args.seed + 1)
                if expert is not None:
                    expert.reset()


if __name__ == "__main__":
    main()
