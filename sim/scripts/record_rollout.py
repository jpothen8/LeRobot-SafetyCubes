"""Render N expert rollouts to an MP4 — the headless way to *see* one
demo-collection cycle without dealing with macOS / mjpython viewer threading.

Renders TWO camera streams every step:
  * `agentview` — fixed third-person camera (the policy's main observation)
  * `wrist_cam` — gripper-mounted camera (the policy's secondary observation
                  in a typical multi-cam VLA setup)

Both are saved separately, and a side-by-side composite is also written by
default. Frames are rendered off independent `mujoco.Renderer`s so the
policy observation (224x224 from `env.render()`) is unaffected.

Example:

    uv run python -m sim.scripts.record_rollout \\
        --out videos/demo.mp4 --n-episodes 2 --seed 0

Then `open videos/demo.mp4` (composite), or open the per-camera files
that get written alongside it (e.g. `videos/demo_wrist.mp4`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from sim.configs import EnvConfig, ExpertConfig, SceneConfig
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True,
                   help="output MP4 path (composite). Per-camera videos are "
                        "saved alongside as <out>_agent.mp4 and <out>_wrist.mp4.")
    p.add_argument("--mjcf", type=str, default=SceneConfig().mjcf_path)
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--n-red-cubes", type=int, default=SceneConfig().n_red_cubes)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--render-size", type=int, nargs=2, default=[480, 640],
                   help="(H W) of each per-camera frame")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-composite", action="store_true",
                   help="skip the side-by-side composite output")
    return p.parse_args()


def _overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    img = frame.copy()
    y = 22
    for text, color in lines:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
        y += 22
    return img


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.putText(img, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _open_writer(path: Path, fps: int, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    return writer


def record_episodes(args: argparse.Namespace) -> None:
    H, W = args.render_size
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agent_path = out_path.with_name(out_path.stem + "_agent.mp4")
    wrist_path = out_path.with_name(out_path.stem + "_wrist.mp4")
    composite_path = out_path

    scene = SceneConfig(mjcf_path=args.mjcf, n_red_cubes=args.n_red_cubes)
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=args.max_steps, seed=args.seed))
    expert = ScriptedExpert(env=env, cfg=ExpertConfig())

    agent_writer = _open_writer(agent_path, args.fps, (W, H))
    wrist_writer = _open_writer(wrist_path, args.fps, (W, H))
    comp_writer = None
    if not args.no_composite:
        comp_writer = _open_writer(composite_path, args.fps, (2 * W, H))

    total_frames = 0
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        expert.reset()
        agent_viz = mujoco.Renderer(env.model, height=H, width=W)
        wrist_viz = mujoco.Renderer(env.model, height=H, width=W)

        settle_budget = max(int(env.cfg.fps), env.cfg.success_dwell_steps + 5)
        settle_left = settle_budget
        for t in range(args.max_steps):
            if expert.done():
                if settle_left <= 0:
                    break
                settle_left -= 1
                action = np.concatenate([env.joint_positions(),
                                          np.array([ExpertConfig().grip_open])])
            else:
                action = expert.act(info)
            obs, _, terminated, truncated, info = env.step(action)

            agent_viz.update_scene(env.data, camera=scene.camera_name)
            agent_frame = agent_viz.render()
            wrist_viz.update_scene(env.data, camera=scene.wrist_camera_name)
            wrist_frame = wrist_viz.render()

            stats = info["stats"]
            lines = [
                (f"ep {ep+1}/{args.n_episodes}   t={t:3d}   phase={expert._phase.name}",
                 (255, 255, 255)),
                (f"ee  ({info['ee_pos'][0]:+.2f},{info['ee_pos'][1]:+.2f},{info['ee_pos'][2]:+.2f})",
                 (200, 230, 255)),
                (f"grasped={info['grasped']}   min_clear={stats['min_clearance']:.3f}m",
                 (180, 255, 180)),
                (f"red_contact={stats['red_contact']}  ceiling={stats['ceiling_violation']}  "
                 f"drop={stats['blue_dropped']}  success={stats['success']}",
                 (255, 200, 120) if not any([stats['red_contact'], stats['ceiling_violation'],
                                             stats['blue_dropped']]) else (80, 80, 255)),
            ]
            agent_annot = _label(_overlay(agent_frame, lines), "agentview")
            wrist_annot = _label(wrist_frame, "wrist_cam")

            agent_writer.write(cv2.cvtColor(agent_annot, cv2.COLOR_RGB2BGR))
            wrist_writer.write(cv2.cvtColor(wrist_annot, cv2.COLOR_RGB2BGR))
            if comp_writer is not None:
                composite = np.concatenate([agent_annot, wrist_annot], axis=1)
                comp_writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            total_frames += 1
            if terminated or truncated:
                hold = args.fps // 2
                for _ in range(hold):
                    agent_writer.write(cv2.cvtColor(agent_annot, cv2.COLOR_RGB2BGR))
                    wrist_writer.write(cv2.cvtColor(wrist_annot, cv2.COLOR_RGB2BGR))
                    if comp_writer is not None:
                        comp_writer.write(cv2.cvtColor(
                            np.concatenate([agent_annot, wrist_annot], axis=1),
                            cv2.COLOR_RGB2BGR))
                    total_frames += 1
                break

        agent_viz.close()
        wrist_viz.close()
        print(f"[ep {ep}] phase={expert._phase.name} stats={stats}")

    agent_writer.release()
    wrist_writer.release()
    if comp_writer is not None:
        comp_writer.release()
    env.close()
    print(f"\nWrote {total_frames} frames")
    print(f"  agentview:  {agent_path}")
    print(f"  wrist_cam:  {wrist_path}")
    if comp_writer is not None:
        print(f"  composite:  {composite_path}")


if __name__ == "__main__":
    record_episodes(parse_args())
