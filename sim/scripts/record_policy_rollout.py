"""Roll a trained safe_pi0 checkpoint in the sim and render videos.

Same MP4 layout as ``sim/scripts/record_rollout.py`` (per-camera + composite),
but the env is driven by :class:`sim.dagger.PolicyRollout` instead of the
scripted expert. Use this to *see* what a trained policy does — it's the
qualitative companion to ``sim.evaluate.evaluate``.

Example::

    uv run python -m sim.scripts.record_policy_rollout \\
        --checkpoint outputs/safe_pi0_bc/checkpoints/last/pretrained_model \\
        --dataset-repo-id local/safe-cube-mixed \\
        --dataset-root data/safe_cube_mixed \\
        --out videos/policy.mp4 --n-episodes 4 --seed 100
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from sim.configs import EnvConfig, SceneConfig
from sim.dagger import PolicyRollout
from sim.env import SafeCubeEnv
from sim.safety_geometry import safety_loss, sdf_clearance
from sim.scripts.collect_demos import TASK_DESCRIPTION


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="path to <output_dir>/checkpoints/last/pretrained_model")
    p.add_argument("--dataset-repo-id", required=True,
                   help="dataset for feature shapes + normalization stats")
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--task", type=str, default=TASK_DESCRIPTION)
    p.add_argument("--out", type=Path, required=True,
                   help="composite MP4 path; <out>_agent.mp4 / <out>_wrist.mp4 written alongside")
    p.add_argument("--mjcf", type=str, default=SceneConfig().mjcf_path)
    p.add_argument("--n-episodes", type=int, default=4)
    p.add_argument("--n-red-cubes", type=int, default=SceneConfig().n_red_cubes)
    p.add_argument("--seed", type=int, default=100,
                   help="base seed (each ep gets seed + ep) — offset from training data on purpose")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--render-size", type=int, nargs=2, default=[480, 640],
                   help="(H W) of each per-camera frame")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-composite", action="store_true")
    p.add_argument("--action-chunking", action="store_true",
                   help="use act_queued (execute full chunk before re-planning) instead of per-step re-inference")
    p.add_argument("--show-sdf-point", action="store_true",
                   help="draw a marker at the point the safety-loss SDF measures from, plus the "
                        "translucent ee_radius bounding sphere, and overlay the live obstacle safety loss")
    p.add_argument("--sdf-point-offset", type=float, nargs=3,
                   default=[-0.0062, -0.0115, 0.0017], metavar=("X", "Y", "Z"),
                   help="offset (m) from the ee site to the SDF query point, expressed in the ee "
                        "site's LOCAL frame so it tracks wrist rotation. Default lands the point on "
                        "the held-cube center (measured ee->cube offset). Pass 0 0 0 for the raw ee site.")
    return p.parse_args()


def _overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    img = frame.copy()
    y = 22
    for text, color in lines:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22
    return img


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    img = frame.copy()
    h = img.shape[0]
    cv2.putText(img, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _add_sdf_marker(renderer: mujoco.Renderer, pos: np.ndarray, ee_radius: float) -> None:
    """Draw the exact point the safety-loss SDF measures clearance from.

    This is the FK end-effector site position (``info['ee_pos']``) — the single
    query point the lateral box-SDF pulls toward the nearest red cube. We add two
    geoms to the already-updated scene (so they sit on top of the rendered world):
      * a small SOLID magenta sphere at the query point itself (magenta so it
        stands out against the gold SO-101 arm), and
      * a translucent cyan sphere of radius ``ee_radius`` — the gripper
        bounding sphere the SDF subtracts (``sdf_clearance``), i.e. the volume
        that actually has to stay clear of the cubes.
    Must be called AFTER ``renderer.update_scene`` (which resets ``ngeom``) and
    BEFORE ``renderer.render``.
    """
    scene = renderer.scene
    eye = np.eye(3, dtype=np.float64).reshape(-1)
    pos = np.asarray(pos, dtype=np.float64)
    # translucent ee_radius bounding sphere (what the SDF treats as "the gripper")
    if scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([ee_radius, 0.0, 0.0]), pos, eye,
            np.array([0.1, 0.9, 1.0, 0.22], dtype=np.float32))
        scene.ngeom += 1
    # solid query-point dot
    if scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.012, 0.0, 0.0]), pos, eye,
            np.array([1.0, 0.1, 1.0, 1.0], dtype=np.float32))
        scene.ngeom += 1


def _obstacle_loss(
    ee_pos: np.ndarray, cube_pos: np.ndarray, cube_half: np.ndarray,
    *, ee_radius: float, alpha: float, margin: float, lateral: bool,
) -> tuple[float, float]:
    """Instantaneous obstacle safety term at the live SDF query point.

    Same geometry the training loss uses (``sdf_clearance`` -> ``safety_loss``),
    but evaluated on the single realized query point instead of the predicted chunk.
    Returns ``(clearance_m, l_obstacle)`` where ``l_obstacle`` is the per-point
    ``softplus(-alpha*(clearance - margin))`` (the raw obstacle term, before the
    ``safety_weight * obstacle_weight`` scaling in the total loss).
    """
    ee = torch.as_tensor(ee_pos, dtype=torch.float32).reshape(1, 1, 3)
    centers = torch.as_tensor(cube_pos, dtype=torch.float32).reshape(1, -1, 3)
    halves = torch.as_tensor(cube_half, dtype=torch.float32).reshape(1, -1, 3)
    clearance = sdf_clearance(ee, centers, halves, ee_radius, lateral=lateral)  # (1, 1)
    loss = safety_loss(clearance, alpha=alpha, margin=margin)                   # scalar
    return float(clearance.item()), float(loss.item())


def _open_writer(path: Path, fps: int, size: tuple[int, int]) -> subprocess.Popen:
    W, H = size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _write(writer: subprocess.Popen, frame_bgr: np.ndarray) -> None:
    writer.stdin.write(frame_bgr.tobytes())


def _close(writer: subprocess.Popen) -> None:
    writer.stdin.close()
    writer.wait()


def main() -> None:
    args = parse_args()
    H, W = args.render_size
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path = out_path.with_name(out_path.stem + "_agent.mp4")
    wrist_path = out_path.with_name(out_path.stem + "_wrist.mp4")

    scene = SceneConfig(mjcf_path=args.mjcf, n_red_cubes=args.n_red_cubes)
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=args.max_steps, seed=args.seed))
    policy = PolicyRollout(
        checkpoint=args.checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        device=args.device,
    )

    # Pull the SDF/safety params straight off the loaded checkpoint's config so the
    # overlaid obstacle loss exactly matches what this policy was trained against.
    pcfg = policy.policy.config
    sdf = dict(
        ee_radius=getattr(pcfg, "ee_radius", 0.03),
        alpha=getattr(pcfg, "sdf_alpha", 50.0),
        margin=getattr(pcfg, "sdf_margin", 0.02),
        lateral=getattr(pcfg, "lateral_clearance", True),
    )
    obstacle_coeff = getattr(pcfg, "safety_weight", 1.0) * getattr(pcfg, "obstacle_weight", 2.0)
    # SDF query point = ee site + R_ee @ offset (offset in the ee-site LOCAL frame so it
    # tracks the wrist as it yaws). Default offset centers it on the held cube.
    sdf_offset = np.asarray(args.sdf_point_offset, dtype=np.float64)

    agent_writer = _open_writer(agent_path, args.fps, (W, H))
    wrist_writer = _open_writer(wrist_path, args.fps, (W, H))
    comp_writer = None if args.no_composite else _open_writer(out_path, args.fps, (2 * W, H))

    successes = contacts = ceilings = 0
    total_frames = 0
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        policy.reset()
        ee_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, scene.ee_site_name)
        agent_viz = mujoco.Renderer(env.model, height=H, width=W)
        wrist_viz = mujoco.Renderer(env.model, height=H, width=W)

        terminated = truncated = False
        for t in range(args.max_steps):
            action = policy.act_queued(obs, args.task) if args.action_chunking else policy.act(obs, args.task)
            obs, _, terminated, truncated, info = env.step(action)

            ee = info["ee_pos"]
            # SDF query point: ee site shifted by the local-frame offset (defaults onto
            # the held-cube center). R_ee rotates the local offset into the world frame.
            R_ee = env.data.site_xmat[ee_site_id].reshape(3, 3)
            sdf_pt = np.asarray(ee, dtype=np.float64) + R_ee @ sdf_offset

            agent_viz.update_scene(env.data, camera=scene.camera_name)
            if args.show_sdf_point:
                _add_sdf_marker(agent_viz, sdf_pt, sdf["ee_radius"])
            agent_frame = agent_viz.render()
            wrist_viz.update_scene(env.data, camera=scene.wrist_camera_name)
            if args.show_sdf_point:
                _add_sdf_marker(wrist_viz, sdf_pt, sdf["ee_radius"])
            wrist_frame = wrist_viz.render()

            stats = info["stats"]
            lines = [
                (f"ep {ep + 1}/{args.n_episodes}   t={t:3d}   POLICY", (255, 255, 255)),
                (f"ee ({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:+.2f})   grasp={info['grasped']}",
                 (200, 230, 255)),
                (f"min_clear={stats['min_clearance']:+.3f}m", (180, 255, 180)),
                (f"red={stats['red_contact']}  ceil={stats['ceiling_violation']}  "
                 f"drop={stats['blue_dropped']}  success={stats['success']}",
                 (255, 200, 120) if not any([stats['red_contact'], stats['ceiling_violation'],
                                             stats['blue_dropped']]) else (80, 80, 255)),
            ]
            if args.show_sdf_point:
                clear_xy, l_obs = _obstacle_loss(
                    sdf_pt, info["cube_positions"], info["cube_half_extents"],
                    ee_radius=sdf["ee_radius"], alpha=sdf["alpha"],
                    margin=sdf["margin"], lateral=sdf["lateral"])
                lines.append(
                    (f"SDF pt (magenta): clear_xy={clear_xy:+.3f}m  obstacle_loss={l_obs:.3f}"
                     f"  (x{obstacle_coeff:g}={obstacle_coeff * l_obs:.3f})", (255, 120, 255)))
            agent_annot = _label(_overlay(agent_frame, lines), "agentview")
            wrist_annot = _label(wrist_frame, "wrist_cam")

            _write(agent_writer, cv2.cvtColor(agent_annot, cv2.COLOR_RGB2BGR))
            _write(wrist_writer, cv2.cvtColor(wrist_annot, cv2.COLOR_RGB2BGR))
            if comp_writer is not None:
                _write(comp_writer, cv2.cvtColor(
                    np.concatenate([agent_annot, wrist_annot], axis=1), cv2.COLOR_RGB2BGR))

            total_frames += 1
            if terminated or truncated:
                # 0.5s freeze at episode end so the outcome is readable.
                for _ in range(args.fps // 2):
                    _write(agent_writer, cv2.cvtColor(agent_annot, cv2.COLOR_RGB2BGR))
                    _write(wrist_writer, cv2.cvtColor(wrist_annot, cv2.COLOR_RGB2BGR))
                    if comp_writer is not None:
                        _write(comp_writer, cv2.cvtColor(
                            np.concatenate([agent_annot, wrist_annot], axis=1), cv2.COLOR_RGB2BGR))
                    total_frames += 1
                break

        successes += int(stats["success"])
        contacts += int(stats["red_contact"])
        ceilings += int(stats["ceiling_violation"])
        print(f"[ep {ep}] stats={stats}")
        agent_viz.close()
        wrist_viz.close()

    _close(agent_writer)
    _close(wrist_writer)
    if comp_writer is not None:
        _close(comp_writer)
    env.close()
    print(f"\nWrote {total_frames} frames over {args.n_episodes} episodes "
          f"(success {successes}/{args.n_episodes}, red_contact {contacts}, ceiling {ceilings})")
    print(f"  agentview:  {agent_path}")
    print(f"  wrist_cam:  {wrist_path}")
    if comp_writer is not None:
        print(f"  composite:  {out_path}")


if __name__ == "__main__":
    main()
