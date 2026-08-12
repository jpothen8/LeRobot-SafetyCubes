"""Render the safety loss' collision spheres as ghost balls on the arm.

The obstacle penalty is only as good as the geometry it queries. The softplus
form used **one** 3 cm sphere at ``gripper_frame_link`` — a frame that sits
~9.8 cm beyond ``gripper_link`` and carries no geometry at all — while the env
fires ``red_contact`` on any arm/finger geom or the held blue cube
(``env.py:481-491``). This script draws the replacement query set on top of a
real rollout so the covering can be checked by eye rather than by trusting a
number.

Each sphere is drawn as a translucent ball, perspective-scaled by depth and
**coloured by its hinge value**: green when clear, amber inside the margin, red
at contact. A correct set should hug the jaws and the held cube, and should stay
green for the whole of a clean expert episode.

Output is a composite agentview + wrist MP4, like the other viz scripts.

Usage::

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \\
        -m sim.scripts.viz_safety_spheres \\
        --out videos/safety_spheres.mp4 --n-episodes 3 --seed 2000
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from sim.configs import EnvConfig, ExpertConfig, SceneConfig
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert
from sim.safety_geometry import (
    DEFAULT_COLLISION_LINKS,
    DEFAULT_COLLISION_OFFSETS,
    DEFAULT_COLLISION_RADII,
    FKChain,
    collision_index,
    hinge,
    held_cube_clearance,
    multi_point_clearance,
    sphere_centers,
)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("videos/safety_spheres.mp4"))
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--n-red-cubes", type=int, default=SceneConfig().n_red_cubes)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--render-size", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    p.add_argument("--sdf-margin", type=float, default=0.005,
                   help="hinge margin (SafePI0Config.hinge_margin). Tight on "
                        "purpose: the spheres contain the arm meshes, so they "
                        "already supply the standoff.")
    p.add_argument("--ee-height-ceiling", type=float, default=SceneConfig().ee_height_ceiling)
    p.add_argument("--ceiling-buffer", type=float, default=0.005)
    p.add_argument("--ee-to-cube-z-offset", type=float, default=0.010)
    p.add_argument("--held-cube-radius", type=float, default=0.0127)
    p.add_argument("--held-cube-offset", type=float, nargs=3,
                   default=[-0.00452, -0.01232, 0.00178],
                   help="blue cube centre in the TCP frame while grasped (measured)")
    p.add_argument("--ball-alpha", type=float, default=0.45)
    p.add_argument("--urdf", type=str, default="sim/assets/so101/so101_new_calib.urdf")
    p.add_argument("--from-action", action="store_true",
                   help="place spheres at FK(action) instead of FK(measured joints). The "
                        "action is a joint TARGET, so this shows the ~26 mm control lag the "
                        "loss actually sees; the default lines the balls up with the render.")
    return p.parse_args()


def _open_writer(path: Path, fps: int, size: tuple[int, int]) -> subprocess.Popen:
    w, h = size
    return subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
         str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _project(p_world, cam_pos, cam_mat, fov_y_deg, h, w):
    """World -> (pixel, depth) for a camera given by its MuJoCo pose."""
    f_px = (h / 2) / np.tan(np.deg2rad(fov_y_deg) / 2)
    p_cam = cam_mat.T @ (np.asarray(p_world, dtype=np.float64) - cam_pos)
    depth = -p_cam[2]
    if depth <= 1e-6:
        return None, None
    return (int(f_px * p_cam[0] / depth + w / 2),
            int(-f_px * p_cam[1] / depth + h / 2)), (depth, f_px)


def _hinge_colour(v: float) -> tuple[int, int, int]:
    """Green (clear) -> amber (inside margin) -> red (contact). RGB."""
    v = float(np.clip(v, 0.0, 1.0))
    if v <= 0.5:
        t = v / 0.5
        return (int(60 + 195 * t), int(220 - 40 * t), 60)      # green -> amber
    t = (v - 0.5) / 0.5
    return (255, int(180 - 180 * t), 60)                        # amber -> red


def _draw_spheres(frame, centers, radii, hinges, cam_pos, cam_mat, fov, alpha):
    """Draw each sphere as a depth-sorted translucent ball."""
    h, w = frame.shape[:2]
    drawn = []
    for c, r, v in zip(centers, radii, hinges):
        pt, dep = _project(c, cam_pos, cam_mat, fov, h, w)
        if pt is None:
            continue
        depth, f_px = dep
        drawn.append((depth, pt, max(int(f_px * r / depth), 1), v))
    drawn.sort(key=lambda x: -x[0])                 # farthest first

    overlay = frame.copy()
    for _, pt, rad_px, v in drawn:
        cv2.circle(overlay, pt, rad_px, _hinge_colour(v), -1, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
    for _, pt, rad_px, v in drawn:                  # crisp rim on top
        cv2.circle(out, pt, rad_px, _hinge_colour(v), 1, cv2.LINE_AA)
    return out


def _overlay_text(frame, lines):
    img = frame.copy()
    y = 18
    for text, colour in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        y += 17
    return img


def main() -> None:
    args = parse_args()
    h, w = args.render_size
    args.out.parent.mkdir(parents=True, exist_ok=True)

    scene = SceneConfig(n_red_cubes=args.n_red_cubes)
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=args.max_steps, seed=args.seed))
    expert = ScriptedExpert(env=env, cfg=ExpertConfig())

    fk = FKChain(args.urdf, "gripper_frame_link", ARM_JOINTS)
    links, link_idx = collision_index(DEFAULT_COLLISION_LINKS)
    offsets = torch.tensor(DEFAULT_COLLISION_OFFSETS, dtype=torch.float32)
    radii = torch.tensor(DEFAULT_COLLISION_RADII, dtype=torch.float32)

    writer = _open_writer(args.out, args.fps, (2 * w, h))
    worst_clear, worst_hinge, frames = np.inf, 0.0, 0

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        expert.reset()
        agent_viz = mujoco.Renderer(env.model, height=h, width=w)
        wrist_viz = mujoco.Renderer(env.model, height=h, width=w)
        agent_cam = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, scene.camera_name)
        wrist_cam = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_CAMERA, scene.wrist_camera_name
        )

        for t in range(args.max_steps):
            if expert.done():
                break
            action = expert.act(info)
            obs, _, terminated, truncated, info = env.step(action)

            q = (np.asarray(action, dtype=np.float32) if args.from_action
                 else np.concatenate([env.joint_positions()[:5], [action[5]]]).astype(np.float32))
            mats = fk.fk_link_transforms(torch.from_numpy(q).reshape(1, 1, 6), links)
            pts = sphere_centers(mats, link_idx, offsets)[0, 0].numpy()      # (K, 3)

            reds = np.asarray(info["cube_positions"], dtype=np.float32).reshape(1, -1, 3)
            halves = np.full_like(reds, scene.red_cube_half)
            clear = multi_point_clearance(
                torch.from_numpy(pts).reshape(1, 1, -1, 3),
                torch.from_numpy(reds), torch.from_numpy(halves), radii,
            )[0, 0].numpy()                                                  # (K,)

            # The carried cube is part of the query set, but only while grasped.
            grasped = float(info["grasped"])
            tcp = fk.fk_link_transforms(
                torch.from_numpy(q).reshape(1, 1, 6), ["gripper_frame_link"])[:, :, 0]
            if grasped:
                cube_off = torch.tensor(args.held_cube_offset, dtype=torch.float32)
                cube_pt = (tcp[0, 0, :3, :3] @ cube_off + tcp[0, 0, :3, 3]).numpy()
                cube_cl = held_cube_clearance(
                    tcp, torch.from_numpy(reds), torch.from_numpy(halves),
                    offset=cube_off, radius=args.held_cube_radius, grasped=None,
                )[0, 0].numpy()
                pts = np.concatenate([pts, cube_pt.reshape(1, 3)])
                clear = np.concatenate([clear, cube_cl])
                draw_radii = np.concatenate([radii.numpy(), [args.held_cube_radius]])
            else:
                draw_radii = radii.numpy()
            hv = hinge(torch.from_numpy((args.sdf_margin - clear) / args.sdf_margin)).numpy()

            # Grasp-gated, exactly as the policy applies it (ceiling_grasped_only).
            # Ungated this reads ~17 during the reach, when there is no held cube
            # at all -- which is the false-fire the reform exists to remove.
            cube_z = float(info["ee_pos"][2]) - args.ee_to_cube_z_offset
            ceil_h = grasped * float(hinge(torch.tensor(
                (cube_z - (args.ee_height_ceiling - args.ceiling_buffer)) / args.ceiling_buffer
            )).item())
            worst_clear = min(worst_clear, float(clear.min()))
            worst_hinge = max(worst_hinge, float(hv.max()))

            agent_viz.update_scene(env.data, camera=scene.camera_name)
            wrist_viz.update_scene(env.data, camera=scene.wrist_camera_name)
            a_frame = _draw_spheres(
                agent_viz.render(), pts, draw_radii, hv,
                env.data.cam_xpos[agent_cam], env.data.cam_xmat[agent_cam].reshape(3, 3),
                float(env.model.cam_fovy[agent_cam]), args.ball_alpha,
            )
            w_frame = _draw_spheres(
                wrist_viz.render(), pts, draw_radii, hv,
                env.data.cam_xpos[wrist_cam], env.data.cam_xmat[wrist_cam].reshape(3, 3),
                float(env.model.cam_fovy[wrist_cam]), args.ball_alpha,
            )

            stats = info["stats"]
            ok = not stats["red_contact"]
            a_frame = _overlay_text(a_frame, [
                (f"ep {ep + 1}/{args.n_episodes}  t={t:3d}  K={len(pts)} spheres"
                 f"  [{'FK(action)' if args.from_action else 'FK(measured)'}]", (255, 255, 255)),
                (f"min sphere clearance {clear.min():+.4f} m   (margin {args.sdf_margin})",
                 (180, 255, 180) if clear.min() > args.sdf_margin else (120, 200, 255)),
                (f"L_obstacle(hinge) {hv.mean():.4f}   max sphere {hv.max():.3f}",
                 (180, 255, 180) if hv.max() == 0 else (120, 200, 255)),
                (f"held-cube z {cube_z:+.4f}  ceiling {args.ee_height_ceiling}  "
                 f"L_ceiling {ceil_h:.4f}  grasped={int(grasped)}",
                 (180, 255, 180) if ceil_h == 0 else (120, 200, 255)),
                (f"env red_contact={stats['red_contact']}  ceiling={stats['ceiling_violation']}",
                 (200, 230, 200) if ok else (80, 80, 255)),
            ])
            composite = np.concatenate([a_frame, w_frame], axis=1)
            writer.stdin.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR).tobytes())
            frames += 1

            if terminated or truncated:
                break

    writer.stdin.close()
    writer.wait()
    env.close()
    print(f"wrote {args.out}  ({frames} frames, {args.n_episodes} episodes)")
    print(f"worst sphere clearance over the run: {worst_clear:+.4f} m")
    print(f"worst per-sphere hinge value:        {worst_hinge:.4f}  "
          f"({'silent on this expert data' if worst_hinge == 0 else 'fired'})")


if __name__ == "__main__":
    main()
