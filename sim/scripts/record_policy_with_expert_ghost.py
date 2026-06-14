"""Roll a trained safe_pi0 checkpoint and overlay a scripted-expert "ghost".

Same as ``sim/scripts/record_policy_rollout.py`` (per-camera + composite MP4s,
identical text overlay), but the **agentview** stream additionally shows what the
scripted expert would have done in the *exact same layout*, drawn as a
semi-transparent coloured arm "ghost".

How the ghost works (and what it is NOT)
----------------------------------------
A *second* :class:`SafeCubeEnv` is created and reset with the **same seed** as
the policy env, so it gets a pixel-identical layout (cube positions, blue spawn,
goal, BFS carry waypoints all derive deterministically from the seed). That
second env is driven by the :class:`ScriptedExpert` and stepped in lockstep with
the policy env.

Each frame the agentview is composited as::

    policy frame (the real arm)
    + tinted silhouette of the expert env's arm
    + 2-D BFS carry-path drawn on the table plane

The expert arm silhouette is extracted via MuJoCo **segmentation rendering** of
the expert env (mask = its arm geoms), then alpha-blended onto the policy frame.

The ghost is a **pure visualization artifact**:
  * It lives only in the *output video*. The policy's observations come from the
    policy env alone, so the policy never "sees" the expert arm.
  * The two envs share no physics — the expert runs its own independent episode.
  * Only the third-person ``agentview`` gets the ghost. The ``wrist`` stream is
    the policy's own wrist camera, untouched (the expert arm does not physically
    exist in the policy env, so it could never appear there).

The episode length is driven by the **policy** episode (so the video matches a
plain policy rollout); if the expert finishes first its arm freezes at the final
pose, if the policy finishes first the expert is simply cut off.

Example (reproduces the bc_v6_20k_chunked layouts, with the ghost)::

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \\
        -m sim.scripts.record_policy_with_expert_ghost \\
        --checkpoint outputs/safe_pi0_bc_v6/checkpoints/last/pretrained_model \\
        --dataset-repo-id local/safe-cube-mixed --dataset-root data/safe_cube_v6 \\
        --out videos/bc_v6_20k_ghost.mp4 --n-episodes 8 --seed 2000 --action-chunking
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np

from sim.configs import EnvConfig, ExpertConfig, SceneConfig
from sim.dagger import PolicyRollout
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert
from sim.scene import plan_carry_path
from sim.scripts.collect_demos import TASK_DESCRIPTION

# Ghost colour (RGB). Frames are RGB until written, so specify in RGB. A bright
# cyan reads clearly against the dark SO-101 arm and the table.
GHOST_TINT_RGB = np.array([90, 230, 255], dtype=np.float32)


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
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--n-red-cubes", type=int, default=SceneConfig().n_red_cubes)
    p.add_argument("--seed", type=int, default=2000,
                   help="base seed (each ep gets seed + ep) — keep >= collection N (held out)")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--render-size", type=int, nargs=2, default=[480, 640],
                   help="(H W) of each per-camera frame")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-composite", action="store_true")
    p.add_argument("--action-chunking", action="store_true",
                   help="use act_queued (execute full chunk before re-planning) instead of per-step re-inference")
    p.add_argument("--ghost-alpha", type=float, default=0.6,
                   help="opacity of the expert ghost silhouette (0..1)")
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


def _composite_ghost(
    policy_rgb: np.ndarray,
    expert_seg: np.ndarray,
    expert_rgb: np.ndarray,
    arm_ids: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Alpha-blend the expert env's arm (a tinted silhouette) onto the policy
    frame. ``expert_seg`` is the (H, W, 2) segmentation render of the expert env;
    channel 0 holds geom ids. The silhouette is the expert env's arm geoms, tinted
    GHOST_TINT_RGB and shaded by the expert arm's own luminance so its structure
    stays readable. A 1px solid outline makes it legible where it overlaps the
    real arm."""
    mask = np.isin(expert_seg[..., 0], arm_ids)
    if not mask.any():
        return policy_rgb

    out = policy_rgb.astype(np.float32)
    # Shade the flat tint by the expert arm's luminance so the ghost keeps a
    # sense of 3D form instead of being a flat blob.
    lum = expert_rgb.astype(np.float32).mean(axis=2, keepdims=True) / 255.0
    ghost = GHOST_TINT_RGB.reshape(1, 1, 3) * (0.35 + 0.65 * lum)

    m = mask[..., None].astype(np.float32) * alpha
    out = out * (1.0 - m) + ghost * m

    # Crisp outline (mask minus its erosion) at full tint.
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    edge = (mask_u8 - eroded).astype(bool)
    out[edge] = GHOST_TINT_RGB

    return np.clip(out, 0, 255).astype(np.uint8)


def _project_point(
    cam_pos: np.ndarray, cam_mat: np.ndarray,
    f_px: float, H: int, W: int,
    p_world: np.ndarray,
) -> tuple[int, int] | None:
    """Project a 3-D world point to image pixel coordinates.

    ``cam_mat`` rows are the camera's local axes expressed in world frame
    (MuJoCo's ``cam_xmat`` convention), so world→camera is ``cam_mat.T``.
    The camera looks along ``-z_cam`` (OpenGL convention), giving
    ``depth = -p_cam[2]``.  Returns ``None`` for points behind the camera.
    """
    p_cam = cam_mat.T @ (p_world - cam_pos)
    depth = -p_cam[2]
    if depth <= 0:
        return None
    return (int(f_px * p_cam[0] / depth + W / 2),
            int(-f_px * p_cam[1] / depth + H / 2))


def _draw_bfs_path(
    frame: np.ndarray,
    projected_wps: list[tuple[int, int] | None],
    expert_phase: str,
    wp_idx: int,
) -> np.ndarray:
    """Overlay the expert's BFS carry-path on ``frame`` (RGB in-place copy).

    The path is a polyline of BFS waypoints projected to the table plane and
    drawn directly on the third-person view. Split into already-traversed
    (olive/dim) and remaining (bright yellow) segments, with the current BFS
    target node highlighted as an orange dot.

    ``expert_phase`` is one of the :class:`~sim.expert._Phase` name strings;
    ``wp_idx`` is ``expert._wp_idx`` (only meaningful during CARRY).
    """
    n = len(projected_wps)
    if n == 0:
        return frame
    img = frame.copy()

    # Determine how many waypoints have been passed.
    if expert_phase in ("DESCEND2", "OPEN", "DONE"):
        split = n        # all traversed
    elif expert_phase == "CARRY":
        split = wp_idx   # 0..split-1 done; split is current target
    else:
        split = 0        # APPROACH/DESCEND/CLOSE/LIFT: nothing traversed yet

    C_DONE = (140, 150, 60)   # dim olive — already traversed
    C_TODO = (255, 230, 20)   # bright yellow — still ahead
    C_TGT  = (255, 130,  0)   # orange — current BFS target node

    # Polyline segments, colour-coded by traversal status.
    for i in range(n - 1):
        p0, p1 = projected_wps[i], projected_wps[i + 1]
        if p0 is None or p1 is None:
            continue
        cv2.line(img, p0, p1, C_DONE if i < split else C_TODO, 2, cv2.LINE_AA)

    # Node dots.
    for i, pt in enumerate(projected_wps):
        if pt is None:
            continue
        is_tgt  = (expert_phase == "CARRY" and i == wp_idx)
        is_done = (i < split)
        if is_tgt:
            cv2.circle(img, pt, 7, (0, 0, 0), -1)
            cv2.circle(img, pt, 6, C_TGT, -1)
        elif is_done:
            cv2.circle(img, pt, 3, C_DONE, -1)
        else:
            cv2.circle(img, pt, 4, (0, 0, 0), -1)
            cv2.circle(img, pt, 3, C_TODO, -1)

    return img


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
    # Policy env (the real arm) and an identically-seeded expert env (ghost source).
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=args.max_steps, seed=args.seed))
    ghost_env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=args.max_steps, seed=args.seed))
    expert = ScriptedExpert(env=ghost_env, cfg=ExpertConfig())
    grip_open = ExpertConfig().grip_open

    policy = PolicyRollout(
        checkpoint=args.checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        device=args.device,
    )

    agent_writer = _open_writer(agent_path, args.fps, (W, H))
    wrist_writer = _open_writer(wrist_path, args.fps, (W, H))
    comp_writer = None if args.no_composite else _open_writer(out_path, args.fps, (2 * W, H))

    successes = contacts = ceilings = 0
    total_frames = 0
    wrist_cam_id = wrist_f_px = None  # computed after first reset
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        policy.reset()
        # Same seed -> identical layout. The expert drives this env independently.
        g_obs, g_info = ghost_env.reset(seed=args.seed + ep)
        expert.reset()
        arm_ids = np.fromiter(ghost_env._arm_geom_ids, dtype=np.int64)

        # BFS path for the policy's actual trajectory — replanned when the
        # policy carries the cube off the original corridor.
        policy_carry_wps: list[tuple[float, float]] = list(env.layout.carry_waypoints or [])
        policy_wp_idx: int = 0

        # Wrist camera intrinsics — FOV is fixed; pose is read per-step since the
        # camera is body-attached and moves with the arm. Computed after first reset
        # because env.model is None until reset() is called.
        if wrist_cam_id is None:
            wrist_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                              scene.wrist_camera_name)
            wrist_f_px = (H / 2) / np.tan(np.deg2rad(env.model.cam_fovy[wrist_cam_id]) / 2)

        agent_viz = mujoco.Renderer(env.model, height=H, width=W)
        wrist_viz = mujoco.Renderer(env.model, height=H, width=W)
        ghost_viz = mujoco.Renderer(ghost_env.model, height=H, width=W)

        # Pre-compute camera intrinsics/extrinsics for this episode.  The
        # agentview camera is fixed (not body-attached), so these are constant;
        # waypoint projection itself is done per-step so replans are reflected.
        mujoco.mj_forward(ghost_env.model, ghost_env.data)
        cam_id = mujoco.mj_name2id(ghost_env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                    scene.camera_name)
        cam_pos = ghost_env.data.cam_xpos[cam_id].copy()
        cam_mat = ghost_env.data.cam_xmat[cam_id].reshape(3, 3)
        fov_y_rad = np.deg2rad(ghost_env.model.cam_fovy[cam_id])
        f_px = (H / 2) / np.tan(fov_y_rad / 2)
        path_z = scene.table_z + 0.003  # 3 mm above table so segments are visible

        # Expert post-DONE settle budget (mirrors sim.rollout.run_expert_episode):
        # hold pose with the gripper open so its final pose is stable, then freeze.
        settle_budget = max(int(ghost_env.cfg.fps), ghost_env.cfg.success_dwell_steps + 5)
        settle_left = settle_budget
        expert_frozen = False

        terminated = truncated = False
        for t in range(args.max_steps):
            # --- policy env (the real arm) ---
            action = policy.act_queued(obs, args.task) if args.action_chunking else policy.act(obs, args.task)
            obs, _, terminated, truncated, info = env.step(action)

            # --- BFS path replan based on policy's actual cube position ---
            if info["grasped"] and policy_carry_wps:
                blue_xy = np.asarray(info["blue_cube_pos"][:2], dtype=np.float64)
                wps_arr = np.asarray(policy_carry_wps, dtype=np.float64)
                # Advance wp_idx past waypoints the cube has already reached.
                goal_xy = np.asarray(info["goal_pos"][:2], dtype=np.float64)
                dist_cube_goal = float(np.linalg.norm(blue_xy - goal_xy))
                while (policy_wp_idx < len(policy_carry_wps) - 1 and
                       np.linalg.norm(np.asarray(policy_carry_wps[policy_wp_idx]) - goal_xy)
                       > dist_cube_goal + 0.02):
                    policy_wp_idx += 1
                # Replan if the cube has drifted off the corridor.
                closest_dist = float(np.linalg.norm(wps_arr - blue_xy, axis=1).min())
                if closest_dist > ExpertConfig().replan_offpath_threshold:
                    reds = np.asarray(info["cube_positions"], dtype=np.float64).reshape(-1, 3)
                    new_path = plan_carry_path(scene, reds, blue_xy, goal_xy)
                    if new_path:
                        policy_carry_wps = new_path
                        policy_wp_idx = 0

            # --- expert ghost env (lockstep, independent physics) ---
            if not expert_frozen:
                if expert.done():
                    if settle_left <= 0:
                        expert_frozen = True
                    else:
                        settle_left -= 1
                        g_action = np.concatenate([ghost_env.joint_positions(), [grip_open]])
                        g_obs, _, _, _, g_info = ghost_env.step(g_action)
                else:
                    g_action = expert.act(g_info)
                    g_obs, _, _, _, g_info = ghost_env.step(g_action)

            # --- render: policy agentview + wrist ---
            agent_viz.update_scene(env.data, camera=scene.camera_name)
            agent_frame = agent_viz.render()
            wrist_viz.update_scene(env.data, camera=scene.wrist_camera_name)
            wrist_frame = wrist_viz.render()

            # --- render: expert arm silhouette (segmentation) + colour ---
            ghost_viz.enable_segmentation_rendering()
            ghost_viz.update_scene(ghost_env.data, camera=scene.camera_name)
            expert_seg = ghost_viz.render()
            ghost_viz.disable_segmentation_rendering()
            ghost_viz.update_scene(ghost_env.data, camera=scene.camera_name)
            expert_rgb = ghost_viz.render()

            agent_ghosted = _composite_ghost(agent_frame, expert_seg, expert_rgb,
                                              arm_ids, args.ghost_alpha)

            # BFS path overlay — tracks the policy's actual blue cube position and
            # replans when it drifts off the original corridor (policy_carry_wps above).
            projected_wps = [
                _project_point(cam_pos, cam_mat, f_px, H, W,
                               np.array([wx, wy, path_z]))
                for (wx, wy) in policy_carry_wps
            ]
            g_phase = "DONE" if (expert_frozen or expert.done()) else expert._phase.name
            agent_ghosted = _draw_bfs_path(agent_ghosted, projected_wps,
                                            g_phase, policy_wp_idx)

            # BFS path on the wrist view.  The wrist camera is body-attached so
            # its pose (cam_xpos/cam_xmat) is read from env.data every step.
            # This frame is used only for the output video — the policy's own
            # observations come from env.step() at 224×224 and are never touched.
            wrist_cam_pos = env.data.cam_xpos[wrist_cam_id].copy()
            wrist_cam_mat = env.data.cam_xmat[wrist_cam_id].reshape(3, 3)
            wrist_projected_wps = [
                _project_point(wrist_cam_pos, wrist_cam_mat, wrist_f_px, H, W,
                               np.array([wx, wy, path_z]))
                for (wx, wy) in policy_carry_wps
            ]
            wrist_frame = _draw_bfs_path(wrist_frame, wrist_projected_wps,
                                         g_phase, policy_wp_idx)

            stats = info["stats"]
            ee = info["ee_pos"]
            lines = [
                (f"ep {ep + 1}/{args.n_episodes}   t={t:3d}   POLICY (white) vs EXPERT (cyan ghost)",
                 (255, 255, 255)),
                (f"ee ({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:+.2f})   grasp={info['grasped']}   expert={g_phase}",
                 (200, 230, 255)),
                (f"min_clear={stats['min_clearance']:+.3f}m", (180, 255, 180)),
                (f"red={stats['red_contact']}  ceil={stats['ceiling_violation']}  "
                 f"drop={stats['blue_dropped']}  success={stats['success']}",
                 (255, 200, 120) if not any([stats['red_contact'], stats['ceiling_violation'],
                                             stats['blue_dropped']]) else (80, 80, 255)),
            ]
            agent_annot = _label(_overlay(agent_ghosted, lines), "agentview (+ ghost + BFS path)")
            wrist_annot = _label(wrist_frame, "wrist_cam (policy + BFS path)")

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
        ghost_viz.close()

    _close(agent_writer)
    _close(wrist_writer)
    if comp_writer is not None:
        _close(comp_writer)
    env.close()
    ghost_env.close()
    print(f"\nWrote {total_frames} frames over {args.n_episodes} episodes "
          f"(success {successes}/{args.n_episodes}, red_contact {contacts}, ceiling {ceilings})")
    print(f"  agentview:  {agent_path}")
    print(f"  wrist_cam:  {wrist_path}")
    if comp_writer is not None:
        print(f"  composite:  {out_path}")


if __name__ == "__main__":
    main()
