"""Render cleanup-DAgger episodes from the dataset with the A* carry path
projected onto the agentview frame.

Each cleanup episode starts mid-field (cube grasped, near a red cube). For each
episode we:
  1. Read the first frame's privileged data to get cube positions, blue cube
     start, and goal.
  2. Plan the A* path (λ=1.0, smooth-interior) from the blue cube start to the
     goal — the same planner used by replan_carry_from_current() during collection.
  3. Decode the stored agentview frames and project the A* path onto each one,
     colouring already-traversed waypoints in olive and remaining ones in yellow.
  4. Annotate with clearance-to-nearest-red and grasped state.

Output: composite MP4  (agentview-left, wrist-right), upscaled to --render-size.

Example:
    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \\
        -m sim.scripts.viz_cleanup_astar \\
        --root data/safe_cube_cleanup_v7.1 \\
        --out videos/cleanup_v7.1_astar.mp4 \\
        --n-episodes 8
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd

from sim.configs import EnvConfig, SceneConfig
from sim.env import SafeCubeEnv
from sim.scene import plan_carry_path


# ── camera params are baked in once (agentview is a fixed camera) ──────────
_CAM_POS: np.ndarray | None = None
_CAM_MAT: np.ndarray | None = None
_CAM_FOV_Y_DEG: float | None = None
_TABLE_Z: float = 0.0


def _init_camera(n_red_cubes: int) -> None:
    global _CAM_POS, _CAM_MAT, _CAM_FOV_Y_DEG, _TABLE_Z
    scene = SceneConfig(n_red_cubes=n_red_cubes)
    env = SafeCubeEnv(EnvConfig(scene=scene, max_episode_steps=10, seed=0))
    env.reset(seed=0)
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, scene.camera_name)
    _CAM_POS = env.data.cam_xpos[cam_id].copy()
    _CAM_MAT = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    _CAM_FOV_Y_DEG = float(env.model.cam_fovy[cam_id])
    _TABLE_Z = scene.table_z
    env.close()


def _project(p_world: np.ndarray, H: int, W: int) -> tuple[int, int] | None:
    """World → pixel, for agentview at render size (H, W)."""
    f_px = (H / 2) / np.tan(np.deg2rad(_CAM_FOV_Y_DEG) / 2)
    p_cam = _CAM_MAT.T @ (np.asarray(p_world, dtype=np.float64) - _CAM_POS)
    depth = -p_cam[2]
    if depth <= 0:
        return None
    return (int(f_px * p_cam[0] / depth + W / 2),
            int(-f_px * p_cam[1] / depth + H / 2))


def _draw_path(
    frame: np.ndarray,
    wps: list[np.ndarray],
    wp_idx: int,
    path_z: float,
) -> np.ndarray:
    """Project A* waypoints onto the frame and draw done/todo/target colouring."""
    H, W = frame.shape[:2]
    ppts = [_project(np.array([wx, wy, path_z]), H, W) for (wx, wy) in wps]

    C_DONE = (140, 150,  60)   # olive — already traversed
    C_TODO = (255, 230,  20)   # yellow — still ahead
    C_TGT  = (255, 130,   0)   # orange — current target node

    img = frame.copy()
    n = len(ppts)
    for i in range(n - 1):
        p0, p1 = ppts[i], ppts[i + 1]
        if p0 is None or p1 is None:
            continue
        cv2.line(img, p0, p1, C_DONE if i < wp_idx else C_TODO, 2, cv2.LINE_AA)

    for i, pt in enumerate(ppts):
        if pt is None:
            continue
        is_tgt  = (i == wp_idx)
        is_done = (i < wp_idx)
        if is_tgt:
            cv2.circle(img, pt, 7, (0, 0, 0), -1)
            cv2.circle(img, pt, 6, C_TGT, -1)
        elif is_done:
            cv2.circle(img, pt, 3, C_DONE, -1)
        else:
            cv2.circle(img, pt, 4, (0, 0, 0), -1)
            cv2.circle(img, pt, 3, C_TODO, -1)
    return img


def _draw_markers(
    frame: np.ndarray,
    blue_xy: np.ndarray,
    goal_xy: np.ndarray,
    red_centers: np.ndarray,
    cube_z: float,
) -> np.ndarray:
    """Draw blue cube (cyan dot), goal (star), red cube centres (faint circles)."""
    H, W = frame.shape[:2]
    img = frame.copy()
    # Red cube centres
    for cx, cy in red_centers[:, :2]:
        pt = _project(np.array([cx, cy, cube_z]), H, W)
        if pt is not None:
            cv2.circle(img, pt, 5, (40, 40, 200), 1, cv2.LINE_AA)
    # Goal
    pt = _project(np.array([goal_xy[0], goal_xy[1], cube_z]), H, W)
    if pt is not None:
        cv2.drawMarker(img, pt, (0, 160, 255), cv2.MARKER_STAR, 14, 2, cv2.LINE_AA)
    # Blue cube current position
    pt = _project(np.array([blue_xy[0], blue_xy[1], cube_z]), H, W)
    if pt is not None:
        cv2.circle(img, pt, 6, (0, 0, 0), -1)
        cv2.circle(img, pt, 5, (255, 200, 0), -1)
    return img


def _text(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]],
          y0: int = 22) -> np.ndarray:
    img = frame.copy()
    y = y0
    for text, color in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        y += 20
    return img


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    img = frame.copy()
    h = img.shape[0]
    cv2.putText(img, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _open_writer(path: Path, fps: int, W: int, H: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _clearance_to_nearest_red(
    pt_xy: np.ndarray, red_centers: np.ndarray, half: float
) -> float:
    """Approx XY clearance from pt_xy to nearest red cube surface."""
    dists = np.linalg.norm(red_centers[:, :2] - pt_xy, axis=1)
    return float(dists.min()) - half


def _advance_wp_idx(wp_idx: int, wps: list[np.ndarray], blue_xy: np.ndarray,
                    goal_xy: np.ndarray) -> int:
    """Advance wp_idx past waypoints the cube has already passed (dist-to-goal proxy)."""
    dist = np.linalg.norm(blue_xy - goal_xy)
    while (wp_idx < len(wps) - 1 and
           np.linalg.norm(np.asarray(wps[wp_idx]) - goal_xy) > dist + 0.02):
        wp_idx += 1
    return wp_idx


def render_cleanup_episodes(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    H = W = args.render_size   # square output

    _init_camera(args.n_red_cubes)
    scene_cfg = SceneConfig(n_red_cubes=args.n_red_cubes)
    cube_half = scene_cfg.red_cube_half
    path_z = _TABLE_Z + 0.003
    cube_z = _TABLE_Z + cube_half

    # ── load dataset ────────────────────────────────────────────────────────
    ep_meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True)
    data_files = sorted((root / "data").rglob("*.parquet"))
    data = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)

    n_total = int(ep_meta["episode_index"].max()) + 1
    n_eps = min(args.n_episodes, n_total)

    # Pick n_eps evenly spaced indices so we sample across the dataset.
    ep_indices = [int(round(i * (n_total - 1) / max(n_eps - 1, 1))) for i in range(n_eps)]

    ag_key = "observation.images.agentview"
    wr_key = "observation.images.wrist"
    writer = _open_writer(out, args.fps, W * 2, H)

    print(f"Rendering {n_eps} cleanup episodes from {root.name} → {out.name}")
    total_frames = 0

    for ep_num, ep_idx in enumerate(ep_indices):
        meta_row = ep_meta[ep_meta["episode_index"] == ep_idx].iloc[0]
        ep_data = data[data["episode_index"] == ep_idx].reset_index(drop=True)
        if len(ep_data) == 0:
            print(f"  ep {ep_idx}: no data rows, skipping")
            continue

        # ── per-episode plan ────────────────────────────────────────────────
        first = ep_data.iloc[0]
        cube_pos_flat = np.asarray(first["privileged.cube_positions"], dtype=np.float64)
        red_centers = cube_pos_flat.reshape(-1, 3)
        blue_start = np.asarray(first["privileged.blue_cube_pos"], dtype=np.float64)
        goal_pos = np.asarray(first["privileged.goal_pos"], dtype=np.float64)

        wps = plan_carry_path(
            SceneConfig(n_red_cubes=len(red_centers)),
            red_centers, blue_start[:2], goal_pos[:2],
        )
        if wps is None:
            wps = [blue_start[:2], goal_pos[:2]]   # fallback: straight line
            print(f"  ep {ep_idx}: A* returned None, using straight-line fallback")
        else:
            print(f"  ep {ep_idx}: A* path = {len(wps)} waypoints, "
                  f"blue_start=({blue_start[0]:.3f},{blue_start[1]:.3f})")

        # Find nearest red at start for the banner
        d_start = _clearance_to_nearest_red(blue_start[:2], red_centers, cube_half)

        ag_src = root / "videos" / ag_key / f"chunk-{int(meta_row[f'videos/{ag_key}/chunk_index']):03d}" / f"file-{int(meta_row[f'videos/{ag_key}/file_index']):03d}.mp4"
        wr_src = root / "videos" / wr_key / f"chunk-{int(meta_row[f'videos/{wr_key}/chunk_index']):03d}" / f"file-{int(meta_row[f'videos/{wr_key}/file_index']):03d}.mp4"
        ag_t0 = float(meta_row[f"videos/{ag_key}/from_timestamp"])
        wr_t0 = float(meta_row[f"videos/{wr_key}/from_timestamp"])

        ag_cap = cv2.VideoCapture(str(ag_src))
        wr_cap = cv2.VideoCapture(str(wr_src))
        ag_cap.set(cv2.CAP_PROP_POS_MSEC, ag_t0 * 1000)
        wr_cap.set(cv2.CAP_PROP_POS_MSEC, wr_t0 * 1000)

        wp_idx = 0
        ep_len = len(ep_data)

        for frame_i, row in ep_data.iterrows():
            ok_ag, ag_frame = ag_cap.read()
            ok_wr, wr_frame = wr_cap.read()
            if not ok_ag or not ok_wr:
                break

            ag_frame = cv2.resize(ag_frame, (W, H))
            wr_frame = cv2.resize(wr_frame, (W, H))

            blue_xy = np.asarray(row["privileged.blue_cube_pos"], dtype=np.float64)[:2]
            goal_xy = goal_pos[:2]
            grasped = bool(float(np.squeeze(np.asarray(row["privileged.grasped"]))) > 0.5)
            ee_xy = np.asarray(row["privileged.ee_pos"], dtype=np.float64)[:2]

            wp_idx = _advance_wp_idx(wp_idx, wps, blue_xy, goal_xy)

            clearance = _clearance_to_nearest_red(blue_xy, red_centers, cube_half)
            dist_goal = float(np.linalg.norm(blue_xy - goal_xy))

            # Draw path + markers on agentview
            ag_frame = _draw_path(ag_frame, wps, wp_idx, path_z)
            ag_frame = _draw_markers(ag_frame, blue_xy, goal_xy, red_centers, cube_z)

            t_ep = frame_i - ep_data.index[0]   # frame within episode
            lines_ag = [
                (f"cleanup ep {ep_idx} ({ep_num+1}/{n_eps})   t={t_ep}/{ep_len}",
                 (255, 255, 255)),
                (f"clear_start={d_start:+.3f}m   now={clearance:+.3f}m   dist_goal={dist_goal:.3f}m",
                 (180, 255, 180) if clearance > 0 else (80, 80, 255)),
                (f"grasped={grasped}   wp {wp_idx}/{len(wps)-1}",
                 (200, 230, 255)),
            ]
            ag_frame = _text(ag_frame, lines_ag)
            ag_frame = _label(ag_frame, "agentview + A* path (λ=1.0)")

            lines_wr = [
                (f"ep {ep_idx}  t={t_ep}", (255, 255, 255)),
                (f"clear={clearance:+.3f}m", (180, 255, 180) if clearance > 0 else (80, 80, 255)),
            ]
            wr_frame = _text(wr_frame, lines_wr)
            wr_frame = _label(wr_frame, "wrist_cam")

            composite = np.concatenate([ag_frame, wr_frame], axis=1)
            writer.stdin.write(composite.tobytes())
            total_frames += 1

        # Half-second freeze on last frame
        if ok_ag and ok_wr:
            for _ in range(args.fps // 2):
                writer.stdin.write(composite.tobytes())
                total_frames += 1

        ag_cap.release()
        wr_cap.release()
        print(f"    ep {ep_idx}: {ep_len} frames written")

    writer.stdin.close()
    writer.wait()
    print(f"\nWrote {total_frames} frames → {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/safe_cube_cleanup_v7.1"),
                   help="Cleanup dataset root")
    p.add_argument("--out", type=Path, default=Path("videos/cleanup_v7.1_astar.mp4"),
                   help="Output composite MP4")
    p.add_argument("--n-episodes", type=int, default=8,
                   help="Number of episodes to render (evenly sampled from dataset)")
    p.add_argument("--n-red-cubes", type=int, default=SceneConfig().n_red_cubes)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--render-size", type=int, default=480,
                   help="Output frame size per panel (square)")
    return p.parse_args()


if __name__ == "__main__":
    render_cleanup_episodes(parse_args())
