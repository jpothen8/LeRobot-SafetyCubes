"""Extract and visualize episodes from a LeRobot dataset as a side-by-side composite video.

Decodes frames from the stored videos and overlays per-step data (ee_pos,
grasped, actor when available) on each frame.

Optional --ghost-base-seed enables a pure-expert ghost overlay on the agentview
panel: the visualizer re-simulates each episode under pure expert control
(same seed = same layout) and composites the expert robot as a cyan mask over
the DAgger frame wherever the two executions diverge.

Usage:
    python -m sim.scripts.visualize_dataset_episodes \
        --root data/safe_cube_dagger_sample \
        --out videos/dagger_r1_data_v2.mp4 \
        --n-episodes 6 \
        --ghost-base-seed 5300
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    p.add_argument("--out", type=Path, required=True, help="Output composite MP4 path")
    p.add_argument("--n-episodes", type=int, default=6)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--render-w", type=int, default=480, help="Width of each camera panel")
    p.add_argument("--render-h", type=int, default=480, help="Height of each camera panel")
    p.add_argument("--agentview-key", type=str, default="observation.images.agentview")
    p.add_argument("--wrist-key", type=str, default="observation.images.wrist")
    p.add_argument("--ghost-base-seed", type=int, default=None,
                   help="If set, run pure-expert re-simulation from this base seed "
                        "(ep N uses seed base+N) and overlay the expert robot as a "
                        "cyan mask on the agentview wherever the two trajectories differ.")
    p.add_argument("--n-red-cubes", type=int, default=8,
                   help="Must match the collection config (needed to build the ghost env).")
    return p.parse_args()


def load_episode_meta(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_data(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def video_path(root: Path, key: str, chunk_idx: int, file_idx: int) -> Path:
    return root / "videos" / key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def _actor_banner(frame: np.ndarray, actor_str: str) -> np.ndarray:
    """Draw a tall colored banner at the top of the frame showing who is driving.

    EXPERT → green tinted strip; POLICY → blue tinted strip.  The label text is
    large so it reads at a glance even in a small preview window.
    """
    img = frame.copy()
    banner_h = 36
    if actor_str == "EXPERT":
        bg = (20, 90, 20)      # dark green in BGR
        fg = (80, 255, 80)
    elif actor_str == "POLICY":
        bg = (80, 20, 20)      # dark blue in BGR
        fg = (80, 80, 255)
    else:
        bg = (40, 40, 40)
        fg = (180, 180, 180)

    cv2.rectangle(img, (0, 0), (frame.shape[1], banner_h), bg, -1)
    label = f">> {actor_str}"
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, fg, 2, cv2.LINE_AA)
    return img


def _overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    img = frame.copy()
    y = 58  # start below the actor banner
    for text, color in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        y += 20
    return img


def _label_bottom(frame: np.ndarray, text: str) -> np.ndarray:
    img = frame.copy()
    h = img.shape[0]
    cv2.putText(img, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _open_writer(path: Path, fps: int, w: int, h: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _apply_expert_ghost(
    ag_frame: np.ndarray,
    ghost_frame: np.ndarray,
    diff_thresh: int = 25,
    dilate_iters: int = 2,
) -> np.ndarray:
    """Overlay the expert ghost as a cyan mask where the two frames differ.

    Pixels where the expert execution diverges (robot arm in different position,
    cube moved differently) are highlighted in cyan (BGR 255,255,0) at 70%
    opacity over the DAgger frame. Where the frames agree the DAgger frame is
    unchanged, keeping the background clean.
    """
    diff = cv2.absdiff(ag_frame, ghost_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iters)

    # Tint expert frame cyan (zero out the red channel in BGR).
    ghost_cyan = ghost_frame.copy()
    ghost_cyan[:, :, 2] = 0

    # Blend only where mask is active.
    mask_3 = np.stack([mask, mask, mask], axis=2)
    blended = cv2.addWeighted(ag_frame, 0.30, ghost_cyan, 0.70, 0)
    return np.where(mask_3 > 0, blended, ag_frame).astype(np.uint8)


def render_episode(
    root: Path,
    ep_meta: pd.Series,
    ep_data: pd.DataFrame,
    ag_key: str,
    wr_key: str,
    out_path: Path,
    fps: int,
    W: int,
    H: int,
    ghost_env=None,
    ghost_expert=None,
    ghost_base_seed: int | None = None,
) -> None:
    ep_idx = int(ep_meta["episode_index"])
    has_actor = "privileged.actor" in ep_data.columns

    ag_src = video_path(root, ag_key,
                        int(ep_meta[f"videos/{ag_key}/chunk_index"]),
                        int(ep_meta[f"videos/{ag_key}/file_index"]))
    wr_src = video_path(root, wr_key,
                        int(ep_meta[f"videos/{wr_key}/chunk_index"]),
                        int(ep_meta[f"videos/{wr_key}/file_index"]))
    ag_t0 = float(ep_meta[f"videos/{ag_key}/from_timestamp"])
    wr_t0 = float(ep_meta[f"videos/{wr_key}/from_timestamp"])

    # Infer termination reason from last frame.
    last = ep_data.iloc[-1]
    last_blue = np.asarray(last["privileged.blue_cube_pos"])
    last_goal = np.asarray(last["privileged.goal_pos"])
    grasped_raw = last["privileged.grasped"]
    last_grasped = bool(float(np.squeeze(grasped_raw)) > 0.5)
    near_goal = float(np.linalg.norm(last_blue[:2] - last_goal[:2])) < 0.04
    ep_len = int(ep_meta["length"])
    if near_goal and not last_grasped:
        term_str, term_color = "SUCCESS", (50, 220, 50)
    elif not last_grasped and ep_len < 400:
        term_str, term_color = "DROPPED", (80, 80, 255)
    elif ep_len >= 400:
        term_str, term_color = "TRUNCATED", (180, 180, 80)
    else:
        term_str, term_color = "RED CONTACT", (50, 50, 255)

    # Initialize ghost simulation.
    ghost_frame: np.ndarray | None = None
    ghost_info: dict = {}
    ghost_done = False
    if ghost_env is not None and ghost_base_seed is not None:
        ghost_obs, ghost_info = ghost_env.reset(seed=ghost_base_seed + ep_idx)
        ghost_expert.reset()
        ghost_frame = cv2.resize(
            cv2.cvtColor(ghost_obs["image"], cv2.COLOR_RGB2BGR), (W, H))

    ag_cap = cv2.VideoCapture(str(ag_src))
    wr_cap = cv2.VideoCapture(str(wr_src))
    ag_cap.set(cv2.CAP_PROP_POS_MSEC, ag_t0 * 1000)
    wr_cap.set(cv2.CAP_PROP_POS_MSEC, wr_t0 * 1000)

    writer = _open_writer(out_path, fps, W * 2, H)

    ep_rows = ep_data.reset_index(drop=True)
    n_rows = len(ep_rows)
    for i, row in ep_rows.iterrows():
        ok_ag, ag_frame = ag_cap.read()
        ok_wr, wr_frame = wr_cap.read()
        if not ok_ag or not ok_wr:
            break

        ag_frame = cv2.resize(ag_frame, (W, H))
        wr_frame = cv2.resize(wr_frame, (W, H))

        # Advance ghost simulation (step i=0 used the reset obs already).
        if ghost_env is not None and i > 0 and not ghost_done:
            try:
                priv = ghost_info.get("privileged", {})
                action_g = ghost_expert.act(priv)
                ghost_obs_new, _, ghost_done, _, ghost_info = ghost_env.step(action_g)
                ghost_frame = cv2.resize(
                    cv2.cvtColor(ghost_obs_new["image"], cv2.COLOR_RGB2BGR), (W, H))
            except Exception:
                ghost_done = True  # hold last frame on any error

        # Composite expert ghost onto agentview.
        if ghost_frame is not None:
            ag_frame = _apply_expert_ghost(ag_frame, ghost_frame)

        ee = np.asarray(row["privileged.ee_pos"])
        grasped = bool(float(np.squeeze(row["privileged.grasped"])) > 0.5)

        if has_actor:
            actor_val = float(np.squeeze(row["privileged.actor"]))
            actor_str = "POLICY" if actor_val > 0.5 else "EXPERT"
        else:
            actor_str = "?"

        # Prominent banner at top of agentview showing who is driving.
        ag_frame = _actor_banner(ag_frame, actor_str)

        lines = [
            (f"ep {ep_idx}  step {int(row['frame_index'])}/{n_rows}", (220, 220, 220)),
            (f"ee ({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f})", (200, 230, 255)),
            (f"grasped={grasped}", (80, 255, 180) if grasped else (180, 180, 180)),
        ]
        if ghost_frame is not None:
            lines.append(("cyan=EXPERT ghost", (255, 255, 0)))

        # Show termination banner on last ~1 s of episode.
        if (n_rows - int(row["frame_index"])) <= fps:
            lines.append((f"END: {term_str}", term_color))

        ag_ann = _label_bottom(_overlay(ag_frame, lines), "agentview + expert ghost")
        wr_ann = _label_bottom(wr_frame, "wrist")

        composite = np.concatenate([ag_ann, wr_ann], axis=1)
        writer.stdin.write(composite.tobytes())

    ag_cap.release()
    wr_cap.release()
    writer.stdin.close()
    writer.wait()


def concat_videos(parts: list[Path], dst: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as flist:
        for p in parts:
            flist.write(f"file '{p.resolve()}'\n")
        flist_path = Path(flist.name)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(flist_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    flist_path.unlink()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    ep_meta = load_episode_meta(root)
    total = len(ep_meta)
    print(f"Dataset: {root.name}  total_episodes={total}")

    all_data = load_data(root)
    has_actor = "privileged.actor" in all_data.columns
    print(f"  actor tracking={'yes' if has_actor else 'no (old dataset)'}")

    # Build ghost env+expert if requested.
    ghost_env = ghost_expert = None
    if args.ghost_base_seed is not None:
        from sim.configs import EnvConfig, ExpertConfig, SceneConfig
        from sim.env import SafeCubeEnv
        from sim.expert import ScriptedExpert

        scene = SceneConfig(n_red_cubes=args.n_red_cubes)
        ghost_env = SafeCubeEnv(EnvConfig(
            scene=scene,
            max_episode_steps=500,
            seed=args.ghost_base_seed,
        ))
        ghost_expert = ScriptedExpert(env=ghost_env, cfg=ExpertConfig())
        print(f"  expert ghost: enabled (base_seed={args.ghost_base_seed})")
    else:
        print("  expert ghost: disabled (pass --ghost-base-seed N to enable)")

    indices = [int(i * (total - 1) / (args.n_episodes - 1)) for i in range(args.n_episodes)]
    selected_meta = ep_meta.iloc[indices].reset_index(drop=True)

    clips: list[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, row in selected_meta.iterrows():
            ep_idx = int(row["episode_index"])
            length = int(row["length"])
            print(f"  ep {ep_idx:4d}  ({length} steps)", end="  ", flush=True)

            ep_data = all_data[all_data["episode_index"] == ep_idx].copy()
            clip = tmp / f"ep{ep_idx:04d}.mp4"
            render_episode(
                root=root, ep_meta=row, ep_data=ep_data,
                ag_key=args.agentview_key, wr_key=args.wrist_key,
                out_path=clip, fps=args.fps, W=args.render_w, H=args.render_h,
                ghost_env=ghost_env, ghost_expert=ghost_expert,
                ghost_base_seed=args.ghost_base_seed,
            )
            clips.append(clip)
            print("done")

        if ghost_env is not None:
            ghost_env.close()

        print(f"Concatenating {len(clips)} episodes → {out}")
        concat_videos(clips, out)

    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
