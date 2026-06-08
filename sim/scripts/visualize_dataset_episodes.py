"""Extract and visualize episodes from a LeRobot dataset as a side-by-side composite video.

Usage:
    python -m sim.scripts.visualize_dataset_episodes \
        --root data/safe_cube_dagger \
        --out videos/dagger_r1_data.mp4 \
        --n-episodes 6
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    p.add_argument("--out", type=Path, required=True, help="Output composite MP4 path")
    p.add_argument("--n-episodes", type=int, default=6, help="Number of episodes to show")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--agentview-key", type=str, default="observation.images.agentview")
    p.add_argument("--wrist-key", type=str, default="observation.images.wrist")
    return p.parse_args()


def load_episode_meta(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def video_path(root: Path, key: str, chunk_idx: int, file_idx: int) -> Path:
    chunk = f"chunk-{chunk_idx:03d}"
    fname = f"file-{file_idx:03d}.mp4"
    return root / "videos" / key / chunk / fname


def extract_segment(
    src: Path, t_start: float, t_end: float, dst: Path, label: str, fps: int
) -> None:
    duration = t_end - t_start
    drawtext = (
        f"drawtext=text='{label}':fontcolor=white:fontsize=20"
        f":borderw=2:bordercolor=black:x=10:y=h-30"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        "-vf", drawtext,
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def hstack_videos(left: Path, right: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(left),
        "-i", str(right),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


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

    df = load_episode_meta(root)
    total = len(df)
    print(f"Dataset: {root.name}  total_episodes={total}")

    # Pick episodes spread across the dataset
    indices = [int(i * (total - 1) / (args.n_episodes - 1)) for i in range(args.n_episodes)]
    selected = df.iloc[indices].reset_index(drop=True)

    ag_key = args.agentview_key
    wr_key = args.wrist_key

    composites: list[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, row in selected.iterrows():
            ep_idx = int(row["episode_index"])
            length = int(row["length"])
            print(f"  ep {ep_idx:4d}  ({length} steps)", end="  ", flush=True)

            ag_src = video_path(root, ag_key, int(row[f"videos/{ag_key}/chunk_index"]),
                                int(row[f"videos/{ag_key}/file_index"]))
            wr_src = video_path(root, wr_key, int(row[f"videos/{wr_key}/chunk_index"]),
                                int(row[f"videos/{wr_key}/file_index"]))

            ag_t0 = float(row[f"videos/{ag_key}/from_timestamp"])
            ag_t1 = float(row[f"videos/{ag_key}/to_timestamp"])
            wr_t0 = float(row[f"videos/{wr_key}/from_timestamp"])
            wr_t1 = float(row[f"videos/{wr_key}/to_timestamp"])

            ag_clip = tmp / f"ep{ep_idx:04d}_agent.mp4"
            wr_clip = tmp / f"ep{ep_idx:04d}_wrist.mp4"
            comp = tmp / f"ep{ep_idx:04d}_comp.mp4"

            extract_segment(ag_src, ag_t0, ag_t1, ag_clip, f"ep{ep_idx} agentview", args.fps)
            extract_segment(wr_src, wr_t0, wr_t1, wr_clip, f"ep{ep_idx} wrist", args.fps)
            hstack_videos(ag_clip, wr_clip, comp)
            composites.append(comp)
            print("done")

        print(f"Concatenating {len(composites)} episodes → {out}")
        concat_videos(composites, out)

    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
