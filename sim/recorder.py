"""Episode recorder backed by LeRobotDataset.

Writes the policy-visible features (image, state, action) plus the privileged
keys the safety loss reads. Privileged per-cube arrays are flattened to 1D
(LeRobotDataset stores 1D float vectors most cleanly); the safety loss can
reshape via the known `n_red_cubes` at load time.

Usage:

    rec = EpisodeRecorder.create(
        repo_id="me/safe-cube-demos", root="data/demos",
        n_red_cubes=8, image_size=(224, 224),
        action_dim=env.action_dim, state_dim=obs["state"].shape[0],
        fps=env.cfg.fps,
    )
    rec.begin_episode(task=...)
    for ...:
        rec.add(obs, action, info)
    rec.save_episode()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class EpisodeRecorder:
    def __init__(self, dataset: LeRobotDataset, n_red_cubes: int) -> None:
        self.dataset = dataset
        self.n_red_cubes = n_red_cubes
        self._task: str | None = None
        self._frames_in_episode = 0

    # ----- factory -------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        root: str | Path | None,
        n_red_cubes: int,
        image_size: tuple[int, int],
        action_dim: int,
        state_dim: int,
        fps: int,
        use_videos: bool = True,
        robot_type: str = "so101_sim",
        camera_encoder: VideoEncoderConfig | None = None,
    ) -> "EpisodeRecorder":
        H, W = image_size
        n_cube_flat = n_red_cubes * 3

        features = {
            "observation.images.agentview": {
                "dtype": "video" if use_videos else "image",
                "shape": (H, W, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "video" if use_videos else "image",
                "shape": (H, W, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"j{i}" for i in range(state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"a{i}" for i in range(action_dim)],
            },
            # Privileged (loss-only). Flattened to 1D.
            "privileged.cube_positions": {
                "dtype": "float32",
                "shape": (n_cube_flat,),
                "names": [f"red{i}_{ax}" for i in range(n_red_cubes) for ax in "xyz"],
            },
            "privileged.cube_half_extents": {
                "dtype": "float32",
                "shape": (n_cube_flat,),
                "names": [f"red{i}_h{ax}" for i in range(n_red_cubes) for ax in "xyz"],
            },
            "privileged.blue_cube_pos": {
                "dtype": "float32", "shape": (3,), "names": ["x", "y", "z"],
            },
            "privileged.goal_pos": {
                "dtype": "float32", "shape": (3,), "names": ["x", "y", "z"],
            },
            "privileged.ee_pos": {
                "dtype": "float32", "shape": (3,), "names": ["x", "y", "z"],
            },
            "privileged.grasped": {
                "dtype": "float32", "shape": (1,), "names": ["grasped"],
            },
        }
        create_kwargs = dict(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos,
        )
        # Optional hardware (NVENC) / custom video encoder. Default (None) leaves
        # LeRobot's stock libsvtav1 software encoder in place.
        if camera_encoder is not None:
            create_kwargs["camera_encoder"] = camera_encoder
        dataset = LeRobotDataset.create(**create_kwargs)
        return cls(dataset=dataset, n_red_cubes=n_red_cubes)

    @classmethod
    def resume(
        cls,
        *,
        repo_id: str,
        root: str | Path | None,
        n_red_cubes: int,
    ) -> "EpisodeRecorder":
        """Append to an existing on-disk dataset (DAgger data aggregation).

        New episodes are added to the dataset already at ``root`` (e.g. a copy
        of the BC demo set), so training each round sees the demos *plus* every
        prior round's relabels. Feature schema must match what ``create`` wrote.
        """
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=root)
        return cls(dataset=dataset, n_red_cubes=n_red_cubes)

    # ----- episode lifecycle --------------------------------------------

    def begin_episode(self, task: str) -> None:
        self._task = task
        self._frames_in_episode = 0

    def add(self, obs: dict, action: np.ndarray, info: dict) -> None:
        if self._task is None:
            raise RuntimeError("begin_episode(task=...) must be called before add().")
        frame = {
            "observation.images.agentview": _to_uint8(obs["image"]),
            "observation.images.wrist": _to_uint8(obs["wrist_image"]),
            "observation.state": np.asarray(obs["state"], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "privileged.cube_positions": np.asarray(info["cube_positions"], dtype=np.float32).reshape(-1),
            "privileged.cube_half_extents": np.asarray(info["cube_half_extents"], dtype=np.float32).reshape(-1),
            "privileged.blue_cube_pos": np.asarray(info["blue_cube_pos"], dtype=np.float32),
            "privileged.goal_pos": np.asarray(info["goal_pos"], dtype=np.float32),
            "privileged.ee_pos": np.asarray(info["ee_pos"], dtype=np.float32),
            "privileged.grasped": np.array([float(info["grasped"])], dtype=np.float32),
            "task": self._task,
        }
        self.dataset.add_frame(frame)
        self._frames_in_episode += 1

    def save_episode(self) -> int:
        n = self._frames_in_episode
        if n == 0:
            self.dataset.clear_episode_buffer()
            return 0
        self.dataset.save_episode()
        self._frames_in_episode = 0
        return n

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()
        self._frames_in_episode = 0

    def finalize(self) -> None:
        # save_episode flushes metadata; kept as a forward-compat hook.
        pass


def _to_uint8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    a = np.clip(a, 0.0, 1.0) if a.max() <= 1.0 else np.clip(a, 0, 255)
    return a.astype(np.uint8)
