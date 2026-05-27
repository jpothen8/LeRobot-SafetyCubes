"""Domain randomization helpers.

Applied between episodes (mostly via SceneConfig.workspace_*) and at reset time
via small in-place edits to the MjModel — cheap perturbations that don't require
re-compiling the model.

Stays intentionally conservative: aggressive DR is a tuning exercise once the
basic pipeline runs. Treat the gains here as starting points.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class DRConfig:
    randomize_lighting: bool = True
    randomize_camera_pose: bool = True
    randomize_table_color: bool = True
    randomize_red_shades: bool = True

    light_diffuse_low: float = 0.5
    light_diffuse_high: float = 1.0

    cam_pos_jitter: float = 0.03           # meters, each axis
    cam_lookat_jitter: float = 0.02

    table_rgb_low: tuple[float, float, float] = (0.70, 0.65, 0.55)
    table_rgb_high: tuple[float, float, float] = (0.95, 0.90, 0.85)


def apply_dr(model: mujoco.MjModel, rng: np.random.Generator, cfg: DRConfig | None = None) -> None:
    cfg = cfg or DRConfig()

    if cfg.randomize_lighting and model.nlight > 0:
        for i in range(model.nlight):
            d = rng.uniform(cfg.light_diffuse_low, cfg.light_diffuse_high)
            model.light_diffuse[i] = [d, d, d]

    if cfg.randomize_camera_pose and model.ncam > 0:
        for i in range(model.ncam):
            jitter = rng.uniform(-cfg.cam_pos_jitter, cfg.cam_pos_jitter, size=3)
            model.cam_pos[i] += jitter

    if cfg.randomize_table_color:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_plane")
        if gid >= 0:
            rgb = rng.uniform(cfg.table_rgb_low, cfg.table_rgb_high)
            model.geom_rgba[gid][:3] = rgb

    if cfg.randomize_red_shades:
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("red_"):
                # Stay clearly red so the policy still treats them as obstacles.
                model.geom_rgba[g][:3] = [
                    rng.uniform(0.65, 0.95),
                    rng.uniform(0.05, 0.20),
                    rng.uniform(0.05, 0.20),
                ]
