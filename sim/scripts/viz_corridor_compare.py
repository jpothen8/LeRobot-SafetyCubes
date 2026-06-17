"""Top-down corridor-planner comparison: BFS vs A* λ=1.0 / 1.5 / 3.0.

Two sections:
  Rows 1-3  — 15 normal layouts (start = blue spawn position)
  Row 4     — 5 DAgger-style layouts (start = a position just outside a red
               cube's clearance ring, simulating mid-carry policy drift)

Run headless:
    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \
        -m sim.scripts.viz_corridor_compare
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

os.environ.setdefault("MUJOCO_GL", "egl")

from sim.configs import EnvConfig, SceneConfig
from sim.env import SafeCubeEnv
from sim.scene import plan_carry_path

N_NORMAL  = 15
N_DAGGER  = 5
N_COLS    = 5
N_ROWS    = (N_NORMAL // N_COLS) + 1   # 3 normal + 1 DAgger = 4

PLANNERS = [
    dict(label="BFS (λ=0)",  weight=0.0, color="#4477AA"),
    dict(label="A* λ=1.0",   weight=1.0, color="#228833"),
    dict(label="A* λ=1.5",   weight=1.5, color="#EE7733"),
    dict(label="A* λ=3.0",   weight=3.0, color="#CC3311"),
]

BASE_CFG = SceneConfig(n_red_cubes=8, path_clearance_weight=0.0)
CLEARANCE = BASE_CFG.path_clearance_radius


def _load_layout(seed: int):
    env = SafeCubeEnv(EnvConfig(scene=BASE_CFG, max_episode_steps=500, seed=seed))
    env.reset(seed=seed)
    layout = env.layout
    env.close()
    return layout


def _paths_from(layout, start_xy: np.ndarray,
                inside: bool = False) -> list[list[np.ndarray] | None]:
    red_centers = np.array([rc.pos for rc in layout.red_cubes])
    goal_xy = layout.goal_xy
    planner = _plan_from_inside if inside else plan_carry_path
    return [
        planner(
            SceneConfig(n_red_cubes=len(layout.red_cubes), path_clearance_weight=p["weight"]),
            red_centers, start_xy, goal_xy,
        )
        for p in PLANNERS
    ]


GATE_MARGIN = 0.025   # new default, matches collect_dagger_cleanup.py
CUBE_HALF   = BASE_CFG.red_cube_half   # ≈ 0.0127 m


def _find_near_cube_start(layout) -> tuple[np.ndarray | None, int | None]:
    """Return an XY *inside* the clearance boundary of one red cube (past the
    gate trigger), still clear of every other cube's hard surface.
    Annulus: (CUBE_HALF + GATE_MARGIN, CLEARANCE) from cube center."""
    centers = np.array([rc.pos[:2] for rc in layout.red_cubes])
    r_inner = CUBE_HALF + GATE_MARGIN + 0.002   # just past gate trigger
    r_outer = CLEARANCE - 0.003                  # just inside clearance boundary
    if r_inner >= r_outer:
        return None, None

    scores = np.abs(centers[:, 0] - 0.25) + np.abs(centers[:, 1])
    for ci in np.argsort(scores):
        cx, cy = centers[ci]
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            r = (r_inner + r_outer) / 2
            px, py = cx + r * np.cos(angle), cy + r * np.sin(angle)
            p = np.array([px, py])
            if not (0.09 <= px <= 0.41 and -0.17 <= py <= 0.17):
                continue
            other = np.delete(centers, ci, axis=0)
            if len(other) and np.any(np.linalg.norm(other - p, axis=1) < CUBE_HALF):
                continue
            return p, int(ci)
    return None, None


def _plan_from_inside(cfg: SceneConfig, red_centers, start_xy: np.ndarray,
                      goal_xy: np.ndarray) -> list[np.ndarray] | None:
    """plan_carry_path with fallback for starts inside the clearance zone.

    If start_xy is blocked, step toward the goal in grid-resolution increments
    until reaching an unblocked cell, plan from there, then prepend start_xy
    so the path shows where the policy actually was.
    """
    wps = plan_carry_path(cfg, red_centers, start_xy, goal_xy)
    if wps is not None:
        return wps

    goal = np.asarray(goal_xy, dtype=np.float64)
    direction = goal - start_xy
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    step = cfg.path_grid_res * direction / norm

    candidate = start_xy.copy().astype(np.float64)
    for _ in range(200):
        candidate += step
        wps = plan_carry_path(cfg, red_centers, candidate, goal_xy)
        if wps is not None:
            return [np.asarray(start_xy, dtype=np.float64)] + wps
    return None


def _draw(ax, layout, paths, start_xy: np.ndarray, title: str,
          dagger: bool = False) -> None:
    if dagger:
        ax.set_facecolor("#FFF8EE")

    h = BASE_CFG.red_cube_half
    centers = np.array([rc.pos[:2] for rc in layout.red_cubes])

    for rc in layout.red_cubes:
        cx, cy = rc.pos[0], rc.pos[1]
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - h, cy - h), 2 * h, 2 * h,
            boxstyle="square,pad=0",
            linewidth=0.8, edgecolor="#991111", facecolor="#FFCCCC", zorder=2,
        ))
        ax.add_patch(plt.Circle(
            (cx, cy), CLEARANCE,
            color="#FF8888", fill=False, lw=0.5, ls="--", alpha=0.4, zorder=1,
        ))

    # Highlight the nearest red cube + gate-margin ring for DAgger plots
    if dagger:
        nearest = int(np.argmin(np.linalg.norm(centers - start_xy, axis=1)))
        cx, cy = centers[nearest]
        # clearance boundary (hard no-go)
        ax.add_patch(plt.Circle(
            (cx, cy), CLEARANCE,
            color="#FF4400", fill=False, lw=1.2, ls="-", alpha=0.8, zorder=3,
        ))
        # gate-margin ring: CUBE_HALF + GATE_MARGIN from center
        ax.add_patch(plt.Circle(
            (cx, cy), CUBE_HALF + GATE_MARGIN,
            color="#884400", fill=False, lw=0.9, ls=":", alpha=0.7, zorder=3,
        ))

    gx, gy = layout.goal_xy
    ax.plot(gx, gy, "*", color="#FF9900", ms=9, zorder=5)

    # Start marker: circle for normal, triangle for DAgger
    marker = "^" if dagger else "o"
    ax.plot(start_xy[0], start_xy[1], marker, color="#2255CC", ms=7, zorder=6)

    for p_cfg, wps in zip(PLANNERS, paths):
        if wps and len(wps) >= 2:
            xs = [w[0] for w in wps]
            ys = [w[1] for w in wps]
            ax.plot(xs, ys, "-", color=p_cfg["color"], lw=1.6, alpha=0.85, zorder=4)
            ax.plot(xs, ys, ".", color=p_cfg["color"], ms=2.5, zorder=4)
        elif wps is None:
            pass  # silently skip; layout text is cluttered

    ax.set_xlim(0.03, 0.47)
    ax.set_ylim(-0.23, 0.23)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x (m)", fontsize=6)
    ax.set_ylabel("y (m)", fontsize=6)
    ax.grid(True, lw=0.3, alpha=0.4)


def main() -> None:
    total = N_NORMAL + N_DAGGER
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(N_COLS * 3.6, N_ROWS * 3.8),
                             constrained_layout=False)
    axes = np.asarray(axes).flatten()

    # ── Normal layouts (rows 0-2) ────────────────────────────────────────────
    for idx in range(N_NORMAL):
        seed = idx
        print(f"  normal {idx+1}/{N_NORMAL}  seed={seed}", flush=True)
        layout = _load_layout(seed)
        start_xy = layout.blue_cube.pos[:2]
        paths = _paths_from(layout, start_xy)
        _draw(axes[idx], layout, paths, start_xy, f"seed {seed}", dagger=False)

    # ── DAgger-style layouts (row 3) ─────────────────────────────────────────
    # Reuse seeds 0-4 but start from a near-violation position mid-field.
    dagger_seeds = [0, 2, 4, 7, 11]
    for k, seed in enumerate(dagger_seeds):
        ax_idx = N_NORMAL + k
        print(f"  dagger {k+1}/{N_DAGGER}  seed={seed}", flush=True)
        layout = _load_layout(seed)
        start_xy, ci = _find_near_cube_start(layout)
        if start_xy is None:
            axes[ax_idx].set_visible(False)
            continue
        paths = _paths_from(layout, start_xy, inside=True)
        _draw(axes[ax_idx], layout, paths, start_xy,
              f"DAgger seed {seed}", dagger=True)

    # Hide any leftover axes
    for ax in axes[total:]:
        ax.set_visible(False)

    # Row labels
    for row, label in enumerate(["normal layouts", "", "", "DAgger (near-cube start)"]):
        if not label:
            continue
        axes[row * N_COLS].set_ylabel(label + "\ny (m)", fontsize=6.5, labelpad=8)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=p["color"], lw=2, label=p["label"])
        for p in PLANNERS
    ] + [
        Line2D([0], [0], marker="o", color="#2255CC", lw=0, ms=6, label="normal start"),
        Line2D([0], [0], marker="^", color="#2255CC", lw=0, ms=6, label="DAgger start"),
        Line2D([0], [0], marker="*", color="#FF9900", lw=0, ms=9, label="goal"),
        mpatches.Patch(facecolor="#FFCCCC", edgecolor="#991111", label="red cube"),
        Line2D([0], [0], color="#FF4400", lw=1.2, ls="-", label="clearance boundary"),
        Line2D([0], [0], color="#884400", lw=0.9, ls=":", label=f"gate margin ({GATE_MARGIN} m)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=8.5,
               frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Carry corridor: BFS vs A* (λ=1.0, 1.5, 3.0)  —  normal + DAgger starts",
        fontsize=12, y=1.005,
    )
    plt.tight_layout(rect=[0, 0.045, 1, 1])

    out = "videos/corridor_compare.png"
    os.makedirs("videos", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
