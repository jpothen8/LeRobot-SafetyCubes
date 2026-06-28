"""Compose a MuJoCo scene by loading the SO-101 URDF and procedurally adding
cubes, goal patch, camera, lights, table, and actuators via MjSpec.

The URDF lives at sim/assets/so101/so101_new_calib.urdf (see sim/assets/README.md).
URDFs don't carry MuJoCo-only constructs (actuators, cameras, sites), so we
build the full scene through MjSpec then call spec.compile() to get an MjModel.

Layout sampling (per project spec):
    - Blue cube on the LEFT (+y) side of the arm — small sampled region.
    - Goal patch FIXED on the RIGHT (-y) side.
    - 6 red cubes inside an 8x8 inch obstacle field between blue and goal.
    - After sampling, a BFS path check rejects layouts where no collision-free
      carry path exists from blue to goal; we then resample.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt

from .configs import SceneConfig

_SQRT2 = math.sqrt(2.0)
# 8-connected grid moves (orthogonal + diagonal), shared by both search modes.
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass
class CubeSpec:
    name: str
    pos: np.ndarray            # (3,)
    half: float
    rgba: tuple[float, float, float, float]


@dataclass
class Layout:
    red_cubes: list[CubeSpec]
    blue_cube: CubeSpec
    goal_xy: np.ndarray        # (2,)
    # XY waypoints from blue→goal that avoid every red cube by ≥
    # path_clearance_radius. Populated by the BFS planner. The expert
    # follows these directly during CARRY; the policy learns from the
    # resulting imitation, not from the waypoints.
    carry_waypoints: list[np.ndarray] = field(default_factory=list)


# Position-control gains (tuned for SO-101 scale). Halved Kp + lower Kd
# slows arm motion ~2x — smoother to watch, no overshoot or oscillation.
_KP = 75.0
_KD = 4.0
_FORCE_LIMIT = 30.0  # N·m — generous for sim; ignores real STS3215 torque limits.

# Non-world arm bodies after URDF→MjSpec conversion. The URDF parser flattens
# `base_link` geoms directly into the world body (without a name), so we also
# treat un-named geoms attached to world (that aren't our floor / table /
# cubes) as part of the arm base for collision-masking purposes.
_ARM_BODIES = frozenset({
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "moving_jaw_so101_v1_link",
})
_NON_ARM_WORLD_GEOMS = frozenset({
    "floor", "table_plane",
    "zone_edge_north", "zone_edge_south", "zone_edge_east", "zone_edge_west",
    "goal_tape_north", "goal_tape_south", "goal_tape_east", "goal_tape_west",
})
# Joint damping / armature added on top of the URDF (URDF carries neither, so the
# resulting system is grossly underdamped without these).
_JOINT_DAMPING = 2.0
_JOINT_ARMATURE = 0.01


def _bfs_parents(
    blocked: np.ndarray, src: tuple[int, int], dst: tuple[int, int],
) -> dict[tuple[int, int], tuple[int, int]] | None:
    """Breadth-first search → parent map. Minimizes the number of cells, so the
    route is the shortest free path and *hugs* the clearance boundary (every
    point ≥ `clearance` from a red is equally good to it, so it picks the
    tightest). Returns ``None`` if ``dst`` is unreachable from ``src``."""
    nx, ny = blocked.shape
    visited = np.zeros_like(blocked)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    q: deque[tuple[int, int]] = deque([src])
    visited[src] = True
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        for di, dj in _NEIGHBORS:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if visited[ni, nj] or blocked[ni, nj]:
                continue
            visited[ni, nj] = True
            parent[(ni, nj)] = cur
            q.append((ni, nj))
    return parent if visited[dst] else None


def _astar_parents(
    blocked: np.ndarray, src: tuple[int, int], dst: tuple[int, int], *,
    grid_res: float, clearance_weight: float, clearance_pref: float,
    repulse_field: np.ndarray | None = None,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    """Clearance-penalized A* → parent map. Each step into a cell within
    ``clearance_pref`` metres of the blocked region pays an extra cost of up to
    ``clearance_weight``× its length, bowing the route toward gap centres.

    When ``repulse_field`` is provided (a precomputed (nx, ny) float array) it
    is used directly instead of the EDT.  :func:`_find_path` supplies this in
    smooth-interior mode so the search can start from positions inside the old
    hard-clearance radius without returning ``None``.

    Heuristic is straight-line distance to ``dst``; admissible → cost-optimal.
    Returns ``None`` if ``dst`` is unreachable from ``src``."""
    nx, ny = blocked.shape
    if repulse_field is not None:
        def repulse(cell: tuple[int, int]) -> float:
            return float(repulse_field[cell])
    else:
        edt = distance_transform_edt(~blocked) * grid_res
        inv_pref = 1.0 / clearance_pref if clearance_pref > 0 else 0.0

        def repulse(cell: tuple[int, int]) -> float:
            # 1 at the clearance boundary, ramping linearly to 0 at clearance_pref.
            return max(0.0, 1.0 - edt[cell] * inv_pref)

    def heuristic(cell: tuple[int, int]) -> float:
        return grid_res * math.hypot(cell[0] - dst[0], cell[1] - dst[1])

    best_g: dict[tuple[int, int], float] = {src: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed = np.zeros_like(blocked)
    heap: list[tuple[float, float, tuple[int, int]]] = [(heuristic(src), 0.0, src)]
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur == dst:
            return parent
        if closed[cur]:
            continue
        closed[cur] = True
        for di, dj in _NEIGHBORS:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if blocked[ni, nj] or closed[ni, nj]:
                continue
            nbr = (ni, nj)
            step = grid_res * (_SQRT2 if di and dj else 1.0)
            ng = g + step * (1.0 + clearance_weight * repulse(nbr))
            if ng < best_g.get(nbr, math.inf):
                best_g[nbr] = ng
                parent[nbr] = cur
                heapq.heappush(heap, (ng + heuristic(nbr), ng, nbr))
    return parent if dst in parent else None


def _find_path(
    reds: list[np.ndarray], blue_xy: np.ndarray, goal_xy: np.ndarray, *,
    clearance: float, grid_res: float,
    bounds_x: tuple[float, float], bounds_y: tuple[float, float],
    waypoint_stride: int = 3,
    clearance_weight: float = 0.0, clearance_pref: float = 0.0,
    wall_x: tuple[float, float] | None = None,
    clearance_interior_weight: float = 0.0,
    clearance_interior_base: float = 0.0,
) -> list[np.ndarray] | None:
    """Plan a 2D grid path of (x, y) world-frame waypoints leading blue→goal.
    Waypoints are downsampled every ``waypoint_stride`` cells; goal_xy is always
    the final waypoint.

    Search mode:
      * ``clearance_weight <= 0`` → plain BFS shortest path (hard ``clearance``
        disks block cells within the radius of each red centre).
      * ``clearance_weight > 0``, ``clearance_interior_weight <= 0`` → A* with
        EDT-based soft clearance penalty and the same hard ``clearance`` disks.
      * ``clearance_weight > 0``, ``clearance_interior_weight > 0`` → **smooth-
        interior A***: no hard-blocked cells from the clearance radius; instead
        a precomputed repulse field extends the soft penalty *inside* the old
        no-go zone.  The field is:

            d ≥ r+p  →  0
            r ≤ d < r+p  →  1 − (d−r)/p          (existing soft ramp, 0→1)
            0 ≤ d < r    →  1 + iw·(1 − d/r)     (inside: 1 at boundary, 1+iw at centre)

        where ``r = clearance``, ``p = clearance_pref``, ``iw =
        clearance_interior_weight``.  Outside the old radius the cost field is
        **identical** to the EDT-based one, so routing is unchanged for paths
        that start in the safe zone.  Inside, the cost rises steeply toward red
        centres — deterring routing through cube bodies — but any real robot
        position can be a valid start without the search returning ``None``.

    ``wall_x = (xlo, xhi)`` erects side barriers (hard-blocked regardless of
    mode).  ``None`` disables it (default).
    """
    x0, x1 = bounds_x
    y0, y1 = bounds_y
    nx = int(np.ceil((x1 - x0) / grid_res)) + 1
    ny = int(np.ceil((y1 - y0) / grid_res)) + 1

    # Smooth-interior mode: clearance disks are NOT hard-blocked; the repulse
    # field handles cost inside them.  Only wall_x remains a hard constraint.
    smooth_interior = clearance_weight > 0.0 and clearance_interior_weight > 0.0

    blocked = np.zeros((nx, ny), dtype=bool)
    if not smooth_interior:
        for cx, cy in reds:
            i_lo = max(0, int(np.floor((cx - clearance - x0) / grid_res)))
            i_hi = min(nx - 1, int(np.ceil((cx + clearance - x0) / grid_res)))
            j_lo = max(0, int(np.floor((cy - clearance - y0) / grid_res)))
            j_hi = min(ny - 1, int(np.ceil((cy + clearance - y0) / grid_res)))
            for i in range(i_lo, i_hi + 1):
                for j in range(j_lo, j_hi + 1):
                    if blocked[i, j]:
                        continue
                    px = x0 + i * grid_res
                    py = y0 + j * grid_res
                    if (px - cx) ** 2 + (py - cy) ** 2 <= clearance ** 2:
                        blocked[i, j] = True

    if wall_x is not None:
        xlo, xhi = wall_x
        col_x = x0 + np.arange(nx) * grid_res
        blocked[(col_x < xlo) | (col_x > xhi), :] = True

    def to_cell(p: np.ndarray) -> tuple[int, int]:
        return (int(round((p[0] - x0) / grid_res)),
                int(round((p[1] - y0) / grid_res)))

    src = to_cell(blue_xy)
    dst = to_cell(goal_xy)
    if not (0 <= src[0] < nx and 0 <= src[1] < ny):
        return None
    if not (0 <= dst[0] < nx and 0 <= dst[1] < ny):
        return None
    if blocked[src] or blocked[dst]:
        return None

    if clearance_weight > 0.0:
        if smooth_interior:
            # Vectorised per-cell distance to the nearest red centre.
            px_grid = x0 + np.arange(nx, dtype=np.float64)[:, None] * grid_res
            py_grid = y0 + np.arange(ny, dtype=np.float64)[None, :] * grid_res
            min_dist = np.full((nx, ny), np.inf)
            for cx, cy in reds:
                np.minimum(min_dist, np.hypot(px_grid - cx, py_grid - cy),
                           out=min_dist)
            # Outside zone (d >= clearance): identical to EDT-based formula.
            inv_pref = 1.0 / clearance_pref if clearance_pref > 0 else 0.0
            outside = min_dist >= clearance
            rep_out = np.maximum(0.0, 1.0 - (min_dist - clearance) * inv_pref)
            # Inside zone (d < clearance): base at boundary, peak at cube centre.
            t = np.where(outside, 0.0, 1.0 - min_dist / clearance)
            rep_in = clearance_interior_base + (clearance_interior_weight - clearance_interior_base) * t
            repulse_field: np.ndarray | None = np.where(outside, rep_out, rep_in)
            parent = _astar_parents(
                blocked, src, dst, grid_res=grid_res,
                clearance_weight=clearance_weight, clearance_pref=clearance_pref,
                repulse_field=repulse_field,
            )
        else:
            parent = _astar_parents(
                blocked, src, dst, grid_res=grid_res,
                clearance_weight=clearance_weight, clearance_pref=clearance_pref,
            )
    else:
        parent = _bfs_parents(blocked, src, dst)
    if parent is None:
        return None

    # Reconstruct path by walking parents back to src.
    cells: list[tuple[int, int]] = [dst]
    cur = dst
    while cur in parent:
        cur = parent[cur]
        cells.append(cur)
    cells.reverse()

    # Downsample, but always keep first and last.
    sampled = cells[::waypoint_stride]
    if sampled[-1] != cells[-1]:
        sampled.append(cells[-1])

    waypoints = [np.array([x0 + i * grid_res, y0 + j * grid_res]) for (i, j) in sampled]
    # Snap final waypoint to the exact goal so the expert lands precisely.
    waypoints[-1] = np.asarray(goal_xy, dtype=np.float64)
    return waypoints


def _path_bounds(cfg: SceneConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    """XY search region for the BFS planner: a touch wider than the obstacle
    square in x, spanning the full blue→goal extent in y (the arm reaches
    roughly x ∈ [0.10, 0.40]). Shared by :func:`sample_layout` (the initial
    spawn→goal plan) and :func:`plan_carry_path` (mid-carry replans) so both
    search the identical grid."""
    cx, cy = cfg.red_field_center
    field_half = cfg.red_field_size / 2
    bounds_x = (cx - field_half - 0.02, cx + field_half + 0.02)
    blue_y_max = max(cfg.blue_y_range)
    goal_y = cfg.goal_pos[1]
    pad = 0.03
    bounds_y = (min(goal_y, -blue_y_max) - pad, max(blue_y_max, abs(goal_y)) + pad)
    return bounds_x, bounds_y


def _field_wall_x(cfg: SceneConfig) -> tuple[float, float] | None:
    """Side-barrier x-limits (the red field's x-extent) when
    ``path_wall_field_sides`` is set, else ``None``. Passed to ``_find_path`` as
    ``wall_x`` to seal the lateral route-around margins. Shared by the initial
    plan and mid-carry replans so both search the identical walled grid."""
    if not cfg.path_wall_field_sides:
        return None
    cx, _ = cfg.red_field_center
    half = cfg.red_field_size / 2
    return (cx - half, cx + half)


def plan_carry_path(
    cfg: SceneConfig, red_centers, start_xy, goal_xy,
) -> list[np.ndarray] | None:
    """Replan a carry corridor from an ARBITRARY start XY to the goal.

    Uses smooth-interior A* (``path_clearance_interior_weight > 0``): no
    hard-blocked clearance disks, so this never returns ``None`` for any real
    robot position (including DAgger cleanup anchors that sit inside the old
    hard-clearance zone).  Outside the clearance radius the cost field is
    identical to the standard EDT-based one; inside it, cost rises steeply
    toward cube centres, deterring routes through obstacle bodies.

    ``red_centers`` may be (n, 2) or (n, 3) — only XY columns are used."""
    reds = [np.asarray(c, dtype=np.float64)[:2] for c in np.asarray(red_centers)]
    bounds_x, bounds_y = _path_bounds(cfg)
    return _find_path(
        reds,
        np.asarray(start_xy, dtype=np.float64),
        np.asarray(goal_xy, dtype=np.float64),
        clearance=cfg.path_clearance_radius,
        grid_res=cfg.path_grid_res,
        bounds_x=bounds_x, bounds_y=bounds_y,
        clearance_weight=cfg.path_clearance_weight,
        clearance_pref=cfg.path_clearance_pref,
        wall_x=_field_wall_x(cfg),
        clearance_interior_weight=cfg.path_clearance_interior_weight,
        clearance_interior_base=cfg.path_clearance_interior_base,
    )


def _sample_red_centers(
    cfg: SceneConfig, rng: np.random.Generator,
    blue_xy: np.ndarray, goal_xy: np.ndarray,
) -> list[np.ndarray] | None:
    """Place `cfg.n_red_cubes` inside the obstacle square with rejection on
    separation. Red-cube ↔ red-cube uses `min_cube_separation`; red-cube ↔
    blue/goal uses the larger `blue_safety_radius` / `goal_safety_radius`
    so the start and drop-off zones stay clear. Returns None on failure
    (caller retries with a fresh blue spawn)."""
    cx, cy = cfg.red_field_center
    half = cfg.red_field_size / 2
    placed: list[np.ndarray] = []
    sep = cfg.min_cube_separation
    for _ in range(cfg.n_red_cubes):
        for _ in range(400):
            x = rng.uniform(cx - half, cx + half)
            y = rng.uniform(cy - half, cy + half)
            p = np.array([x, y])
            ok = (
                all(np.linalg.norm(p - q) >= sep for q in placed)
                and np.linalg.norm(p - blue_xy) >= cfg.blue_safety_radius
                and np.linalg.norm(p - goal_xy) >= cfg.goal_safety_radius
            )
            if ok:
                placed.append(p)
                break
        else:
            return None
    return placed


def sample_layout(cfg: SceneConfig, rng: np.random.Generator) -> Layout:
    """Sample a workable scene: blue on +y, goal at the fixed -y point,
    6 red cubes in the obstacle square, with a guaranteed connectivity path
    from blue→goal."""
    # Path-check region shared with mid-carry replans (see _path_bounds).
    bounds_x, bounds_y = _path_bounds(cfg)
    goal_xy = np.array(cfg.goal_pos, dtype=np.float64)

    for attempt in range(cfg.max_layout_attempts):
        blue_xy = np.array([
            rng.uniform(*cfg.blue_x_range),
            rng.uniform(*cfg.blue_y_range),
        ])
        reds = _sample_red_centers(cfg, rng, blue_xy, goal_xy)
        if reds is None:
            continue
        waypoints = _find_path(
            reds, blue_xy, goal_xy,
            clearance=cfg.path_clearance_radius,
            grid_res=cfg.path_grid_res,
            bounds_x=bounds_x, bounds_y=bounds_y,
            clearance_weight=cfg.path_clearance_weight,
            clearance_pref=cfg.path_clearance_pref,
            wall_x=_field_wall_x(cfg),
        )
        if waypoints is None:
            continue

        red_specs = [
            CubeSpec(
                name=f"red_{i}",
                pos=np.array([p[0], p[1], cfg.table_z + cfg.red_cube_half]),
                half=cfg.red_cube_half,
                rgba=(0.85, 0.15, 0.15, 1.0),
            )
            for i, p in enumerate(reds)
        ]
        blue = CubeSpec(
            name="blue",
            pos=np.array([blue_xy[0], blue_xy[1], cfg.table_z + cfg.blue_cube_half]),
            half=cfg.blue_cube_half,
            rgba=(0.15, 0.30, 0.90, 1.0),
        )
        return Layout(red_cubes=red_specs, blue_cube=blue,
                      goal_xy=goal_xy, carry_waypoints=waypoints)

    raise RuntimeError(
        f"Failed to sample a workable layout in {cfg.max_layout_attempts} attempts. "
        f"Try lowering n_red_cubes, shrinking min_cube_separation, or enlarging "
        f"red_field_size."
    )


def _camera_xyaxes(pos: np.ndarray, lookat: np.ndarray) -> list[float]:
    fwd = lookat - pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, -fwd)
    right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(-fwd, right)
    return [*right, *up]


def _attach_position_actuator(
    spec: mujoco.MjSpec, *, name: str, joint_name: str,
    ctrlrange: tuple[float, float], kp: float, kd: float,
) -> None:
    """Add a position-controlled actuator wired to `joint_name`."""
    spec.add_actuator(
        name=name,
        target=joint_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[kp] + [0.0] * 9,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -kp, -kd] + [0.0] * 7,
        ctrllimited=1,
        ctrlrange=list(ctrlrange),
        forcelimited=1,
        forcerange=[-_FORCE_LIMIT, _FORCE_LIMIT],
    )


def _add_world_decor(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    wb = spec.worldbody
    # Floor plane (large, behind the table).
    wb.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.05],
        pos=[0.0, 0.0, -0.001],
        rgba=[0.6, 0.6, 0.6, 1.0],
    )
    # Table-surface visual patch.
    wb.add_geom(
        name="table_plane",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.30, 0.30, 0.005],
        pos=[0.275, 0.0, cfg.table_z - 0.005],
        rgba=[0.85, 0.82, 0.78, 1.0],
        contype=1, conaffinity=1,
    )

    # Visual border around the red-cube zone (4 thin black strips). Visual
    # only (contype=conaffinity=0) so they don't perturb physics.
    fx, fy = cfg.red_field_center
    half = cfg.red_field_size / 2
    thickness = 0.004                       # 8 mm strip thickness in plane (half-extent 4 mm)
    # ~0.2 mm total height — flat like electrical tape stuck to the table.
    # Lifted ~1 mm above the table top to avoid z-fighting with table_plane.
    height_half = 0.0001
    z_strip = cfg.table_z + 0.001 + height_half
    edges = [
        ("zone_edge_north", (fx, fy + half, z_strip), (half + thickness, thickness, height_half)),
        ("zone_edge_south", (fx, fy - half, z_strip), (half + thickness, thickness, height_half)),
        ("zone_edge_east",  (fx + half, fy, z_strip), (thickness, half + thickness, height_half)),
        ("zone_edge_west",  (fx - half, fy, z_strip), (thickness, half + thickness, height_half)),
    ]
    for name, pos, size in edges:
        wb.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(size),
            pos=list(pos),
            rgba=[0.0, 0.0, 0.0, 1.0],
            # MjSpec.compile() prunes geoms with contype=0/conaffinity=0 as
            # "dead" — so we tag the border with a unique bit (4) that no
            # other geom uses. Mask intersect with arm(2/1), cubes(1/1),
            # floor/table(1/1) all evaluates to 0 ⇒ no actual collisions,
            # but the renderer still draws them.
            contype=4, conaffinity=4,
        )

    # Lights.
    wb.add_light(pos=[0.3, 0.0, 1.2], dir=[0.0, 0.0, -1.0], diffuse=[0.8, 0.8, 0.8])
    wb.add_light(pos=[0.7, 0.5, 1.0], dir=[-0.3, -0.3, -1.0], diffuse=[0.4, 0.4, 0.4])
    # Camera (fixed pose).
    xyaxes = _camera_xyaxes(np.array(cfg.camera_pos), np.array(cfg.camera_lookat))
    wb.add_camera(
        name=cfg.camera_name,
        pos=list(cfg.camera_pos),
        xyaxes=xyaxes,
        fovy=cfg.camera_fovy,
    )


def _add_wrist_camera(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Mount a wrist camera *rigidly* on gripper_link, like a real bolted-on
    USB cam (the standard LeRobot setup).

    The camera is parented to `gripper_link`, so it rides with the wrist. We
    use `fixed` mode (NOT `targetbody`): targetbody auto-aims at a target and
    gimbal-locks the roll to world-up, so the image would *not* roll with
    `wrist_roll` — unphysical for a bolted-on camera. With `fixed` mode the
    whole camera frame is rigid in the gripper, so it rotates with the wrist.
    The orientation is frozen post-compile by `_freeze_wrist_camera_orientation`
    (it needs forward kinematics, unavailable at spec-build time).
    """
    body = next((b for b in spec.bodies if b.name == "gripper_link"), None)
    if body is None:
        return
    body.add_camera(
        name=cfg.wrist_camera_name,
        pos=list(cfg.wrist_camera_pos),
        mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
        fovy=cfg.wrist_camera_fovy,
    )


def _freeze_wrist_camera_orientation(model: mujoco.MjModel, cfg: SceneConfig) -> None:
    """Freeze the wrist camera's orientation in the gripper frame.

    Frames the gripper so it sits centered and pointing straight up:
      * optical axis aimed at the gripper's geometric centroid  → centered;
      * roll set so the gripper's reach direction (gripper_link → the grasp
        `ee_site`, i.e. the way the fingers point) projects to image-up  →
        the gripper stands vertical, fingers up.
    Everything is computed in the gripper frame, so the orientation stays
    correct if `wrist_camera_pos` changes. Baked into `cam_quat` with
    `mode=fixed`, so the frame is rigid in `gripper_link` and rolls with
    `wrist_roll` like a real bolted-on camera. Must run post-compile (uses FK).
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.wrist_camera_name)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.ee_site_name)
    if cid < 0 or gid < 0 or sid < 0:
        return

    # FK at the home pose on a throwaway buffer (geometry within the gripper is
    # arm-pose-independent, but the gripper jaw opening is set by home_gripper).
    data = mujoco.MjData(model)
    for nm, val in zip(cfg.arm_joint_names, cfg.home_qpos):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = val
    gj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg.gripper_joint_name)
    if gj >= 0:
        data.qpos[model.jnt_qposadr[gj]] = cfg.home_gripper
    mujoco.mj_forward(model, data)

    gmat = data.xmat[gid].reshape(3, 3)
    gpos = data.xpos[gid]

    def to_local(p):
        return gmat.T @ (p - gpos)

    def in_gripper_subtree(b):
        while b > 0:
            if b == gid:
                return True
            b = int(model.body_parentid[b])
        return False

    centers = [to_local(data.geom_xpos[g]) for g in range(model.ngeom)
               if in_gripper_subtree(int(model.geom_bodyid[g]))]
    centroid = np.mean(centers, axis=0) if centers else np.zeros(3)
    reach = to_local(data.site_xpos[sid])           # gripper_link → grasp point

    cam = np.array(cfg.wrist_camera_pos)
    fwd = centroid - cam
    fwd /= np.linalg.norm(fwd) + 1e-9               # aim at centroid → centered
    up = reach / (np.linalg.norm(reach) + 1e-9)     # reach dir → image up
    # Camera frame (in gripper frame): x=right, y=up, z=back(-fwd).
    z = -fwd
    y = up - (up @ z) * z
    y /= np.linalg.norm(y) + 1e-9
    x = np.cross(y, z)
    x /= np.linalg.norm(x) + 1e-9
    R = np.column_stack([x, y, z])

    # Extra upward tilt: rotate the camera about its own right axis so the
    # optical axis pitches up toward the gripper's reach (keeps it horizontally
    # centered and vertical).
    a = np.radians(cfg.wrist_camera_pitch_up_deg)
    c, s = np.cos(a), np.sin(a)
    R = R @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())

    model.cam_mode[cid] = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    model.cam_targetbodyid[cid] = -1
    model.cam_quat[cid] = quat


def _add_ee_site(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Add an end-effector tracking site. The SO-101 URDF already has a dummy
    `gripper_frame_link` body at the gripper tip — we hang the site on it.

    Also add a `grasp_site` at the jaw-gap center (midway between the open
    finger faces, at the fingertip plane). The URDF gripper frame sits at the
    FIXED jaw, ~3 cm to the side of where a grasped cube actually goes, so
    targeting it perches the cube on one jaw instead of between them. The
    grasp site is the true tool-center point: IK and the magnetic grip target
    it so the cube ends up centered between the fingers."""
    target_name = "gripper_frame_link"
    body = next((b for b in spec.bodies if b.name == target_name), None)
    if body is None:
        # Fallback: attach to the deepest body in the chain.
        body = max((b for b in spec.bodies if b.name not in ("world",)),
                   key=lambda b: len(b.name))
    body.add_site(
        name=cfg.ee_site_name,
        pos=[0.0, 0.0, 0.0],
        size=[0.005, 0.005, 0.005],
        rgba=[1.0, 1.0, 0.0, 0.6],
        group=3,
    )
    body.add_site(
        name=cfg.grasp_site_name,
        pos=list(cfg.grasp_site_offset),
        size=[0.005, 0.005, 0.005],
        rgba=[0.0, 1.0, 1.0, 0.6],
        group=3,
    )


def _add_goal_tape(wb, cfg: SceneConfig, layout: Layout) -> None:
    """4 blue tape strips marking the goal square (3x3 inch interior)."""
    gx, gy = float(layout.goal_xy[0]), float(layout.goal_xy[1])
    half = cfg.goal_size / 2
    thickness = 0.004                       # 8 mm wide strips
    height_half = 0.0001                    # ~0.2 mm tall (electrical tape)
    z_strip = cfg.table_z + 0.001 + height_half
    blue_rgba = [0.1, 0.3, 0.95, 1.0]
    edges = [
        ("goal_tape_north", (gx, gy + half, z_strip), (half + thickness, thickness, height_half)),
        ("goal_tape_south", (gx, gy - half, z_strip), (half + thickness, thickness, height_half)),
        ("goal_tape_east",  (gx + half, gy, z_strip), (thickness, half + thickness, height_half)),
        ("goal_tape_west",  (gx - half, gy, z_strip), (thickness, half + thickness, height_half)),
    ]
    for name, pos, size in edges:
        wb.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(size),
            pos=list(pos),
            rgba=blue_rgba,
            contype=4, conaffinity=4,        # see note in _add_world_decor
        )


def _add_cubes_and_goal(spec: mujoco.MjSpec, cfg: SceneConfig, layout: Layout) -> None:
    wb = spec.worldbody
    for red in layout.red_cubes:
        b = wb.add_body(name=red.name, pos=list(red.pos))
        b.add_freejoint(name=f"{red.name}_free")
        b.add_geom(
            name=f"{red.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[red.half, red.half, red.half],
            rgba=list(red.rgba),
            mass=0.05,
            friction=[1.0, 0.05, 0.001],
            condim=4,
        )

    b = wb.add_body(name=layout.blue_cube.name, pos=list(layout.blue_cube.pos))
    b.add_freejoint(name="blue_free")
    b.add_geom(
        name=f"{layout.blue_cube.name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[layout.blue_cube.half] * 3,
        rgba=list(layout.blue_cube.rgba),
        mass=0.05,
        friction=[1.0, 0.05, 0.001],
        condim=4,
    )

    # Invisible site at the goal center (kept so the env can read its world
    # position via site_xpos). The *visible* goal indicator is drawn as
    # blue tape strips in _add_goal_tape.
    wb.add_site(
        name="goal",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[layout.goal_xy[0], layout.goal_xy[1], cfg.table_z + 0.001],
        size=[0.001, 0.001, 0.001],
        rgba=[0.0, 0.0, 0.0, 0.0],         # invisible
        group=3,
    )
    _add_goal_tape(wb, cfg, layout)


def _add_actuators(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Add one position-controlled actuator per joint. ctrlrange = joint range."""
    joints_by_name = {j.name: j for j in spec.joints}
    for jname in (*cfg.arm_joint_names, cfg.gripper_joint_name):
        if jname not in joints_by_name:
            raise KeyError(
                f"Joint '{jname}' not found in URDF. Available: {list(joints_by_name)}"
            )
        jr = joints_by_name[jname].range
        # URDF-loaded joints carry their limits; fall back to a wide range if zero.
        if jr[0] == 0.0 and jr[1] == 0.0:
            jr = [-3.14, 3.14]
        _attach_position_actuator(
            spec,
            name=f"{jname}_act",
            joint_name=jname,
            ctrlrange=(float(jr[0]), float(jr[1])),
            kp=_KP, kd=_KD,
        )


def build_scene(cfg: SceneConfig, layout: Layout) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Build and compile the full scene. Returns (model, spec)."""
    src = cfg.resolved_mjcf()
    if not src.exists():
        raise FileNotFoundError(
            f"Robot model not found at {src}. See sim/assets/README.md."
        )
    spec = mujoco.MjSpec.from_file(str(src))
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.timestep = 0.002

    # URDFs don't ship joint damping/armature; without these, position
    # controllers driving the SO-101 oscillate wildly. Apply uniformly to all
    # of the arm + gripper joints (the cube freejoints we'll add later are
    # different bodies — they get added after this and are unaffected).
    arm_joints = set(cfg.arm_joint_names) | {cfg.gripper_joint_name}
    for j in spec.joints:
        if j.name in arm_joints:
            # MjsJoint.damping is a length-3 vector (per-axis); only the first
            # entry matters for a 1-dof hinge but all three must be set.
            j.damping = [_JOINT_DAMPING, 0.0, 0.0]
            j.armature = _JOINT_ARMATURE

    _add_world_decor(spec, cfg)
    _add_ee_site(spec, cfg)
    _add_wrist_camera(spec, cfg)
    _add_cubes_and_goal(spec, cfg, layout)
    _add_actuators(spec, cfg)

    model = spec.compile()

    # Disable self-collision among arm bodies (incl. base-link geoms that the
    # URDF parser flattened into world). Bit-mask trick:
    #   arm geoms      → contype=2, conaffinity=1
    #   world (floor / table / cubes) → contype=1, conaffinity=1 (default)
    # Two geoms collide iff their masks intersect, so arm-vs-arm = 0 (off)
    # but arm-vs-cube = 1 (on).
    for g in range(model.ngeom):
        bid = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        is_arm_link = bname in _ARM_BODIES
        is_flattened_base = (
            bid == 0 and gname not in _NON_ARM_WORLD_GEOMS
            and not gname.endswith("_geom")          # excludes our cube geoms
        )
        if is_arm_link or is_flattened_base:
            model.geom_contype[g] = 2
            model.geom_conaffinity[g] = 1

    _freeze_wrist_camera_orientation(model, cfg)

    return model, spec


# Back-compat alias: the env used to expect a path-based loader.
def build_scene_xml(cfg: SceneConfig, layout: Layout) -> tuple[mujoco.MjModel, Path]:
    model, _ = build_scene(cfg, layout)
    # Path retained in signature for callers that wanted to keep a debug file.
    return model, Path("<in-memory MjSpec>")
