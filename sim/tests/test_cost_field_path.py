"""Unit tests for the corridor planner's two search modes (no MuJoCo needed —
``_find_path`` is pure grid search over numpy).

    uv run pytest sim/tests/test_cost_field_path.py -q

Covers the contract that matters for the cost-field A* variant:
  * weight=0 reproduces the old BFS path exactly (default is unchanged),
  * weight>0 keeps the corridor strictly *farther* from obstacles than BFS
    while never violating the hard clearance radius,
  * connectivity is identical (A* finds a path iff BFS does).
"""

from __future__ import annotations

import numpy as np

from sim.scene import _find_path

# A wide, open grid so the A* variant has room to bow away from the obstacle.
BOUNDS_X = (0.0, 0.4)
BOUNDS_Y = (0.0, 0.4)
GRID_RES = 0.005
CLEARANCE = 0.045

# Blue→goal is a straight horizontal line at y=0.2; a single red sits just below
# it, so the *shortest* (BFS) route must graze the top of the clearance disk
# while there is open space above for a higher-clearance detour.
BLUE = np.array([0.05, 0.20])
GOAL = np.array([0.35, 0.20])
REDS = [np.array([0.20, 0.16])]


def _plan(weight: float, pref: float = 0.04):
    return _find_path(
        REDS, BLUE, GOAL,
        clearance=CLEARANCE, grid_res=GRID_RES,
        bounds_x=BOUNDS_X, bounds_y=BOUNDS_Y,
        clearance_weight=weight, clearance_pref=pref,
    )


def _min_clearance(waypoints: list[np.ndarray]) -> float:
    """Smallest distance from any red center to the polyline through the
    waypoints (densely sampled, so it catches between-waypoint dips)."""
    reds = np.asarray(REDS)
    pts = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(2, int(np.linalg.norm(b - a) / (GRID_RES / 2)))
        pts.append(np.linspace(a, b, n))
    samples = np.vstack(pts)
    dists = np.linalg.norm(samples[:, None, :] - reds[None, :, :], axis=2)
    return float(dists.min())


def test_default_weight_matches_bfs():
    # weight=0 must reproduce the plain BFS path byte-for-byte (no behavior
    # change unless the cost field is explicitly switched on).
    bfs = _find_path(
        REDS, BLUE, GOAL,
        clearance=CLEARANCE, grid_res=GRID_RES,
        bounds_x=BOUNDS_X, bounds_y=BOUNDS_Y,
    )  # no weight kwargs -> default 0.0
    explicit = _plan(weight=0.0)
    assert bfs is not None and explicit is not None
    assert len(bfs) == len(explicit)
    for a, b in zip(bfs, explicit):
        assert np.allclose(a, b)


def test_astar_stays_farther_than_bfs():
    bfs = _plan(weight=0.0)
    astar = _plan(weight=3.0, pref=0.05)
    assert bfs is not None and astar is not None
    bfs_clear = _min_clearance(bfs)
    astar_clear = _min_clearance(astar)
    # BFS hugs the boundary (≈ CLEARANCE); A* should keep meaningfully farther.
    assert astar_clear > bfs_clear + 0.01, (bfs_clear, astar_clear)


def test_both_modes_respect_hard_clearance():
    # The hard clearance radius is a constraint in BOTH modes — neither path may
    # come closer than CLEARANCE to a red center (allow one grid cell of
    # discretization slack on the densely-sampled polyline).
    tol = GRID_RES
    for weight in (0.0, 3.0):
        path = _plan(weight=weight)
        assert path is not None
        assert _min_clearance(path) >= CLEARANCE - tol


def test_connectivity_matches_bfs_when_blocked():
    # Fully wall off the goal: both modes must report no path (A* doesn't
    # invent a route through the hard-blocked region).
    wall = [np.array([0.30, y]) for y in np.arange(0.0, 0.4 + 1e-9, 0.02)]
    common = dict(
        clearance=CLEARANCE, grid_res=GRID_RES,
        bounds_x=BOUNDS_X, bounds_y=BOUNDS_Y,
    )
    bfs = _find_path(wall, BLUE, GOAL, **common)
    astar = _find_path(wall, BLUE, GOAL, clearance_weight=3.0, clearance_pref=0.05, **common)
    assert bfs is None and astar is None


def test_wall_x_confines_path_and_blocks_route_around():
    # A red dead-center between start (bottom) and goal (top) lets a high-λ A*
    # bow far sideways to avoid it. wall_x must keep every waypoint inside the
    # band, so the corridor weaves past the red instead of routing around it.
    blue, goal = np.array([0.20, 0.04]), np.array([0.20, 0.36])
    reds = [np.array([0.20, 0.20])]
    wall = (0.10, 0.30)
    common = dict(clearance=CLEARANCE, grid_res=GRID_RES,
                  bounds_x=BOUNDS_X, bounds_y=BOUNDS_Y,
                  clearance_weight=3.0, clearance_pref=0.05)
    free = np.array(_find_path(reds, blue, goal, **common))
    walled = np.array(_find_path(reds, blue, goal, wall_x=wall, **common))
    # The walled path stays within the barrier (± one grid cell of rounding)…
    assert walled[:, 0].min() >= wall[0] - GRID_RES
    assert walled[:, 0].max() <= wall[1] + GRID_RES
    # …and is laterally tighter than the unwalled route-around.
    assert np.ptp(walled[:, 0]) < np.ptp(free[:, 0])
