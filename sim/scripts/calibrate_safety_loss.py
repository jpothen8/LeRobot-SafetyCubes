"""Measure the safety loss on expert data before trusting it to shape a policy.

The safety term is only legitimate if it is *small next to the imitation loss on
the expert's own trajectories* — those trajectories are what the policy is being
asked to copy, so a penalty that is large there is not a constraint, it is a
competing objective. That ratio is this script's headline number.

For the shipped ``softplus`` form it is catastrophic. Measured on
``safe_cube_agg_v7.1`` ground-truth expert actions, ``L_obstacle`` ≈ 0.42 and
``L_ceiling`` ≈ 0.61; at the default weights that is
``L_safety = 0.42 + 4·0.61 = 2.87`` against a converged flow loss of **0.0087**
(``safe_pi0_bc_v7_ablation_nosafety``) — a **330×** ratio. The "constraint"
outweighs the objective it is supposed to modify by two and a half orders of
magnitude, which is why ``λ=1`` scored 82.7% where ``λ=0`` scored 95.2%.

What it reports
---------------
1. **FK fidelity** — FK(action) vs ``privileged.ee_pos``. The action is a joint
   *target*, so control lag puts these ~24 mm apart; every threshold below has to
   absorb that, which is why they are measured rather than read off the scene.
2. **Grip offset** — ``ee_pos.z − blue_cube_pos.z`` over grasped frames. The env
   checks the *held cube's* height (``env.py:493-503``) while the loss checked the
   TCP's against the same number; this is the constant that reconciles them.
3. **Per-link clearance** — for every candidate collision sphere, the distribution
   of distance to the nearest red cube, laterally and in 3D. Shows which links
   ever come near a cube and how much room the expert actually leaves.
4. **Loss report** — both penalty forms, lateral and 3D, as an absolute value and
   as a ratio against the imitation loss.

``--calibrate-radii`` additionally fits each sphere's radius against MuJoCo's
exact ``mj_geomDistance`` between that body's real geoms and the red cubes, so
the sphere approximation is unbiased rather than guessed. This needs the sim, so
run it under ``env -u DISPLAY MUJOCO_GL=egl``.

Usage::

    PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.calibrate_safety_loss \\
        --dataset-root data/safe_cube_agg_v7.1 --n-frames 50000

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \\
        -m sim.scripts.calibrate_safety_loss \\
        --dataset-root data/safe_cube_agg_v7.1 --calibrate-radii
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from sim.safety_geometry import (
    FKChain,
    box_sdf,
    ceiling_hinge_loss,
    clearance_hinge_loss,
    height_ceiling_loss,
    multi_point_clearance,
    safety_loss,
    sphere_centers,
)

# Candidate collision spheres. `gripper_frame_link` is the TCP *frame* the old
# single-sphere loss used -- note it carries no geometry of its own and sits
# ~9.8 cm beyond `gripper_link`, so a sphere there misses the jaw bodies that
# actually collide. The bodies with real geoms are the last three.
CANDIDATE_LINKS = [
    "gripper_frame_link",
    "gripper_link",
    "moving_jaw_so101_v1_link",
    "wrist_link",
]
# Reference converged flow loss: safe_pi0_bc_v7_ablation_nosafety (lambda=0) at
# step 20000. The ratio matters at *convergence*, when the imitation loss is
# smallest -- that is the binding case.
DEFAULT_REFERENCE_FLOW_LOSS = 0.0087


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/safe_cube_agg_v7.1"))
    p.add_argument("--n-frames", type=int, default=50000,
                   help="random subsample of frames to measure (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sdf-margin", type=float, default=0.02,
                   help="softplus form only")
    p.add_argument("--hinge-margin", type=float, default=0.005,
                   help="hinge form only. Much tighter than --sdf-margin: the "
                        "sphere covering contains the arm, so it already supplies "
                        "the standoff.")
    p.add_argument("--sdf-alpha", type=float, default=50.0)
    p.add_argument("--ee-radius", type=float, default=0.03)
    p.add_argument("--ee-height-ceiling", type=float, default=0.035)
    p.add_argument("--ceiling-alpha", type=float, default=250.0)
    p.add_argument("--ceiling-buffer", type=float, default=0.005)
    p.add_argument("--ceiling-weight", type=float, default=4.0)
    p.add_argument("--obstacle-weight", type=float, default=1.0)
    p.add_argument("--reference-flow-loss", type=float, default=DEFAULT_REFERENCE_FLOW_LOSS)
    p.add_argument("--urdf", type=str, default="sim/assets/so101/so101_new_calib.urdf")
    p.add_argument("--calibrate-radii", action="store_true",
                   help="fit ONE sphere per link against mj_geomDistance (needs the sim). "
                        "Kept for reference; the links are too long for this to work well.")
    p.add_argument("--derive-spheres", action="store_true",
                   help="derive the multi-sphere query set from the MuJoCo geoms and verify "
                        "it against mj_geomDistance (needs the sim)")
    p.add_argument("--spheres-per-geom", type=int, default=64,
                   help="MAXIMUM spheres per geom; the derivation adds spheres until the "
                        "covering radius drops below --sphere-target-radius.")
    p.add_argument("--sphere-target-radius", type=float, default=0.012,
                   help="stop adding spheres to a geom once every sphere is this small. "
                        "Sets the fidelity of the mesh covering.")
    p.add_argument("--radius-shrink", type=float, default=0.0,
                   help="subtract from every derived radius. Only meaningful for the legacy "
                        "bounding-box derivation (--sphere-source bbox); the vertex covering "
                        "is already tight, so leave this at 0.")
    p.add_argument("--sphere-max-tcp-dist", type=float, default=0.09,
                   help="drop spheres further than this from the TCP in the canonical "
                        "pose. Only the gripper ends can realistically reach a 25 mm "
                        "cube: `wrist_link` sits >=97 mm out and is the binding body on "
                        "2.7%% of poses, so 0.09 removes it wholesale. Note the residual "
                        "sphere bias acts as phantom standoff, so trimming *further* "
                        "works against a tight hinge_margin.")
    p.add_argument("--sphere-source", choices=("mesh", "bbox"), default="mesh",
                   help="'mesh' clusters the actual mesh vertices (tight, orientation-"
                        "independent). 'bbox' is the original geom_size derivation, kept "
                        "only to reproduce the old constants.")
    p.add_argument("--radii-samples", type=int, default=400)
    p.add_argument("--blue-cube-half", type=float, default=0.0127)
    return p.parse_args()


def env_blue_half(args: argparse.Namespace) -> float:
    return args.blue_cube_half


# ---------------------------------------------------------------- data ------
def load_frames(root: Path, n: int, seed: int) -> dict[str, np.ndarray]:
    """Read the privileged columns straight from parquet (no video decode)."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet under {root}/data")
    cols = ["action", "observation.state", "privileged.cube_positions",
            "privileged.cube_half_extents", "privileged.blue_cube_pos",
            "privileged.ee_pos", "privileged.grasped"]
    table = pq.read_table(files, columns=cols) if len(files) > 1 else pq.read_table(files[0], columns=cols)
    total = table.num_rows

    rng = np.random.default_rng(seed)
    idx = np.arange(total) if n <= 0 or n >= total else np.sort(rng.choice(total, n, replace=False))

    def col(name: str, width: int) -> np.ndarray:
        flat = np.asarray(table[name].to_numpy(zero_copy_only=False).tolist(), dtype=np.float32)
        return flat.reshape(total, width)[idx]

    return {
        "action": col("action", 6),
        "state": col("observation.state", 6),
        "cube_positions": col("privileged.cube_positions", 24).reshape(-1, 8, 3),
        "cube_half_extents": col("privileged.cube_half_extents", 24).reshape(-1, 8, 3),
        "blue_cube_pos": col("privileged.blue_cube_pos", 3),
        "ee_pos": col("privileged.ee_pos", 3),
        "grasped": np.asarray(table["privileged.grasped"].to_numpy(), dtype=np.float32)[idx],
        "n_total": total,
        "n_used": len(idx),
    }


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def describe(name: str, a: np.ndarray, unit: str = "m") -> str:
    return (f"  {name:26s} mean {a.mean():7.4f}{unit}  p1 {pct(a, 1):7.4f}  p5 {pct(a, 5):7.4f}  "
            f"p50 {pct(a, 50):7.4f}  p95 {pct(a, 95):7.4f}  min {a.min():7.4f}  max {a.max():7.4f}")


# ------------------------------------------------------------- geometry -----
def link_points(fk: FKChain, actions: np.ndarray, links: list[str]) -> torch.Tensor:
    """(N, 1, K, 3) world positions of each candidate link, from the raw action."""
    q = torch.from_numpy(actions).float().unsqueeze(1)          # (N, 1, 6)
    with torch.no_grad():
        return fk.fk_links(q, links)


def clearances(points: torch.Tensor, centers: np.ndarray, halves: np.ndarray,
               radii: torch.Tensor, lateral: bool) -> np.ndarray:
    """(N, K) clearance of each sphere to the nearest red cube."""
    c = torch.from_numpy(centers).float()
    h = torch.from_numpy(halves).float()
    with torch.no_grad():
        cl = multi_point_clearance(points, c, h, radii, lateral=lateral)   # (N, 1, K)
    return cl.squeeze(1).numpy()


# ------------------------------------------------------ sphere derivation ---
def _label(v: np.ndarray, c: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Nearest-centroid assignment, chunked so the (N, k) matrix stays small."""
    return np.concatenate([((v[i:i + chunk, None] - c[None]) ** 2).sum(-1).argmin(1)
                           for i in range(0, len(v), chunk)])


def _lloyd(v: np.ndarray, k: int, seed: int, iters: int = 15) -> np.ndarray:
    """Tiny k-means (farthest-point seeding) so the module keeps no scipy/sklearn dep.

    Fitted on a subsample -- these meshes carry ~27 k vertices and we only need
    the centroids; the radii are measured against *every* vertex afterwards, so
    the containment guarantee does not depend on the subsample.
    """
    rng = np.random.default_rng(seed)
    fit = v if len(v) <= 4000 else v[rng.choice(len(v), 4000, replace=False)]
    c = fit[rng.integers(len(fit))][None]
    d2 = ((fit - c[0]) ** 2).sum(-1)
    while len(c) < k:                                   # farthest-point seeding
        c = np.concatenate([c, fit[d2.argmax()][None]])
        d2 = np.minimum(d2, ((fit - c[-1]) ** 2).sum(-1))
    for _ in range(iters):
        lab = _label(fit, c)
        for j in range(k):                              # keep empty clusters put
            if (lab == j).any():
                c[j] = fit[lab == j].mean(0)
    return c


def _sphere_cover(v: np.ndarray, max_k: int, target: float, seed: int
                  ) -> list[tuple[np.ndarray, float]]:
    """Cover a vertex cloud with spheres, each radius = max distance to its centroid.

    The union of the returned spheres always **contains** the mesh -- every
    vertex lies inside its own sphere -- so a sphere that is clear of a red cube
    proves the real geom is too. That containment is what lets ``sdf_margin`` go
    to ~0: penetrating the sphere set is already a conservative stand-in for
    touching the arm, so no extra standoff is needed on top of it.

    Doubles ``k`` until every sphere is under ``target``, so the search costs
    O(log) fits rather than one per candidate ``k``.
    """
    k, out = 1, None
    while True:
        c = _lloyd(v, k, seed)
        lab = _label(v, c)
        out = [(c[j], float(np.sqrt(((v[lab == j] - c[j]) ** 2).sum(-1).max())))
               for j in range(k) if (lab == j).any()]
        if max(r for _, r in out) <= target or k >= max_k:
            return out
        k = min(max_k, k * 2)


def derive_spheres(
    bodies: list[str], per_geom: int, seed: int, shrink: float = 0.0,
    *, source: str = "mesh", target_radius: float = 0.012,
) -> list[tuple[str, list[float], float]]:
    """Approximate each gripper geom by a set of spheres.

    A single sphere per link cannot work here: fitting one against
    ``mj_geomDistance`` returns a 76 mm radius for ``gripper_link``, because the
    link origin sits ~10 cm behind geometry that is itself 14 cm long.

    ``source="mesh"`` (default) clusters the geom's **actual mesh vertices** and
    sizes each sphere to its cluster's spread. ``source="bbox"`` reproduces the
    original derivation, which walked spheres along the longest axis of
    ``geom_size`` and sized them to the cross-section. The bbox form is wrong in
    two compounding ways and is kept only for reference:

    * ``geom_size`` for a mesh is ``max|v|`` **about the mesh origin**, not the
      bounding-box half-extent, so it is inflated by however far the mesh sits
      off its own origin (``gripper_link`` g25 reads 71.8 mm in z for a mesh that
      is 53.4 mm long).
    * the cross-section is sized by ``max`` of the two short axes, which is over
      2x too fat for these plate-like parts (``wrist_link`` g23 is 19.6 x 41.6 mm).

    Together those made the spheres read up to 14 mm over-conservative from one
    direction while still reading *thin* from another — a ~25 mm spread against a
    20 mm ``sdf_margin``, and not removable by a scalar ``--radius-shrink``.

    Returns ``(link_name, offset_in_body_frame, radius)`` triples — constants the
    policy maps through FK via :func:`sim.safety_geometry.sphere_centers`.
    """
    import mujoco

    from sim.configs import EnvConfig, SceneConfig
    from sim.env import SafeCubeEnv

    env = SafeCubeEnv(EnvConfig(scene=SceneConfig(), seed=seed))
    env.reset(seed=seed)
    m = env.model
    out: list[tuple[str, list[float], float]] = []
    for name in bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] != bid:
                continue
            gp, gs = np.asarray(m.geom_pos[gi]), np.asarray(m.geom_size[gi])
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[gi])
            rot = rot.reshape(3, 3)

            if source == "mesh" and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                did = m.geom_dataid[gi]
                lo = m.mesh_vertadr[did]
                v = np.asarray(m.mesh_vert[lo:lo + m.mesh_vertnum[did]]).reshape(-1, 3)
                for c, r in _sphere_cover(v.astype(np.float64), per_geom,
                                          target_radius, seed):
                    out.append((name, (rot @ c + gp).round(5).tolist(),
                                round(max(r - shrink, 1e-3), 5)))
                continue

            axis = int(np.argmax(gs))
            radius = float(np.max(np.delete(gs, axis)))
            span = max(float(gs[axis]) - radius, 0.0)
            n = per_geom if span > 1e-6 else 1
            for s in np.linspace(-span, span, n):
                local = np.zeros(3)
                local[axis] = s
                out.append((name, (rot @ local + gp).round(5).tolist(),
                            round(max(radius - shrink, 1e-3), 5)))
    env.close()
    return out


def verify_spheres(spheres, n_samples: int, seed: int, poses: np.ndarray) -> None:
    """Compare the sphere approximation against MuJoCo's exact geom distance.

    Reports **per body** as well as overall. The overall number takes a min over
    every sphere against a min over every geom, so a body that is rarely the
    closest one can be badly mis-sized without moving the aggregate at all --
    which is exactly how the original bounding-box radii passed review.
    """
    import mujoco

    from sim.configs import EnvConfig, SceneConfig
    from sim.env import SafeCubeEnv

    env = SafeCubeEnv(EnvConfig(scene=SceneConfig(), seed=seed))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(poses), size=n_samples, replace=len(poses) < n_samples)
    bodies = sorted({s[0] for s in spheres})
    err: list[float] = []
    per_body: dict[str, list[float]] = {b: [] for b in bodies}

    for s in range(n_samples):
        env.reset(seed=seed + s)
        m, d = env.model, env.data           # reset() rebuilds both -- re-resolve every time
        d.qpos[:5] = poses[idx[s], :5]
        mujoco.mj_forward(m, d)
        red_geoms = list(env._red_geom_ids)

        centers = np.array([d.xpos[b] for b in env._red_body_ids], dtype=np.float32)
        halves = np.full_like(centers, env.cfg.scene.red_cube_half)
        ct = torch.from_numpy(centers).unsqueeze(0)
        ht = torch.from_numpy(halves).unsqueeze(0)

        def sphere_min(sel: list) -> float:
            out = np.inf
            for link, off, rad in sel:
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
                world = d.xmat[bid].reshape(3, 3) @ np.asarray(off) + d.xpos[bid]
                sdf = box_sdf(
                    torch.from_numpy(world.astype(np.float32)).reshape(1, 1, 3), ct, ht)
                out = min(out, float(sdf.amin().item()) - rad)
            return out

        body_geoms = {
            b: [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)]
            for b in bodies
        }
        for b in bodies:
            sel = [s3 for s3 in spheres if s3[0] == b]
            if not sel or not body_geoms[b]:
                continue
            t = min((mujoco.mj_geomDistance(m, d, g, rg, 1.0, None)
                     for g in body_geoms[b] for rg in red_geoms), default=np.inf)
            if np.isfinite(t):
                per_body[b].append(sphere_min(sel) - t)

        true_min = min(
            (mujoco.mj_geomDistance(m, d, g, rg, 1.0, None)
             for b in bodies for g in body_geoms[b] for rg in red_geoms),
            default=np.inf,
        )
        if np.isfinite(true_min):
            err.append(sphere_min(spheres) - true_min)

    env.close()
    e = np.asarray(err)
    print(describe("sphere_clear - true_clear", e))
    print(f"   negative = conservative (fires early). {100 * (e <= 0).mean():.1f}% of poses "
          f"are conservative; median bias {np.median(e) * 1000:+.1f} mm.")
    print(f"\n   {'per-body bias (mm)':32s} {'n':>5s} {'median':>8s} {'p5':>8s} {'p95':>8s}")
    for b in bodies:
        v = np.asarray(per_body[b]) * 1000
        if len(v):
            print(f"   {b:32s} {len(v):5d} {np.median(v):8.1f} "
                  f"{np.percentile(v, 5):8.1f} {np.percentile(v, 95):8.1f}")


# --------------------------------------------------------- radii fitting ----
def calibrate_radii(
    links: list[str], n_samples: int, seed: int, poses: np.ndarray
) -> dict[str, float]:
    """Fit each sphere radius so `origin_distance - radius` tracks true geom distance.

    Replays **real expert arm poses** against real layouts and compares, per body,
    MuJoCo's exact ``mj_geomDistance`` to the red cubes against the box SDF
    evaluated at the link origin. The radius that makes the sphere approximation
    unbiased is the gap between them; we take a high quantile so the sphere errs
    toward over- rather than under-estimating risk.

    Sampling on-distribution matters: uniform-random joint angles put the arm in
    configurations the policy never visits, and the origin-to-geom gap is
    direction-dependent, so off-distribution poses would fit the wrong radius.
    """
    import mujoco

    from sim.configs import EnvConfig, SceneConfig
    from sim.env import SafeCubeEnv

    env = SafeCubeEnv(EnvConfig(scene=SceneConfig(), seed=seed))
    rng = np.random.default_rng(seed)
    pose_idx = rng.choice(len(poses), size=n_samples, replace=len(poses) < n_samples)
    gaps: dict[str, list[float]] = {}

    for s in range(n_samples):
        env.reset(seed=seed + s)
        # `reset()` REBUILDS model and data (new layout -> new mjModel), so every
        # id has to be re-resolved each iteration; caching them across resets
        # silently yields stale indices.
        m, d = env.model, env.data
        body_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in links}
        body_ids = {n: b for n, b in body_ids.items() if b >= 0}
        geoms_of = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
                    for n, b in body_ids.items()}
        red_geoms = list(env._red_geom_ids)

        d.qpos[:5] = poses[pose_idx[s], :5]      # a real expert joint target
        mujoco.mj_forward(m, d)

        centers = np.array([d.xpos[b] for b in env._red_body_ids], dtype=np.float32)
        halves = np.full_like(centers, env.cfg.scene.red_cube_half)
        for name, bid in body_ids.items():
            true_min = min(
                (mujoco.mj_geomDistance(m, d, g, rg, 1.0, None)
                 for g in geoms_of[name] for rg in red_geoms),
                default=np.inf,
            )
            if not np.isfinite(true_min):
                continue
            sdf = box_sdf(
                torch.from_numpy(np.asarray(d.xpos[bid], dtype=np.float32)).reshape(1, 1, 3),
                torch.from_numpy(centers).unsqueeze(0),
                torch.from_numpy(halves).unsqueeze(0),
            )
            gaps.setdefault(name, []).append(float(sdf.amin().item()) - true_min)

    env.close()
    return {n: float(np.percentile(v, 90)) for n, v in gaps.items() if v}


# ------------------------------------------------------------- reporting ----
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"=== calibrate_safety_loss: {args.dataset_root} ===\n")
    data = load_frames(args.dataset_root, args.n_frames, args.seed)
    print(f"frames: {data['n_used']} sampled of {data['n_total']} total\n")

    fk = FKChain(args.urdf, "gripper_frame_link",
                 ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
    links = CANDIDATE_LINKS
    pts = link_points(fk, data["action"], links)              # (N, 1, K, 3)
    tcp = pts[:, 0, links.index("gripper_frame_link"), :].numpy()

    # 1. FK fidelity -- FK(action) is a joint TARGET, privileged.ee_pos is measured.
    print("1. FK(action) vs privileged.ee_pos  [control lag, not error]")
    print(describe("|FK_tcp - ee_pos|", np.linalg.norm(tcp - data["ee_pos"], axis=1)))
    print(describe("z(FK_tcp) - z(ee_pos)", tcp[:, 2] - data["ee_pos"][:, 2]))

    # 2. Grip offset -- the constant that reconciles TCP height with cube height.
    g = data["grasped"] > 0.5
    print(f"\n2. Grip offset over {g.sum()} grasped frames ({g.mean():.1%})")
    if g.any():
        off = data["ee_pos"][g, 2] - data["blue_cube_pos"][g, 2]
        print(describe("ee_pos.z - cube.z", off))
        print(describe("held cube z", data["blue_cube_pos"][g, 2]))
        print(describe("FK_tcp z (grasped)", tcp[g, 2]))
        print(f"  -> ee_to_cube_z_offset = {np.median(off):.4f} m")
        print(f"  -> env ceiling is on the CUBE at {args.ee_height_ceiling}; the TCP sits "
              f"{np.median(off) * 1000:.1f} mm above it, so the old loss compared the wrong body.")

    # 3. Per-link clearance, lateral and 3D, with unit radii (radius subtracted later).
    zero = torch.zeros(len(links))
    print("\n3. Per-link distance to nearest red cube (radius NOT yet subtracted)")
    for mode, lat in (("lateral(XY)", True), ("3D", False)):
        cl = clearances(pts, data["cube_positions"], data["cube_half_extents"], zero, lateral=lat)
        print(f"  -- {mode} --")
        for k, name in enumerate(links):
            print(describe(name, cl[:, k]))
        if lat:
            inside = (cl <= 0).mean(axis=0)
            print("   fraction of frames laterally INSIDE a red prism (fly-over region):")
            for k, name in enumerate(links):
                print(f"     {name:28s} {inside[k]:6.2%}")

    # 4. The headline: safety loss vs the imitation loss it has to coexist with.
    radii = torch.full((len(links),), args.ee_radius)
    tcp_only = pts[:, :, [links.index("gripper_frame_link")], :]
    ref = args.reference_flow_loss

    print(f"\n4. Loss report (imitation reference L_flow = {ref:.4f})")
    print("   Both forms on the SAME expert frames. A term that is large here is not a")
    print("   constraint on the policy -- it is a competing objective.\n")

    ee_r = torch.tensor([args.ee_radius])
    rows = []
    for lat in (True, False):
        cl_tcp = torch.from_numpy(
            clearances(tcp_only, data["cube_positions"], data["cube_half_extents"], ee_r, lateral=lat)
        ).float().unsqueeze(1)
        soft = safety_loss(cl_tcp.squeeze(-1), alpha=args.sdf_alpha, margin=args.sdf_margin).item()
        hin = clearance_hinge_loss(cl_tcp, margin=args.hinge_margin).item()
        rows.append((f"L_obstacle TCP-sphere {'lateral' if lat else '3D':7s}", soft, hin))

    cl_multi = torch.from_numpy(
        clearances(pts, data["cube_positions"], data["cube_half_extents"], radii, lateral=False)
    ).float().unsqueeze(1)
    rows.append(("L_obstacle link-origins 3D",
                 safety_loss(cl_multi.reshape(len(tcp), -1), alpha=args.sdf_alpha,
                             margin=args.sdf_margin).item(),
                 clearance_hinge_loss(cl_multi, margin=args.hinge_margin).item()))

    # The real query set: spheres on the actual geoms (plus the held cube at the TCP).
    spheres = None
    if args.derive_spheres:
        spheres = derive_spheres(
            ["wrist_link", "gripper_link", "moving_jaw_so101_v1_link"],
            args.spheres_per_geom, args.seed, shrink=args.radius_shrink,
            source=args.sphere_source, target_radius=args.sphere_target_radius,
        )
        # NOTE: no held-cube sphere. See safety_geometry.DEFAULT_COLLISION_* --
        # it is not gated on `grasped`, the cube is not at the frame origin, and
        # the cube is the one body the carry corridor already keeps clear. The
        # measurement below is kept as a diagnostic only.
        sph_links = sorted({s[0] for s in spheres})
        mats = fk.fk_link_transforms(
            torch.from_numpy(data["action"]).float().unsqueeze(1), sph_links
        )
        centers = sphere_centers(
            mats,
            torch.tensor([sph_links.index(s[0]) for s in spheres]),
            torch.tensor([s[1] for s in spheres], dtype=torch.float32),
        )
        sph_r = torch.tensor([s[2] for s in spheres], dtype=torch.float32)
        cl_sph = torch.from_numpy(
            clearances(centers, data["cube_positions"], data["cube_half_extents"],
                       sph_r, lateral=False)
        ).float().unsqueeze(1)
        rows.append((f"L_obstacle geom-spheres 3D (K={len(spheres)})",
                     safety_loss(cl_sph.reshape(len(tcp), -1), alpha=args.sdf_alpha,
                                 margin=args.sdf_margin).item(),
                     clearance_hinge_loss(cl_sph, margin=args.hinge_margin).item()))
        print("   (geom-sphere clearance over expert frames:"
              f" min {cl_sph.min():.4f}  p1 {np.percentile(cl_sph.numpy(), 1):.4f}"
              f"  p50 {np.percentile(cl_sph.numpy(), 50):.4f})")

    # Ceiling: old form on the TCP (the bug) vs new form on the held cube.
    gm = torch.from_numpy(data["grasped"]).float()
    tcp_z = torch.from_numpy(tcp[:, 2]).float().reshape(-1, 1)
    cube_z = torch.from_numpy(data["blue_cube_pos"][:, 2]).float().reshape(-1, 1)
    ceil_old = height_ceiling_loss(tcp_z, args.ee_height_ceiling,
                                   alpha=args.ceiling_alpha, weight=gm).item()
    ceil_new = ceiling_hinge_loss(cube_z, args.ee_height_ceiling,
                                  buffer=args.ceiling_buffer, weight=gm).item()
    rows.append(("L_ceiling (old=TCP, new=cube)", ceil_old, ceil_new))

    print(f"   {'term':34s} {'softplus':>12s} {'hinge':>12s}")
    for name, a, b in rows:
        print(f"   {name:34s} {a:12.5f} {b:12.5f}")

    obs_old = rows[0][1]          # lateral TCP sphere == the shipped configuration
    # The proposed form is the geom-sphere row when it was computed, else the
    # link-origin approximation (which the long links make a poor stand-in).
    obs_new = rows[3][2] if spheres is not None else rows[2][2]
    tot_old = args.obstacle_weight * obs_old + args.ceiling_weight * ceil_old
    tot_new = args.obstacle_weight * obs_new + args.ceiling_weight * ceil_new
    print(f"\n   L_safety (obstacle_weight={args.obstacle_weight}, "
          f"ceiling_weight={args.ceiling_weight}):")
    print(f"     shipped  softplus : {tot_old:9.5f}   =  {tot_old / ref:8.1f} x L_flow")
    print(f"     proposed hinge    : {tot_new:9.5f}   =  {tot_new / ref:8.1f} x L_flow")
    print(f"\n   At lambda=1 the safety term should be a small fraction of L_flow on expert")
    print(f"   data. Target <= 0.05x; shipped is {tot_old / ref:.0f}x.")

    if spheres is not None:
        print(f"\n5. Collision spheres derived from the MuJoCo geoms "
              f"({args.spheres_per_geom} per geom, {len(spheres)} total)")
        for link, off, rad in spheres:
            print(f"   {link:28s} offset {str(off):28s} r={rad:.4f}")

        # Held blue cube, expressed in the TCP body frame (constant: the grip is a
        # rigid kinematic lock), so it rides through FK like any other sphere.
        if g.any():
            # MEASURED joints, not the action. The action is a joint *target*, so
            # FK(action) carries ~24 mm of control lag -- which shows up here as a
            # 21 mm median offset with 14.6 mm spread, i.e. mostly lag. FK(measured)
            # gives 13.2 mm with 0.7 mm spread: the grip really is a rigid lock
            # (it is a kinematic attach, not friction), so the offset is a constant
            # worth baking into the sphere rather than noise to average away.
            mats = fk.fk_link_transforms(
                torch.from_numpy(data["state"][g]).float().unsqueeze(1), ["gripper_frame_link"]
            )[:, 0, 0]                                                    # (Ng, 4, 4)
            delta = torch.from_numpy(data["blue_cube_pos"][g]).float() - mats[:, :3, 3]
            local = torch.einsum("nij,nj->ni", mats[:, :3, :3].transpose(1, 2), delta).numpy()
            print(f"   {'gripper_frame_link (held cube)':28s} offset "
                  f"{np.round(np.median(local, axis=0), 5).tolist()!s:28s} "
                  f"r={env_blue_half(args):.4f}")
            print(describe("   held-cube offset spread", np.linalg.norm(local - np.median(local, axis=0), axis=1)))

        print("\n   Verifying against mj_geomDistance:")
        verify_spheres(spheres, args.radii_samples, args.seed, data["action"])

    if args.calibrate_radii:
        print(f"\n5. Fitting sphere radii against mj_geomDistance "
              f"({args.radii_samples} sampled poses)")
        fitted = calibrate_radii(links, args.radii_samples, args.seed, data["action"])
        for name, r in fitted.items():
            print(f"   {name:28s} radius {r:.4f} m")
        print("   (p90 of origin_distance - true_geom_distance: the sphere that makes the")
        print("    approximation unbiased, erring toward over-estimating risk.)")


if __name__ == "__main__":
    main()
