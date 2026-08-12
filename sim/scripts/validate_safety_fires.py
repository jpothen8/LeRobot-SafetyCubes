"""Does the reformed safety loss actually fire on real collisions? (gate D2)

``calibrate_safety_loss.py`` shows the hinge form is ~0 on expert data. That is
necessary but *not* sufficient — a loss that is identically zero everywhere would
score just as well. The complementary check is that it lights up exactly where
the env records a real ``red_contact``, and that it does so with enough lead time
for a gradient to be useful.

Expert data cannot answer this: the scripted expert produces ~0 contacts, so
there is nothing to fire on. So we roll out a *trained policy* with
``terminate_on_red_contact=False`` and keep going through the collision, then
compare the per-frame hinge value on contact frames against clean frames.

Also reports **which geoms actually collide**, from the env's own
``contact_history``. That is the empirical justification for the collision-sphere
set: if a body never appears here, spheres on it are dead weight.

Usage::

    env -u DISPLAY MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python \\
        -m sim.scripts.validate_safety_fires \\
        --checkpoint outputs/safe_pi0_cleanup_v7.1/checkpoints/last/pretrained_model \\
        --dataset-root data/safe_cube_agg_v7.1 --n-episodes 40 --seed 20000
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from sim.safety_geometry import (
    DEFAULT_COLLISION_LINKS,
    DEFAULT_COLLISION_OFFSETS,
    DEFAULT_COLLISION_RADII,
    FKChain,
    collision_index,
    collision_sphere_clearance,
    held_cube_clearance,
    hinge,
)

from sim.scripts.collect_demos import TASK_DESCRIPTION

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-repo-id", default="local/safe-cube-mixed")
    p.add_argument("--dataset-root", default="data/safe_cube_agg_v7.1")
    p.add_argument("--n-episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--n-red-cubes", type=int, default=8)
    p.add_argument("--hinge-margin", type=float, default=0.005)
    p.add_argument("--held-cube-radius", type=float, default=0.0127)
    p.add_argument("--held-cube-offset", type=float, nargs=3,
                   default=[-0.00452, -0.01232, 0.00178])
    p.add_argument("--task", type=str, default=TASK_DESCRIPTION)
    p.add_argument("--urdf", default="sim/assets/so101/so101_new_calib.urdf")
    p.add_argument("--out", type=Path, default=Path("outputs/validate_safety_fires.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from sim.configs import EnvConfig, SceneConfig
    from sim.dagger import PolicyRollout
    from sim.env import SafeCubeEnv

    fk = FKChain(args.urdf, "gripper_frame_link", ARM_JOINTS)
    links, link_idx = collision_index(DEFAULT_COLLISION_LINKS)
    offsets = torch.tensor(DEFAULT_COLLISION_OFFSETS, dtype=torch.float32)
    radii = torch.tensor(DEFAULT_COLLISION_RADII, dtype=torch.float32)
    cube_off = torch.tensor(args.held_cube_offset, dtype=torch.float32)

    def hinge_per_frame(q: np.ndarray, reds: np.ndarray, halves: np.ndarray,
                        grasped: float) -> tuple[float, float]:
        """Max per-sphere hinge value, and the min clearance, for one frame."""
        qt = torch.from_numpy(q.astype(np.float32)).reshape(1, 1, -1)
        c = torch.from_numpy(reds.astype(np.float32)).unsqueeze(0)
        h = torch.from_numpy(halves.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            clear = collision_sphere_clearance(
                fk, qt, c, h, links=links, link_index=link_idx,
                offsets=offsets, radii=radii)
            tcp = fk.fk_link_transforms(qt, ["gripper_frame_link"])[:, :, 0]
            cube = held_cube_clearance(
                tcp, c, h, offset=cube_off, radius=args.held_cube_radius,
                grasped=torch.tensor([grasped]))
            clear = torch.cat([clear, cube], dim=-1)
            hv = hinge((args.hinge_margin - clear) / args.hinge_margin)
        return float(hv.max()), float(clear.min())

    policy = PolicyRollout(
        checkpoint=args.checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
    )
    env = SafeCubeEnv(EnvConfig(
        scene=SceneConfig(n_red_cubes=args.n_red_cubes),
        max_episode_steps=args.max_steps,
        terminate_on_red_contact=False,
    ))

    contact_h, clean_h, contact_c, clean_c = [], [], [], []
    lead_times: list[int] = []
    geom_counts: collections.Counter = collections.Counter()
    n_contact_eps = 0

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        policy.reset()
        prev_hist = 0
        per_frame: list[tuple[float, float, bool]] = []
        halves = np.asarray(info["cube_half_extents"], dtype=np.float32).reshape(-1, 3)

        for _ in range(args.max_steps):
            action = policy.act_queued(obs, args.task)
            obs, _, term, trunc, info = env.step(action)
            # Scan contacts ourselves so we can resolve each colliding geom to its
            # BODY. `contact_history` stores only the geom name, and the arm's
            # collision geoms are unnamed meshes -- they all log as "?", which is
            # useless for deciding which bodies need spheres.
            hist = env._stats.contact_history
            new_contact = len(hist) > prev_hist
            if new_contact:
                prev_hist = len(hist)
                m, dat = env.model, env.data
                for ci in range(dat.ncon):
                    c = dat.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if g1 not in env._red_geom_ids and g2 not in env._red_geom_ids:
                        continue
                    other = g2 if g1 in env._red_geom_ids else g1
                    if other not in env._arm_geom_ids and other not in env._blue_geom_ids:
                        continue
                    bid = int(m.geom_bodyid[other])
                    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
                    gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}"
                    geom_counts[f"{bname}  ({gname})"] += 1

            reds = np.asarray(info["cube_positions"], dtype=np.float32).reshape(-1, 3)
            hv, cl = hinge_per_frame(
                np.asarray(obs["state"], dtype=np.float32),
                reds, halves, float(info.get("grasped", 0.0)))
            per_frame.append((hv, cl, new_contact))
            if term or trunc:
                break

        hits = [i for i, (_, _, c) in enumerate(per_frame) if c]
        if hits:
            n_contact_eps += 1
            first = hits[0]
            # How many frames BEFORE first contact did the loss go non-zero?
            lead = 0
            while first - lead - 1 >= 0 and per_frame[first - lead - 1][0] > 0:
                lead += 1
            lead_times.append(lead)
        for hv, cl, c in per_frame:
            (contact_h if c else clean_h).append(hv)
            (contact_c if c else clean_c).append(cl)

    env.close()

    def d(name: str, v: list[float]) -> str:
        a = np.asarray(v)
        if not len(a):
            return f"  {name:22s} (none)"
        return (f"  {name:22s} n={len(a):6d}  mean {a.mean():9.4f}  p50 {np.median(a):9.4f}  "
                f"p95 {np.percentile(a, 95):9.4f}  max {a.max():9.4f}")

    print(f"\nEpisodes: {args.n_episodes}   with >=1 red_contact: {n_contact_eps}")
    print(f"Frames: {len(clean_h) + len(contact_h)}  (contact frames: {len(contact_h)})")
    print("\nPer-frame MAX sphere hinge value  (0 = silent, 1.0 = at the surface)")
    print(d("contact frames", contact_h))
    print(d("clean frames", clean_h))
    print("\nPer-frame MIN sphere clearance (m)")
    print(d("contact frames", contact_c))
    print(d("clean frames", clean_c))

    if clean_h and contact_h:
        fire_clean = float(np.mean(np.asarray(clean_h) > 0))
        fire_contact = float(np.mean(np.asarray(contact_h) > 0))
        print(f"\nfires (hinge > 0) on {100 * fire_contact:5.1f}% of CONTACT frames")
        print(f"fires (hinge > 0) on {100 * fire_clean:5.1f}% of CLEAN frames")
    if lead_times:
        lt = np.asarray(lead_times)
        print(f"\nlead time before first contact: median {np.median(lt):.0f} frames, "
              f"p90 {np.percentile(lt, 90):.0f}, max {lt.max()}")

    print("\nWhich geoms actually collide (env contact_history):")
    for name, n in geom_counts.most_common():
        print(f"  {name:34s} {n:6d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_episodes": args.n_episodes, "n_contact_episodes": n_contact_eps,
        "contact_hinge": contact_h, "clean_hinge": clean_h,
        "contact_clearance": contact_c, "clean_clearance": clean_c,
        "lead_times": lead_times, "geom_counts": dict(geom_counts),
    }))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
