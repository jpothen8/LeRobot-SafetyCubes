"""Does the safety loss rank trained checkpoints by their measured safety? (gate D3)

The last and strictest of the three validation gates. D1 says the loss is ~0 on
expert data; D2 says it fires on real collisions. Neither shows it is a *useful
training signal* — for that, a checkpoint that collides more must score worse.

This is the gate the shipped ``softplus`` form **fails**: across the four existing
π0 checkpoints its ``l_obstacle`` *anti*-correlates with measured ``red_contact``,
i.e. minimising it made the policy less safe, which is exactly the pathology that
produced λ=1 (82.7%) losing to λ=0 (95.2%).

Method: run each checkpoint's own ``forward`` over an identical set of batches
(same data, same torch seed, so the flow-time draws match sample-for-sample) and
read the loss dict. Both penalty forms are evaluated on the same forward pass, so
the comparison isolates the form.

Ground-truth ``red_contact`` comes from ``outputs/benchmark_*.jsonl`` (1000 shared
held-out seeds, 10000-10999).

Usage::

    PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.rank_checkpoints_by_safety \\
        --dataset-root data/safe_cube_agg_v7.1 --n-batches 24 --batch-size 16
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

import sim.safe_pi0_policy  # noqa: F401  -- registers `safe_pi0`

# (label, checkpoint dir, benchmark jsonl glob)
MODELS = [
    ("bc_v7 (Cell A, lam=1)", "outputs/safe_pi0_bc_v7",
     "outputs/benchmark_bcv7_shard*.jsonl"),
    ("bc_v7_nosafety (Cell B, lam=0)", "outputs/safe_pi0_bc_v7_ablation_nosafety",
     "outputs/benchmark_bcv7_ablation_nosafety_shard*.jsonl"),
    ("cleanup_v7.1 (weave DAgger)", "outputs/safe_pi0_cleanup_v7.1",
     "outputs/benchmark_cleanup_v7.1_s10000.jsonl"),
    ("cleanup_r2_weave (DAgger r2)", "outputs/safe_pi0_cleanup_r2_weave",
     "outputs/benchmark_r2_weave*.jsonl"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-repo-id", default="local/safe-cube-mixed")
    p.add_argument("--dataset-root", default="data/safe_cube_agg_v7.1")
    p.add_argument("--n-batches", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-noise-frac", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("outputs/rank_checkpoints_by_safety.json"))
    return p.parse_args()


def measured_red_contact(pattern: str) -> tuple[float, int]:
    import glob

    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(line) for line in open(f) if line.strip()]
    if not rows:
        return float("nan"), 0
    return sum(r["red_contact"] for r in rows) / len(rows), len(rows)


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without a scipy dependency, with MIDRANK ties.

    Tie handling is not a detail here: the reformed loss is exactly 0.0 for most
    checkpoints, and plain ``argsort`` would silently impose an arbitrary order on
    those ties and report a confident correlation that is pure tie-breaking
    artifact. Tied values must share the average rank, and an all-tied vector has
    no ranking at all -> NaN.
    """
    def ranks(v: list[float]) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):                    # midrank each tie group
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def n_distinct(v: list[float]) -> int:
    return len(np.unique(np.asarray(v, dtype=float)))


def main() -> None:
    args = parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    results = []

    for label, ckpt_dir, bench in MODELS:
        ckpt = f"{ckpt_dir}/checkpoints/last/pretrained_model"
        if not Path(ckpt).exists():
            print(f"SKIP {label}: {ckpt} missing")
            continue

        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        # Identical batches for every checkpoint: same delta_timestamps, same
        # sampler seed. Rebuilt per model because delta_timestamps is cfg-derived.
        ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root,
                            delta_timestamps=resolve_delta_timestamps(cfg, meta))
        policy = make_policy(cfg, ds_meta=ds.meta).eval()
        pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=ckpt,
                                          dataset_stats=ds.meta.stats)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
            generator=torch.Generator().manual_seed(args.seed))

        # Without the normalization processor + uint8->float conversion the batch
        # is un-normalized and every geometry number downstream is garbage
        # (lerobot_train.py:471).
        batches = []
        for _, b in zip(range(args.n_batches), dl):
            b = {k: (v.to(cfg.device) if torch.is_tensor(v) else v) for k, v in b.items()}
            for cam in ds.meta.camera_keys:
                if cam in b and b[cam].dtype == torch.uint8:
                    b[cam] = b[cam].to(dtype=torch.float32) / 255.0
            batches.append(pre(b))

        row: dict = {"label": label}
        for form, gate in (("softplus", 1.0), ("hinge", args.max_noise_frac)):
            policy.config.safety_form = form
            policy.config.safety_max_noise_frac = gate
            obs, ceil, flow = [], [], []
            torch.manual_seed(args.seed)        # pair the flow-time draws
            for b in batches:
                with torch.no_grad():
                    _, d = policy.forward(dict(b))
                obs.append(d["l_obstacle"])
                ceil.append(d["l_ceiling"])
                flow.append(d["l_flow"])
            row[form] = {"l_obstacle": float(np.mean(obs)),
                         "l_ceiling": float(np.mean(ceil)),
                         "l_flow": float(np.mean(flow))}

        rc, n = measured_red_contact(bench)
        row["red_contact"] = rc
        row["n_eval"] = n
        results.append(row)
        print(f"  {label:34s} rc={100 * rc:5.2f}%  "
              f"softplus obs={row['softplus']['l_obstacle']:.5f}  "
              f"hinge obs={row['hinge']['l_obstacle']:.6f}")

        del policy, batches, ds, dl
        gc.collect()
        torch.cuda.empty_cache()

    if not results:
        print("no checkpoints evaluated")
        return

    rc = [r["red_contact"] for r in results]
    print(f"\n{'model':34s} {'red_contact':>12s} {'sp l_obst':>11s} {'sp l_ceil':>11s} "
          f"{'hi l_obst':>11s} {'hi l_ceil':>11s} {'l_flow':>9s}")
    for r in sorted(results, key=lambda x: x["red_contact"]):
        print(f"{r['label']:34s} {100 * r['red_contact']:11.2f}% "
              f"{r['softplus']['l_obstacle']:11.5f} {r['softplus']['l_ceiling']:11.5f} "
              f"{r['hinge']['l_obstacle']:11.6f} {r['hinge']['l_ceiling']:11.6f} "
              f"{r['hinge']['l_flow']:9.5f}")

    print("\nSpearman rank correlation vs measured red_contact "
          "(want POSITIVE: worse loss == more collisions)")
    tot = {f: [r[f]["l_obstacle"] + 4.0 * r[f]["l_ceiling"] for r in results]
           for f in ("softplus", "hinge")}
    for form in ("softplus", "hinge"):
        for term in ("l_obstacle", "l_ceiling", "L_safety"):
            v = tot[form] if term == "L_safety" else [r[form][term] for r in results]
            k = n_distinct(v)
            note = "" if k == len(v) else f"   [{len(v) - k + 1} tied values -- midranked]"
            if k == 1:
                note = "   [ALL TIED -- no ranking information]"
            print(f"  {form:9s} {term:11s} rho = {spearman(v, rc):+.3f}{note}")
    print(f"\n  n = {len(rc)} checkpoints. Spearman needs |rho| = 1.0 for p < 0.05 at n=4,")
    print("  so treat anything below that as DIRECTIONAL evidence, not a significant result.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
