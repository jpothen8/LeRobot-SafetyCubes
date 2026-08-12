"""Correlate per-checkpoint ROLLOUT safety-loss statistics with measured red_contact.

Companion to ``rank_checkpoints_by_safety.py`` (gate D3). That script scores each
checkpoint on *expert-state* batches, which is the distribution training actually
sees — but the reformed loss is ~0 there by construction (gate D1), so it ties
most checkpoints at exactly 0.0 and the rank correlation rests on one value.

This version reads the per-frame dumps written by ``validate_safety_fires.py``
(one per checkpoint, produced by ``outputs/run_rank_rollouts.sh``) and ranks on
continuous rollout statistics instead: mean hinge over clean frames, the fraction
of frames that fire, and the 5th-percentile clearance. Those have thousands of
samples per checkpoint rather than one scalar, so they can actually separate
policies that all look identical on expert data.

Caveat worth stating plainly: measured over a policy's own rollouts, "how often
does the loss fire" is *partly* a restatement of "how often does it collide", so
this is a consistency check, not independent evidence that minimising the loss
causes safety. Only the λ sweep can establish that.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from sim.scripts.rank_checkpoints_by_safety import n_distinct, spearman

MODELS = [
    ("bc_v7 (Cell A, lam=1)", "outputs/fires_bcv7.json",
     "outputs/benchmark_bcv7_shard*.jsonl"),
    ("bc_v7_nosafety (Cell B, lam=0)", "outputs/fires_bcv7_nosafety.json",
     "outputs/benchmark_bcv7_ablation_nosafety_shard*.jsonl"),
    ("cleanup_v7.1 (weave DAgger)", "outputs/fires_cleanup_v71.json",
     "outputs/benchmark_cleanup_v7.1_s10000.jsonl"),
    ("cleanup_r2_weave (DAgger r2)", "outputs/fires_r2_weave.json",
     "outputs/benchmark_r2_weave*.jsonl"),
]


def main() -> None:
    rows = []
    for label, dump, bench in MODELS:
        if not Path(dump).exists():
            print(f"SKIP {label}: {dump} missing")
            continue
        d = json.loads(Path(dump).read_text())
        bench_rows: list[dict] = []
        for f in sorted(glob.glob(bench)):
            bench_rows += [json.loads(x) for x in open(f) if x.strip()]
        clean = np.asarray(d["clean_hinge"], dtype=float)
        allh = np.concatenate([clean, np.asarray(d["contact_hinge"], dtype=float)])
        cl = np.asarray(d["clean_clearance"], dtype=float)
        rows.append({
            "label": label,
            "red_contact": sum(r["red_contact"] for r in bench_rows) / len(bench_rows),
            "mean_hinge_all": float(allh.mean()),
            "mean_hinge_clean": float(clean.mean()),
            "frac_firing": float((allh > 0).mean()),
            "clear_p5": float(np.percentile(cl, 5)),
            "n_frames": int(allh.size),
        })

    if len(rows) < 2:
        print("need >=2 checkpoint dumps; run outputs/run_rank_rollouts.sh first")
        return

    rc = [r["red_contact"] for r in rows]
    print(f"\n{'model':34s} {'red_contact':>12s} {'mean hinge':>11s} {'clean hinge':>12s} "
          f"{'% firing':>9s} {'clear p5':>9s} {'frames':>8s}")
    for r in sorted(rows, key=lambda x: x["red_contact"]):
        print(f"{r['label']:34s} {100 * r['red_contact']:11.2f}% {r['mean_hinge_all']:11.5f} "
              f"{r['mean_hinge_clean']:12.5f} {100 * r['frac_firing']:8.2f}% "
              f"{r['clear_p5']:9.4f} {r['n_frames']:8d}")

    print("\nSpearman vs measured red_contact (midrank ties)")
    for key, want in (("mean_hinge_all", "+"), ("mean_hinge_clean", "+"),
                      ("frac_firing", "+"), ("clear_p5", "-")):
        v = [r[key] for r in rows]
        k = n_distinct(v)
        note = "" if k == len(v) else f"   [{len(v) - k + 1} tied]"
        print(f"  {key:18s} rho = {spearman(v, rc):+.3f}  (want {want}){note}")
    print(f"\n  n = {len(rows)} checkpoints -- directional evidence only; "
          "|rho| = 1.0 is needed for p < 0.05 at n=4.")


if __name__ == "__main__":
    main()
