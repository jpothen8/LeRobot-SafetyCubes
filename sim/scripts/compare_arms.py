"""Paired comparison of benchmark arms evaluated on the SAME held-out seeds.

Every arm in the λ sweep is benchmarked on seeds 10000-10999, so the arms are
*paired* per layout, not independent samples. Comparing them with an unpaired
two-proportion test throws away that pairing and badly overstates the
uncertainty: layout difficulty is by far the largest variance component, and it
is shared. McNemar's test conditions on exactly that shared difficulty and only
looks at the seeds where the two arms *disagree*.

Reports, per arm vs the baseline:
  * success and red_contact, with the discordant pair counts McNemar acts on
  * an exact two-sided binomial p on those discordant pairs
  * a Wilcoxon signed-rank z on per-episode ``min_clearance`` -- the continuous
    metric, which has far more power than a ~5% binary rate

Usage::

    PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.compare_arms \\
        --baseline 'outputs/benchmark_lam0_shard*.jsonl' -- \\
        lam1:'outputs/benchmark_lam1_shard*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
import math

import numpy as np


def load(pattern: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r["seed"]] = r
    return rows


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (small n; no scipy)."""
    if n == 0:
        return float("nan")

    def pmf(i: int) -> float:
        return math.comb(n, i) * p**i * (1 - p) ** (n - i)

    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1 + 1e-9)))


def mcnemar(base: list[bool], arm: list[bool]) -> tuple[int, int, float]:
    """Discordant counts (base-only, arm-only) and the exact p."""
    b = sum(1 for x, y in zip(base, arm) if x and not y)
    c = sum(1 for x, y in zip(base, arm) if y and not x)
    return b, c, binom_two_sided(min(b, c), b + c)


def wilcoxon_z(d: np.ndarray) -> tuple[float, float]:
    """Signed-rank z and normal-approx two-sided p (n is ~1000 here)."""
    d = d[d != 0]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    order = np.argsort(np.abs(d), kind="stable")
    r = np.empty(n, dtype=float)
    r[order] = np.arange(1, n + 1, dtype=float)
    a = np.abs(d)
    for v in np.unique(a):                       # midrank ties
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    w = float(r[d > 0].sum())
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sd if sd else float("nan")
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="glob for the baseline arm's shards")
    p.add_argument("--baseline-label", default="baseline")
    p.add_argument("arms", nargs="+", help="label:glob pairs")
    args = p.parse_args()

    base = load(args.baseline)
    if not base:
        print(f"no rows for baseline {args.baseline}")
        return

    print(f"baseline = {args.baseline_label}  ({len(base)} episodes)\n")
    print(f"{'arm':16s} {'n':>5s} {'success':>18s} {'red_contact':>20s} "
          f"{'clearance (Wilcoxon)':>24s}")
    for spec in args.arms:
        label, _, pattern = spec.partition(":")
        arm = load(pattern)
        seeds = sorted(set(base) & set(arm))
        if not seeds:
            print(f"{label:16s} (no overlapping seeds)")
            continue

        bs = [bool(base[s]["success"]) for s in seeds]
        as_ = [bool(arm[s]["success"]) for s in seeds]
        b1, c1, p1 = mcnemar(bs, as_)
        br = [bool(base[s]["red_contact"]) for s in seeds]
        ar = [bool(arm[s]["red_contact"]) for s in seeds]
        b2, c2, p2 = mcnemar(br, ar)
        d = np.array([arm[s]["min_clearance"] - base[s]["min_clearance"] for s in seeds])
        z, pz = wilcoxon_z(d)

        print(f"{label:16s} {len(seeds):5d} "
              f"{100 * sum(as_) / len(seeds):6.1f}% ({c1:3d}w/{b1:3d}l p={p1:5.3f}) "
              f"{100 * sum(ar) / len(seeds):6.1f}% ({b2:3d}w/{c2:3d}l p={p2:5.3f}) "
              f"{np.median(d) * 1000:+6.1f}mm z={z:+6.2f} p={pz:5.3f}")

    print("\nsuccess: w = seeds the arm solved and the baseline did not.")
    print("red_contact: w = seeds where the arm avoided a contact the baseline had.")
    print("clearance: median per-seed change in min_clearance (positive = safer).")
    print("\nPre-registered win condition (CLAUDE.md 2c / plan Phase E): an arm beats")
    print("lambda=0 iff paired success >= baseline - 1pp AND red_contact drops >= 1.5pp")
    print("(or the clearance distribution shifts right at p < 0.05).")


if __name__ == "__main__":
    main()
