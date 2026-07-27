"""Combine per-episode jsonl shards written by ``benchmark_policy.py`` into one
aggregate summary line, matching its ``_summary_line`` format.

Example::

    PYTHONPATH=$PWD .venv/bin/python -m sim.scripts.aggregate_benchmark \\
        outputs/benchmark_bcv7_shard*.jsonl
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("shards", nargs="+", help="jsonl shard paths (per-episode stat rows)")
    args = p.parse_args()

    rows: list[dict] = []
    for path in args.shards:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())

    n = len(rows)
    successes = sum(r["success"] for r in rows)
    contacts = sum(r["red_contact"] for r in rows)
    ceilings = sum(r["ceiling_violation"] for r in rows)
    fly_overs = sum(r["fly_over"] for r in rows)
    drops = sum(r["blue_dropped"] for r in rows)
    timeouts = sum(
        1 for r in rows
        if not r["success"] and not r["red_contact"]
        and not r["ceiling_violation"] and not r["blue_dropped"]
    )
    mean_steps = sum(r["steps"] for r in rows) / n
    mean_clear = sum(r["min_clearance"] for r in rows) / n
    print(
        f"episodes={n}/{n}  SUCCESS={successes}/{n} ({successes / n:.1%})  "
        f"red_contact={contacts}/{n} ({contacts / n:.1%})  ceiling={ceilings}/{n}  "
        f"fly_over={fly_overs}/{n}  blue_drop={drops}/{n}  timeout={timeouts}/{n}  "
        f"mean_steps={mean_steps:.1f}  mean_min_clear={mean_clear:.3f}m"
    )


if __name__ == "__main__":
    main()
