#!/usr/bin/env python3
"""Run the coupling-recovery benchmark over every scenario and report.

Simulates each scenario, recovers coupling metrics from the *neural* data with
standard CCA / lagged-CCA / partial-CCA, and scores them against the configured
ground truth. Prints a table and writes ``data/generated/recovery_benchmark.json``.

Run from the ``popsim`` subproject directory::

    python scripts/recovery_benchmark.py
    python scripts/recovery_benchmark.py --dynamics lds --n-timesteps 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from popsim.benchmark import format_table, rows_to_dicts, run_benchmark  # noqa: E402

HERE = os.path.dirname(__file__)
DEFAULT_OUT = os.path.normpath(
    os.path.join(HERE, "..", "data", "generated", "recovery_benchmark.json")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-timesteps", type=int, default=6000)
    parser.add_argument("--k", type=int, default=5, help="PCs retained per area")
    parser.add_argument("--max-lag", type=int, default=50)
    parser.add_argument(
        "--dynamics", default="ar1", choices=["ar1", "lds", "oscillatory"]
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = run_benchmark(
        n_timesteps=args.n_timesteps,
        k=args.k,
        max_lag=args.max_lag,
        builder_kwargs={"dynamics": args.dynamics},
    )
    print(format_table(rows))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {"dynamics": args.dynamics, "n_timesteps": args.n_timesteps,
             "k": args.k, "rows": rows_to_dicts(rows)},
            fh,
            indent=2,
        )
    print(f"\nwrote {args.out}")

    n_pass = sum(r.passed for r in rows)
    sys.exit(0 if n_pass == len(rows) else 1)


if __name__ == "__main__":
    main()
