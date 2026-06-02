#!/usr/bin/env python3
"""Drive the real striatum_cca pipeline with popsim ground-truth simulations.

For a few representative scenarios this builds trial-structured populations with
:func:`popsim.simulate_trials`, then runs ``striatum_cca``'s cross-validated CCA,
held-out lagged directionality curve, and partial CCA on them -- demonstrating
that the analysis pipeline reads back the coupling that was configured.

Requires the sibling ``striatum_cca`` package to be importable (its src dir is
auto-discovered; or set ``STRIATUM_CCA_SRC``) and ``h5py`` installed.

Run from the ``popsim`` subproject directory::

    python scripts/striatum_cca_demo.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from popsim import scenarios, simulate_trials  # noqa: E402
from popsim.bridge import analyse_pair, striatum_cca_available  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--n-bins", type=int, default=60)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-lag", type=int, default=25)
    args = parser.parse_args()

    if not striatum_cca_available():
        sys.exit(
            "striatum_cca is not importable. Ensure the cca subproject is present "
            "and h5py is installed (set STRIATUM_CCA_SRC to override the path)."
        )

    print(
        f"{'scenario':<14}{'cv_cc1':>8}{'peak_lag':>10}"
        f"{'partial_cc1':>13}  note"
    )
    print("-" * 64)

    # (scenario builder kwargs, conditioning area, one-line expectation)
    cases = [
        ("no_coupling", {}, None, "A-B uncoupled"),
        ("zero_lag", {}, None, "A->B, lag 0"),
        ("lagged", {"lag_ab": 8}, None, "A leads B by ~8"),
        ("mediated", {}, "C", "collapses given C"),
        ("partial_mediation", {}, "C", "survives given C"),
    ]
    for name, kw, zarea, note in cases:
        cfg = scenarios.SCENARIOS[name](n_timesteps=1, **kw)
        result = simulate_trials(cfg, n_trials=args.n_trials, n_bins=args.n_bins)
        out = analyse_pair(
            result, "A", "B", k=args.k, partial_area=zarea, max_lag=args.max_lag
        )
        pcc = "-" if out.partial_cc is None else f"{out.partial_cc[0]:.2f}"
        print(
            f"{name:<14}{out.held_out_cc[0]:>8.2f}{out.peak_lag:>10d}"
            f"{pcc:>13}  {note}"
        )


if __name__ == "__main__":
    main()
