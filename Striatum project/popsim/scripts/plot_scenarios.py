#!/usr/bin/env python3
"""Validation figures for each simulated coupling scenario.

For every scenario this produces a multi-panel PNG under ``figures/`` showing:

1. Example latent traces for areas A and B.
2. The A-vs-B latent cross-correlogram (reveals lag / direction of coupling).
3. A bar chart of canonical correlations: marginal CCA(A, B) vs partial
   CCA(A, B | C) (reveals mediation collapse).

Plots are saved as static PNGs (not inline) per the repository's visualisation
policy; every panel is labelled with units. The underlying arrays are also
re-derivable from ``scripts/generate_datasets.py`` for regeneration.

Run from the ``popsim`` subproject directory::

    python scripts/plot_scenarios.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from popsim import scenarios, simulate  # noqa: E402
from popsim.metrics import cca, cross_correlation, partial_cca  # noqa: E402

HERE = os.path.dirname(__file__)
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))


def plot_scenario(name: str, out_dir: str, n_timesteps: int, max_lag: int) -> str:
    cfg = scenarios.SCENARIOS[name](n_timesteps=n_timesteps)
    r = simulate(cfg)
    zA, zB, zC = r.latents["A"], r.latents["B"], r.latents["C"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Scenario: {name}", fontweight="bold")

    # Panel 1: example latent traces.
    ax = axes[0]
    t = np.arange(min(600, n_timesteps))
    ax.plot(t, zA[: t.size, 0], label="A latent 0", lw=1.0)
    ax.plot(t, zB[: t.size, 0], label="B latent 0", lw=1.0)
    ax.set_xlabel("time (bins)")
    ax.set_ylabel("latent activity (a.u.)")
    ax.set_title("Example latent traces")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: A-vs-B cross-correlogram on the leading latent dimension. When the
    # scenario has epochs, draw one curve per epoch so a changing lag/direction
    # is visible (a single pooled curve would average the epochs together).
    ax = axes[1]
    bounds = cfg.epoch_boundaries
    if bounds:
        edges = [0, *bounds, n_timesteps]
        names = [f"epoch {i + 1}" for i in range(len(edges) - 1)]
        for i, label in enumerate(names):
            seg = slice(edges[i], edges[i + 1])
            lags, corr = cross_correlation(zA[seg, 0], zB[seg, 0], max_lag)
            peak = lags[np.nanargmax(np.abs(corr))]
            ax.plot(lags, corr, lw=1.2, label=f"{label} (peak {peak:+d})")
        ax.set_title("A->B cross-correlogram per epoch")
    else:
        lags, corr = cross_correlation(zA[:, 0], zB[:, 0], max_lag)
        ax.plot(lags, corr, lw=1.2)
        peak = lags[np.nanargmax(np.abs(corr))]
        ax.axvline(peak, color="r", lw=0.8, ls="--", label=f"peak lag = {peak}")
        ax.set_title("A->B latent cross-correlogram")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("lag (bins): A(t) vs B(t+lag)")
    ax.set_ylabel("correlation")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: marginal vs partial canonical correlations.
    ax = axes[2]
    marg, _, _ = cca(zA, zB)
    part, _, _ = partial_cca(zA, zB, zC)
    k = min(len(marg), len(part))
    x = np.arange(k)
    ax.bar(x - 0.2, marg[:k], width=0.4, label="CCA(A, B)")
    ax.bar(x + 0.2, part[:k], width=0.4, label="partial CCA(A, B | C)")
    ax.set_xlabel("canonical component")
    ax.set_ylabel("canonical correlation")
    ax.set_ylim(0, 1)
    ax.set_title("Marginal vs partial CCA")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n-timesteps", type=int, default=4000)
    parser.add_argument("--max-lag", type=int, default=50)
    parser.add_argument("--scenarios", nargs="*", default=list(scenarios.SCENARIOS))
    args = parser.parse_args()

    for name in args.scenarios:
        path = plot_scenario(name, args.out, args.n_timesteps, args.max_lag)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
