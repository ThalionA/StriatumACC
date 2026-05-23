"""Stage 1 validation: run the residual CCA on the real cohort.

Loads ``preprocessed_data.mat``, classifies the cohort, fits the 5-fold
cross-validated residual CCA for every (animal, pair, epoch), and reports the
held-out canonical correlations. This is the Stage-1 checkpoint deliverable —
no surrogates or lagged refits yet (those are Stage 2).

Run:  python scripts/stage1_validate.py
Outputs:  results/stage1_validation.npz, figures/stage1_*.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from striatum_cca import config, dataio, pipeline  # noqa: E402

CFG = config.DEFAULT
EPOCHS = config.EPOCH_NAMES


def main() -> None:
    print(f"Loading {config.PREPROCESSED_DATA} ...")
    animals = dataio.load_animals()
    entries, yoked = dataio.classify_cohort(animals, CFG)
    print(f"Loaded {len(animals)} animals. Yoked (cohort-mean) LP = {yoked}.\n")

    _print_cohort(animals, entries)

    # Fit every (animal, pair).
    fits: dict[tuple[int, str, str], pipeline.PairFit] = {}
    skips: list[pipeline.SkippedPair] = []
    for animal in animals:
        entry = entries[animal.animal_id]
        for area_x, area_y in config.PAIRS:
            result = pipeline.fit_pair(animal, area_x, area_y, entry, CFG)
            if isinstance(result, pipeline.PairFit):
                fits[(animal.animal_id, area_x, area_y)] = result
            else:
                skips.append(result)
    print(f"\nFitted {len(fits)} (animal x pair) cells; {len(skips)} skipped.\n")

    _print_pair_summary(fits)
    _save_results(fits, entries, yoked)
    _plot_dms_acc_detail(fits)
    _plot_all_pairs_grid(fits)
    print("\nStage 1 validation complete.")


def _print_cohort(animals, entries) -> None:
    print(f"{'animal':>6} {'role':>11} {'lp':>5} {'raw_lp':>7} {'usable_tr':>10}")
    for a in animals:
        e = entries[a.animal_id]
        print(
            f"{a.animal_id:>6} {e.role:>11} {e.lp:>5} "
            f"{str(e.raw_lp):>7} {dataio.n_usable_trials(a):>10}"
        )


def _learner_mean(fits, area_x, area_y, epoch, role) -> tuple[float, int]:
    """Mean held-out CC1 over animals of a given role for one pair/epoch."""
    vals = [
        f.epochs[epoch].held_out_r[0]
        for (aid, ax, ay), f in fits.items()
        if ax == area_x and ay == area_y and f.role == role
    ]
    if not vals:
        return float("nan"), 0
    return float(np.nanmean(vals)), len(vals)


def _print_pair_summary(fits) -> None:
    print("Held-out CC1 (learner group mean) by pair and epoch:")
    print(f"{'pair':>9} {'n':>3}  {'naive':>8} {'interm.':>8} {'expert':>8}")
    for area_x, area_y in config.PAIRS:
        means = [_learner_mean(fits, area_x, area_y, e, "learner") for e in EPOCHS]
        n = means[0][1]
        cells = "  ".join(f"{m:8.3f}" for m, _ in means)
        print(f"{area_x + '-' + area_y:>9} {n:>3}  {cells}")


def _save_results(fits, entries, yoked) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for (aid, area_x, area_y), fit in sorted(fits.items()):
        for epoch in EPOCHS:
            cv = fit.epochs[epoch]
            rows.append(
                dict(
                    animal=aid,
                    pair=f"{area_x}-{area_y}",
                    epoch=epoch,
                    role=fit.role,
                    lp=fit.lp,
                    k=fit.k,
                    samples_per_pc=cv.samples_per_pc,
                    held_out_cc1=cv.held_out_r[0],
                    in_sample_cc1=cv.in_sample_r[0],
                    var_x=float(fit.pca_x_by_epoch[epoch].explained_variance_ratio.sum()),
                    var_y=float(fit.pca_y_by_epoch[epoch].explained_variance_ratio.sum()),
                )
            )
    out = config.RESULTS_DIR / "stage1_validation.npz"
    np.savez(
        out,
        animal=np.array([r["animal"] for r in rows]),
        pair=np.array([r["pair"] for r in rows]),
        epoch=np.array([r["epoch"] for r in rows]),
        role=np.array([r["role"] for r in rows]),
        lp=np.array([r["lp"] for r in rows]),
        k=np.array([r["k"] for r in rows]),
        samples_per_pc=np.array([r["samples_per_pc"] for r in rows]),
        held_out_cc1=np.array([r["held_out_cc1"] for r in rows]),
        in_sample_cc1=np.array([r["in_sample_cc1"] for r in rows]),
        var_x=np.array([r["var_x"] for r in rows]),
        var_y=np.array([r["var_y"] for r in rows]),
        yoked_lp=yoked,
    )
    print(f"\nSaved {len(rows)} rows to {out}")


def _plot_dms_acc_detail(fits) -> None:
    """Per-animal held-out vs in-sample CC1 across epochs, for DMS-ACC."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6, 5))
    for (aid, area_x, area_y), fit in fits.items():
        if (area_x, area_y) != ("DMS", "ACC"):
            continue
        ho = [fit.epochs[e].held_out_r[0] for e in EPOCHS]
        ins = [fit.epochs[e].in_sample_r[0] for e in EPOCHS]
        colour = "tab:blue" if fit.role == "learner" else "tab:orange"
        ax.plot(x, ho, "-o", color=colour, alpha=0.7, markersize=4)
        ax.plot(x, ins, ":", color=colour, alpha=0.35)
    ax.plot([], [], "-o", color="tab:blue", label="held-out (learner)")
    ax.plot([], [], "-o", color="tab:orange", label="held-out (non-learner, yoked)")
    ax.plot([], [], ":", color="gray", label="in-sample (biased)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["naive", "intermediate", "expert"])
    ax.set_xlabel("learning epoch")
    ax.set_ylabel("CC1 (canonical correlation, dimensionless)")
    ax.set_title("DMS-ACC residual CCA — per-animal CC1 across learning")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = config.FIGURES_DIR / "stage1_dms_acc_detail.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_all_pairs_grid(fits) -> None:
    """Grid of learner-group-mean held-out CC1 by epoch, one panel per pair."""
    x = np.arange(3)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    for ax, (area_x, area_y) in zip(axes.ravel(), config.PAIRS):
        for role, colour in (("learner", "tab:blue"), ("nonlearner", "tab:orange")):
            means, ns = zip(
                *[_learner_mean(fits, area_x, area_y, e, role) for e in EPOCHS]
            )
            if ns[0] == 0:
                continue
            ax.plot(x, means, "-o", color=colour, label=f"{role} (n={ns[0]})")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{area_x}-{area_y}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["naive", "int.", "exp."])
        ax.legend(frameon=False, fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out CC1")
    for ax in axes[1, :]:
        ax.set_xlabel("learning epoch")
    fig.suptitle("Stage 1 — held-out CC1 by area pair and learning epoch")
    fig.tight_layout()
    path = config.FIGURES_DIR / "stage1_all_pairs_grid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
