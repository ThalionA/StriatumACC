"""Partial CCA add-on (D3): DMS/DLS/ACC triplet.

For each animal recording all three striatal-cingulate areas, compares each
pair's plain held-out CC1 with its partial CC1 (third area regressed out).
Saves results/partial.pkl and figures/partial_cca.png.

Run:  python scripts/run_partial.py
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dataclasses  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from striatum_cca import config, core, dataio, partial, pipeline  # noqa: E402

EPOCHS = config.EPOCH_NAMES
# (pair X, pair Y, partialled-out area)
TRIPLET = [("DMS", "DLS", "ACC"), ("DMS", "ACC", "DLS"), ("DLS", "ACC", "DMS")]


def main() -> None:
    print(f"Loading cohort from {config.PREPROCESSED_DATA} ...")
    animals = dataio.load_animals()
    for zscore in (False, True):
        cfg = dataclasses.replace(config.DEFAULT, zscore_units=zscore)
        run_variant(animals, cfg, "z1" if zscore else "z0")


def run_variant(animals, cfg, tag: str) -> None:
    entries, _ = dataio.classify_cohort(animals, cfg)

    rows = []
    for animal in animals:
        entry = entries[animal.animal_id]
        prepped = {a: pipeline.prepare_area(animal, a, entry, cfg)
                   for a in ("DMS", "DLS", "ACC")}
        if any(prepped[a] is None for a in prepped):
            continue
        for area_x, area_y, area_z in TRIPLET:
            for epoch in EPOCHS:
                sx, sy, sz = (prepped[area_x][epoch], prepped[area_y][epoch],
                              prepped[area_z][epoch])
                plain = core.cca_cv(sx, sy, cfg).held_out_r[0]
                part = partial.partial_cca_cv(sx, sy, sz, cfg).held_out_r[0]
                rows.append(dict(animal=animal.animal_id, role=entry.role,
                                 pair=f"{area_x}-{area_y}", partialled=area_z,
                                 epoch=epoch, plain_cc1=plain, partial_cc1=part))
    print(f"[{tag}] {len(rows)} (animal x pair x epoch) partial-CCA cells "
          f"from {len({r['animal'] for r in rows})} animals.")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / f"partial_{tag}.pkl", "wb") as fh:
        pickle.dump({"rows": rows, "cfg": cfg}, fh)

    _summary(rows)
    _plot(rows, tag)


def _summary(rows) -> None:
    print("\nPlain vs partial held-out CC1 (learner-group mean):")
    print(f"{'pair (|partialled)':>22}  {'naive':>14} {'interm.':>14} {'expert':>14}")
    for area_x, area_y, area_z in TRIPLET:
        lab = f"{area_x}-{area_y} | {area_z}"
        cells = []
        for epoch in EPOCHS:
            vals = [(r["plain_cc1"], r["partial_cc1"]) for r in rows
                    if r["pair"] == f"{area_x}-{area_y}" and r["epoch"] == epoch
                    and r["role"] == "learner"]
            if vals:
                pl = np.nanmean([v[0] for v in vals])
                pa = np.nanmean([v[1] for v in vals])
                cells.append(f"{pl:.3f}->{pa:.3f}")
            else:
                cells.append("-")
        print(f"{lab:>22}  " + "  ".join(f"{c:>14}" for c in cells))


def _plot(rows, tag: str) -> None:
    learner = [r for r in rows if r["role"] == "learner"]
    fig, ax = plt.subplots(figsize=(6, 6))
    colours = {"DMS-DLS": "tab:blue", "DMS-ACC": "tab:orange",
               "DLS-ACC": "tab:green"}
    for pair, colour in colours.items():
        pr = [r for r in learner if r["pair"] == pair]
        ax.scatter([r["plain_cc1"] for r in pr], [r["partial_cc1"] for r in pr],
                   s=32, alpha=0.7, color=colour,
                   label=f"{pair} (n={len(pr)})")
    lim = [-0.15, 0.7]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("plain held-out CC1")
    ax.set_ylabel("partial held-out CC1 (third striatal area removed)")
    ax.set_title("Partial CCA — does striatal-cingulate coupling survive\n"
                 "removing the third area? (below diagonal = reduced)")
    ax.legend(frameon=False, fontsize=8)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"partial_cca_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
