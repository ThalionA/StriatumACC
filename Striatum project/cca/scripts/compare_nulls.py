"""Compare the trial-permutation and circshift surrogate nulls.

Both committed-config runs (3 epochs, LP-7) use identical data and CCA -- only
the surrogate differs -- so any difference in the significant-dimension counts,
and hence in the CC / IFI pooled over those dimensions, is purely the null's
effect. Reports per area-pair x epoch.

Writes figures/null_comparison.csv and figures/null_comparison.png.

Run:  python scripts/compare_nulls.py
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from striatum_cca import config  # noqa: E402

ALPHA = 0.05
EPOCHS = config.EPOCH_NAMES
PAIR_NAMES = [f"{ax}-{ay}" for ax, ay in config.PAIRS]
NULLS = ("trials", "circshift")


def _load(null):
    path = config.RESULTS_DIR / f"stage2_committed_{null}.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)["results"]


def _sig(ea):
    return np.where(ea.p_per_dim < ALPHA)[0]


def main():
    res = {n: _load(n) for n in NULLS}
    rows = []
    for (ax, ay), name in zip(config.PAIRS, PAIR_NAMES):
        learners = {n: [r for r in res[n]
                        if (r.area_x, r.area_y) == (ax, ay)
                        and r.role == "learner"] for n in NULLS}
        for e in EPOCHS:
            cell = {"pair": name, "epoch": e,
                    "n_learn": len(learners["trials"])}
            for n in NULLS:
                pooled_cc, pooled_ifi, n_sig = [], [], 0
                for r in learners[n]:
                    js = _sig(r.epochs[e])
                    n_sig += len(js)
                    pooled_cc.extend(r.epochs[e].held_out_cc[js].tolist())
                    pooled_ifi.extend(r.epochs[e].ifi_windows[js, 2].tolist())
                cc = np.array([v for v in pooled_cc if np.isfinite(v)])
                ifi = np.array([v for v in pooled_ifi if np.isfinite(v)])
                cell[f"nsig_{n}"] = n_sig
                cell[f"cc_{n}"] = float(np.mean(cc)) if cc.size else np.nan
                cell[f"ifi_{n}"] = float(np.mean(ifi)) if ifi.size else np.nan
            rows.append(cell)

    cols = ["pair", "epoch", "n_learn", "nsig_trials", "nsig_circshift",
            "cc_trials", "cc_circshift", "ifi_trials", "ifi_circshift"]
    with open(config.FIGURES_DIR / "null_comparison.csv", "w",
              newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for r in rows:
            wr.writerow(["" if isinstance(r.get(c), float)
                         and not np.isfinite(r[c])
                         else (round(r[c], 4) if isinstance(r.get(c), float)
                               else r.get(c)) for c in cols])
    print("saved figures/null_comparison.csv")

    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    axes = axes.ravel()
    x = np.arange(len(EPOCHS))
    for ax, name in zip(axes, PAIR_NAMES):
        pr = [r for r in rows if r["pair"] == name]
        ax.bar(x - 0.2, [r["nsig_trials"] for r in pr], 0.4,
               color="tab:blue", label="trial-perm")
        ax.bar(x + 0.2, [r["nsig_circshift"] for r in pr], 0.4,
               color="tab:orange", label="circshift")
        ax.set_xticks(x)
        ax.set_xticklabels(["naive", "inter", "expert"], fontsize=7)
        ax.set_title(f"{name}  (n={pr[0]['n_learn']})", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes[::5]:
        ax.set_ylabel("# significant dims", fontsize=8)
    fig.suptitle("Significant subspace dimensions per pair x epoch -- "
                 "trial-permutation vs circshift null (committed config, LP-7)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "null_comparison.png", dpi=150)
    plt.close(fig)
    print("saved figures/null_comparison.png")

    tt = sum(r["nsig_trials"] for r in rows)
    tc = sum(r["nsig_circshift"] for r in rows)
    print(f"\ntotal significant dims:  trial-perm = {tt}   circshift = {tc}")
    print(f"\n{'pair':>9} {'epoch':>12}  {'nsig t/c':>10}  {'cc t/c':>15}")
    for r in rows:
        cc = (f"{r['cc_trials']:.3f}/{r['cc_circshift']:.3f}"
              if np.isfinite(r["cc_trials"]) and np.isfinite(r["cc_circshift"])
              else "  -  ")
        print(f"  {r['pair']:>9} {r['epoch']:>12}  "
              f"{r['nsig_trials']:>4}/{r['nsig_circshift']:<4}  {cc:>15}")


if __name__ == "__main__":
    main()
