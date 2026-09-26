"""CCA with movement regressed out: per animal-pair-epoch held-out CC1 with the
real confound vs the median of its 10 shifted controls (below the diagonal =
coupling carried by movement), per arm; and the naive -> expert CC1 change for
plain vs movement-removed CCA (learners only).

Usage: python scripts/plot_cca_movement.py
Reads results/cca_movement.npz; writes figures/cca_movement.{svg,png}.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cca_movement import CONTROL_FRACTIONS, effects

ARMS = ("spatial", "temporal")
BASES = {"vr": "VR speed + licks", "vr_video": "VR + video (4 ROI ME + 10 face SVD)"}


def learning_change(t, arm, variant):
    m = (t["arm"] == arm) & (t["role"] == "learner") & (t["variant"] == variant)
    out = []
    for s, p in sorted(set(zip(t["session"][m], t["pair"][m]))):
        sel = m & (t["session"] == s) & (t["pair"] == p)
        v = dict(zip(t["epoch"][sel], t["cc1"][sel]))
        out.append(v["expert"] - v["naive"])
    return np.array(out)


def main():
    t = np.load(ROOT / "results" / "cca_movement.npz", allow_pickle=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
    for r, arm in enumerate(ARMS):
        for c, (base, label) in enumerate(BASES.items()):
            e = effects(t, arm, base)
            e = e[np.isfinite(e).all(1)]
            a = ax[r, c]
            below = e[:, 3] == 0
            a.scatter(e[~below, 2], e[~below, 1], s=18, c="0.4", label="within control range")
            a.scatter(e[below, 2], e[below, 1], s=24, c="C3",
                      label=f"below all {len(CONTROL_FRACTIONS)} controls: {below.mean():.2f} "
                            f"(chance {1 / (len(CONTROL_FRACTIONS) + 1):.2f})")
            lim = [min(e[:, 1:3].min(), 0) - 0.02, e[:, 1:3].max() + 0.02]
            a.plot(lim, lim, "k--", lw=0.8)
            a.set(xlim=lim, ylim=lim, xlabel="held-out CC1, shifted-confound controls (median)",
                  ylabel="held-out CC1, real confound removed",
                  title=f"{arm} arm: {label} removed\n(one point per animal × pair × epoch, n={e.shape[0]})")
            a.legend(fontsize=7, loc="upper left")
        a = ax[r, 2]
        plain, vv = learning_change(t, arm, "plain"), learning_change(t, arm, "vr_video")
        for i in range(plain.size):
            a.plot([0, 1], [plain[i], vv[i]], "-o", color="0.5", ms=4)
        a.axhline(0, color="k", lw=0.8)
        a.set(xticks=[0, 1], xticklabels=["plain", "VR + video removed"], xlim=(-0.4, 1.4),
              ylabel="held-out CC1: expert − naive",
              title=f"{arm} arm: learning change\nper learner animal-pair (n={plain.size})")
    fig.suptitle("Does shared movement drive cross-area CCA? Four video animals, committed cca pipeline "
                 "(plain CCA; confound regressed out of residual neuron tensors before PCA)")
    for ext in ("svg", "png"):
        fig.savefig(ROOT / "figures" / f"cca_movement.{ext}", dpi=105)  # 15 in x 105 dpi = 1575 px
    print("wrote figures/cca_movement.svg/.png")


if __name__ == "__main__":
    main()
