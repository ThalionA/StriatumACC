"""Example neurons and publication statistics for the per-area encoding result.

Reads the per-neuron encoding results (``results/encoding_v6/``) and writes:

  figures/fig_encoding_stats.png     per-area % neurons encoding each latent,
                                     Wilson 95% CIs, binomial test vs chance.
  figures/fig_encoding_examples.png  example RPE-encoding neurons — firing and
                                     the RPE latent, and their RZ trial course.

Run:  python -m scripts.plot_encoding_detail
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.config import TaskConfig, RZ_MASK                          # noqa
from rl_model.io_real import load_real_cohort                            # noqa
from rl_model.neural_encoding import load_neural, LATENTS                 # noqa

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")
ENCDIR = os.path.join(RESDIR, "encoding_v6")
MAT = os.path.join(HERE, "..", "processed_data", "preprocessed_data5cm.mat")
LATENTS_NPZ = os.path.join(RESDIR, "rl_latents.npz")
ALPHA = 0.05
MODEL = "beh_spatial"                       # conservative model — headline
PLOT_AREAS = ("acc", "dms", "dls", "v1", "ca1", "dg")
COLOR = {"value": "#3457a6", "rpe": "#b03030", "precision": "#d9a300"}
RZ0, RZ1 = 25, 34                           # RZ bin slice


def wilson(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def load_results():
    """Per-neuron records pooled across mice."""
    recs = []
    for f in sorted(x for x in os.listdir(ENCDIR) if x.endswith(".npz")):
        mid = f[:-4]
        d = np.load(os.path.join(ENCDIR, f))
        for c in range(len(d["area"])):
            r = dict(mouse=mid, cell=c, area=str(d["area"][c]))
            for k in LATENTS:
                r[f"dR2_{k}"] = float(d[f"dR2_{MODEL}_{k}"][c])
                r[f"p_{k}"] = float(d[f"pval_{MODEL}_{k}"][c])
                r[f"pbin_{k}"] = float(d[f"pvalbin_{MODEL}_{k}"][c])
            recs.append(r)
    return recs


# --------------------------------------------------------------------------
def stats_figure(recs):
    areas = [a for a in PLOT_AREAS
             if sum(r["area"] == a for r in recs) > 0]
    print(f"\n{'area':5s} {'n':>5s} | " + " | ".join(
        f"{k:^30s}" for k in LATENTS))
    print(f"{'':5s} {'':>5s} | " + " | ".join(
        f"{'trial%(CI)  binomP  bin%  dR2':^30s}" for _ in LATENTS))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    width = 0.26
    xpos = np.arange(len(areas))
    for li, k in enumerate(LATENTS):
        fracs, los, his, stars = [], [], [], []
        for a in areas:
            sub = [r for r in recs if r["area"] == a]
            n = len(sub)
            ksig = sum(r[f"p_{k}"] < ALPHA for r in sub)
            frac = ksig / n
            lo, hi = wilson(ksig, n)
            bp = stats.binomtest(ksig, n, ALPHA, alternative="greater").pvalue
            fracs.append(frac * 100)
            los.append((frac - lo) * 100)
            his.append((hi - frac) * 100)
            stars.append("*" if bp < ALPHA else "")
        off = (li - 1) * width
        ax.bar(xpos + off, fracs, width, color=COLOR[k], label=k,
               yerr=[los, his], capsize=3, error_kw=dict(lw=1))
        for x, fr, hi_e, st in zip(xpos + off, fracs, his, stars):
            if st:
                ax.text(x, fr + hi_e + 1.5, st, ha="center", fontsize=12)

    # per-area text table
    for a in areas:
        sub = [r for r in recs if r["area"] == a]
        n = len(sub)
        cells = []
        for k in LATENTS:
            ksig = sum(r[f"p_{k}"] < ALPHA for r in sub)
            kbin = sum(r[f"pbin_{k}"] < ALPHA for r in sub)
            lo, hi = wilson(ksig, n)
            bp = stats.binomtest(ksig, n, ALPHA, alternative="greater").pvalue
            md = np.mean([r[f"dR2_{k}"] for r in sub])
            cells.append(f"{ksig/n*100:3.0f}%({lo*100:2.0f}-{hi*100:2.0f}) "
                         f"p={bp:.0e} bin{kbin/n*100:3.0f}% {md:+.3f}")
        print(f"{a:5s} {n:5d} | " + " | ".join(cells))
    ax.axhline(ALPHA * 100, c="0.5", ls="--", lw=1, label="chance (5%)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([a.upper() for a in areas])
    ax.set_ylabel("% neurons encoding (significant unique ΔR²)")
    ax.set_title("Per-area encoding of RL latents — conservative model "
                 "(behaviour + drift + spatial controlled)\n"
                 "error bars: Wilson 95% CI;  * binomial test vs chance p<0.05",
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_encoding_stats.png"), dpi=140)
    plt.close(fig)
    print("\nwrote fig_encoding_stats.png")


# --------------------------------------------------------------------------
def example_figure(recs):
    cfg = TaskConfig()
    cohort = load_real_cohort(MAT, cfg)
    neural = load_neural(MAT)
    lat_npz = np.load(LATENTS_NPZ, allow_pickle=True)
    mid_index = {m["mouse"]: i for i, m in enumerate(cohort)}
    rz = np.asarray(RZ_MASK)

    def rz_resid(M):
        """Per-bin demean (remove spatial tuning), then RZ-window average.
        NaN-aware — invalid bins are skipped."""
        d = M - np.nanmean(M, 0, keepdims=True)
        return np.nanmean(d[:, rz], 1)

    # top RPE encoders, two per striatal/cingulate area, skipping near-silent
    # neurons whose RZ residual has no usable variance
    picks = []
    for a in ("dms", "dls", "acc"):
        cand = sorted((r for r in recs if r["area"] == a
                       and r["p_rpe"] < ALPHA),
                      key=lambda r: -r["dR2_rpe"])
        got = 0
        for r in cand:
            i = mid_index[r["mouse"]]
            nt = cohort[i]["n_trials"]
            fr = neural[i]["fr"][:nt, :, r["cell"]]
            c = rz_resid(fr)
            if not np.isfinite(c).any() or np.nanstd(c) < 0.15:
                continue
            picks.append(r)
            got += 1
            if got == 2:
                break
    if not picks:
        print("no significant RPE neurons to show")
        return

    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.6 * n))
    if n == 1:
        axes = axes[None, :]
    for row, r in enumerate(picks):
        i = mid_index[r["mouse"]]
        nt = cohort[i]["n_trials"]
        fr = neural[i]["fr"][:nt, :, r["cell"]]            # (nt, 50)
        rpe = np.asarray(lat_npz["rpe"][i], dtype=float)    # (nt, 50)
        a0, a1, a2 = axes[row]

        lo, hi = np.nanpercentile(fr, [2, 98])
        a0.imshow(fr, aspect="auto", cmap="magma", vmin=lo, vmax=hi,
                  extent=[0, 50, nt, 0], interpolation="nearest")
        a0.axvspan(RZ0, RZ1, color="#1a7a3a", alpha=0.15)
        a0.set_title(f"{r['mouse']} {r['area'].upper()} cell {r['cell']} "
                     f"— z-firing", fontsize=9)
        a0.set_ylabel("trial", fontsize=8)
        a0.set_xlabel("spatial bin", fontsize=8)

        vr = np.nanpercentile(np.abs(rpe), 98)
        a1.imshow(rpe, aspect="auto", cmap="RdBu_r", vmin=-vr, vmax=vr,
                  extent=[0, 50, nt, 0], interpolation="nearest")
        a1.axvspan(RZ0, RZ1, color="#1a7a3a", alpha=0.15)
        a1.set_title("RPE latent", fontsize=9)
        a1.set_xlabel("spatial bin", fontsize=8)

        # trial-by-trial: RZ-window firing residual vs RPE residual
        rf, rr = rz_resid(fr), rz_resid(rpe)
        fin = np.isfinite(rf) & np.isfinite(rr)
        rval = float(np.corrcoef(rf[fin], rr[fin])[0, 1])
        a2.scatter(rr[fin], rf[fin], s=14, c="#b03030", alpha=0.5,
                   edgecolor="none")
        a2.set_title(f"RZ trial-by-trial  r={rval:+.2f}  "
                     f"ΔR²={r['dR2_rpe']:+.3f}  p={r['p_rpe']:.3f}", fontsize=9)
        a2.set_xlabel("RPE (RZ residual)", fontsize=8)
        a2.set_ylabel("firing (RZ residual)", fontsize=8)
    fig.suptitle("Example RPE-encoding neurons — firing residual tracks the "
                 "model's RPE latent, trial-by-trial in the RZ window",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGDIR, "fig_encoding_examples.png"), dpi=140)
    plt.close(fig)
    print("wrote fig_encoding_examples.png")


def main():
    recs = load_results()
    print(f"loaded {len(recs)} neurons from {ENCDIR}")
    stats_figure(recs)
    example_figure(recs)
    print("DONE")


if __name__ == "__main__":
    main()
