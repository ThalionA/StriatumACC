"""Band power across the corridor and across trials (reads the tables written by
run_lfp_position_trial.py; no statistics here).

Per cohort, for the raw maps and for the speed-residualised maps:
  * lfp_position_trial_<mouse>_<cohort>[_speedresid]  -- one animal: trial x
    position heatmaps, rows = areas, columns = running speed + four bands;
    learning point (red) and reward zone (dashed) marked; unusable trials blank.
  (reward zone 100-135 a.u. = 125-169 cm; the visual cue zone starts at 80 a.u. = 100 cm)
  * lfp_position_profiles_<cohort>[_speedresid]  -- position profile of each
    epoch; per animal the epoch's trials are averaged, then mean +- SEM across
    animals (the animal is the unit).
  * lfp_trial_evolution_<cohort>[_speedresid]  -- corridor-mean power against
    trial relative to the learning point, 5-trial blocks, mean +- SEM across
    animals.
Colour / y units: z of log10 band power, per channel over corridor + dark
(within-session scale; absolute power is not comparable across animals), then
averaged over the area's channels. Control-cohort learning points are the task
average (matched TIME windows, not a learning criterion).

    /opt/anaconda3/bin/python scripts/plot_lfp_position_trial.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config  # noqa: E402

BANDS = ("theta", "beta", "low_gamma", "high_gamma")
BAND_LABEL = {"theta": "theta 4-8 Hz", "beta": "beta 15-30 Hz",
              "low_gamma": "low gamma 30-80 Hz", "high_gamma": "high gamma 80-150 Hz"}
EPOCHS = ("Naive", "Intermediate", "Expert")
EPOCH_C = {"Naive": "#9a9a9a", "Intermediate": "#1f77b4", "Expert": "#d62728"}
BIN_CM = 5.0
REWARD_CM = (100 / 4 * BIN_CM, 135 / 4 * BIN_CM)   # 100-135 a.u. (ProcessStriatumTask.m)
VISUAL_CM = 80 / 4 * BIN_CM                        # visual cue zone starts at 80 a.u. (ProcessStriatumTask.m)
BLOCK = 5                                          # trials per block, trial-evolution figure
REL_RANGE = (-40, 40)                              # trials relative to the learning point
UNITS = "z log10 power"


def load_cohort(cohort: str) -> dict[int, dict]:
    """{mouse: {'areas': {area: (band, bin, trial) map}, 'speed', 'usable', 'lp_raw', epochs...}}
    merging the striatum and visual probes of each animal."""
    out: dict[int, dict] = {}
    for f in sorted((config.RESULTS_DIR / f"lfp_position_trial_{cohort}").glob("*.npz")):
        d = np.load(f)
        m = int(d["mouse_id"])
        rec = out.setdefault(m, {"raw": {}, "resid": {}, "speed": d["speed_cm_s"],
                                 "usable": d["usable"], "lp_raw": int(d["lp_raw"]),
                                 **{e: d[f"epoch_{e}"] for e in EPOCHS}})
        bands = [str(b) for b in d["bands"]]
        idx = [bands.index(b) for b in BANDS]
        for ai, area in enumerate(d["areas"]):
            rec["raw"][str(area)] = d["maps"][ai][idx]
            rec["resid"][str(area)] = d["maps_speed_resid"][ai][idx]
    return out


def _save(fig, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(config.FIGURES_DIR / f"{name}.{ext}",
                    dpi=min(150, 1600 / max(fig.get_size_inches())))
    plt.close(fig)


def _heat(ax, img, n_trials, vlim, cmap, usable):
    shown = np.full_like(img, np.nan)
    shown[:, usable] = img[:, usable]
    return ax.imshow(shown.T, aspect="auto", origin="lower", cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                     extent=(0, img.shape[0] * BIN_CM, 0.5, n_trials + 0.5), interpolation="nearest")


def plot_animal(mouse: int, rec: dict, cohort: str, kind: str) -> None:
    areas = [a for a in config.AREAS if a in rec[kind]]
    n_tr = rec["speed"].shape[1]
    usable = rec["usable"]
    fig, axes = plt.subplots(len(areas), 1 + len(BANDS), figsize=(15, 2.4 * len(areas) + 1.3),
                             squeeze=False, constrained_layout=True)
    lims = {}
    for b in range(len(BANDS)):
        vals = np.concatenate([rec[kind][a][b][:, usable].ravel() for a in areas])
        v = np.nanpercentile(np.abs(vals), 98)
        lims[b] = (-v, v)
    for r, area in enumerate(areas):
        ax = axes[r, 0]
        if r == 0:
            im = _heat(ax, rec["speed"], n_tr, np.nanpercentile(rec["speed"][:, usable], [2, 98]),
                       "viridis", usable)
            fig.colorbar(im, ax=ax, label="speed (cm/s)")
            ax.set_title("running speed")
        else:
            ax.axis("off")
        for b, band in enumerate(BANDS):
            ax = axes[r, b + 1]
            im = _heat(ax, rec[kind][area][b], n_tr, lims[b], "RdBu_r", usable)
            if r == 0:
                ax.set_title(BAND_LABEL[band])
            if b == len(BANDS) - 1:
                fig.colorbar(im, ax=ax, label=UNITS)
            ax.set_ylabel(f"{area}\ntrial" if b == 0 else "")
        for ax in axes[r]:
            if not ax.axison:
                continue
            for x in REWARD_CM:
                ax.axvline(x, color="k", ls="--", lw=0.7)
            ax.axvline(VISUAL_CM, color="#7e2f8e", ls=":", lw=1.0)
            if rec["lp_raw"] >= 0:
                ax.axhline(rec["lp_raw"] + 1, color="#d62728", lw=1.2)
            ax.set_xlabel("position (cm)" if r == len(areas) - 1 else "")
    lp_note = "learning point" if cohort == "task" else "task-average learning point (matched time)"
    fig.suptitle(f"{cohort} {mouse}: LFP band power by trial and position"
                 f"{' (speed-residualised)' if kind == 'resid' else ''}\n"
                 f"{UNITS} per channel over corridor + dark, channel mean; red line = {lp_note}; "
                 f"dashed = reward zone; purple dotted = visual cue zone start; {usable.size} usable trials")
    _save(fig, f"lfp_position_trial_{mouse}_{cohort}{'_speedresid' if kind == 'resid' else ''}")


def _mean_sem(stack):
    stack = np.asarray(stack, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = np.sum(np.isfinite(stack), axis=0)
        return np.nanmean(stack, 0), np.nanstd(stack, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), n


def plot_profiles(data: dict, cohort: str, kind: str) -> None:
    areas = [a for a in config.AREAS if any(a in rec[kind] for rec in data.values())]
    x = (np.arange(50) + 0.5) * BIN_CM
    fig, axes = plt.subplots(len(areas), 1 + len(BANDS), figsize=(15, 2.3 * len(areas) + 1.3),
                             squeeze=False, constrained_layout=True)
    for r, area in enumerate(areas):
        recs = [rec for rec in data.values() if area in rec[kind]]
        for c in range(1 + len(BANDS)):
            ax = axes[r, c]
            for e in EPOCHS:
                prof = []
                for rec in recs:
                    tr = rec[e]
                    if tr.size == 0:
                        continue
                    src = rec["speed"] if c == 0 else rec[kind][area][c - 1]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        prof.append(np.nanmean(src[:, tr], axis=1))
                if not prof:
                    continue
                mu, se, _ = _mean_sem(prof)
                ax.plot(x, mu, color=EPOCH_C[e], lw=1.4, label=f"{e} (N={len(prof)})")
                ax.fill_between(x, mu - se, mu + se, color=EPOCH_C[e], alpha=0.2, lw=0)
            ax.axvspan(*REWARD_CM, color="0.9", zorder=0)
            ax.axvline(VISUAL_CM, color="#7e2f8e", ls=":", lw=1.0)
            ax.set_title(f"{area}: {'running speed' if c == 0 else BAND_LABEL[BANDS[c - 1]]}", fontsize=9)
            ax.set_ylabel("speed (cm/s)" if c == 0 else UNITS, fontsize=8)
            ax.set_xlabel("position (cm)" if r == len(areas) - 1 else "")
            if c == 0:
                ax.legend(fontsize=6.5)
    fig.suptitle(f"{cohort}: band power across the corridor by epoch"
                 f"{' (speed-residualised)' if kind == 'resid' else ''} -- per animal the epoch's "
                 f"trials are averaged, then mean +- SEM across animals;\ngrey band = reward zone, purple dotted = visual cue zone start")
    _save(fig, f"lfp_position_profiles_{cohort}{'_speedresid' if kind == 'resid' else ''}")


def plot_trial_evolution(data: dict, cohort: str, kind: str) -> None:
    areas = [a for a in config.AREAS if any(a in rec[kind] for rec in data.values())]
    edges = np.arange(REL_RANGE[0], REL_RANGE[1] + BLOCK, BLOCK)
    centres = edges[:-1] + BLOCK / 2
    fig, axes = plt.subplots(len(areas), 1 + len(BANDS), figsize=(15, 2.3 * len(areas) + 1.3),
                             squeeze=False, constrained_layout=True)
    for r, area in enumerate(areas):
        recs = [rec for rec in data.values() if area in rec[kind] and rec["lp_raw"] >= 0]
        for c in range(1 + len(BANDS)):
            ax = axes[r, c]
            curves = []
            for rec in recs:
                src = rec["speed"] if c == 0 else rec[kind][area][c - 1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    per_trial = np.nanmean(src, axis=0)          # corridor mean per trial
                rel = rec["usable"] - rec["lp_raw"]
                vals = per_trial[rec["usable"]]
                block = np.full(centres.size, np.nan)
                for k in range(centres.size):
                    m = (rel >= edges[k]) & (rel < edges[k + 1])
                    if m.sum() >= 2:
                        block[k] = np.nanmean(vals[m])
                curves.append(block)
            if curves:
                mu, se, n = _mean_sem(curves)
                mu[n < 2] = np.nan
                ax.plot(centres, mu, "-o", ms=3, color="k", lw=1.2)
                ax.fill_between(centres, mu - se, mu + se, color="k", alpha=0.15, lw=0)
            ax.axvline(0, color="#d62728", lw=1)
            ax.set_title(f"{area}: {'running speed' if c == 0 else BAND_LABEL[BANDS[c - 1]]} "
                         f"(N={len(recs)})", fontsize=9)
            ax.set_ylabel("speed (cm/s)" if c == 0 else UNITS, fontsize=8)
            ax.set_xlabel("trial relative to learning point" if r == len(areas) - 1 else "")
    fig.suptitle(f"{cohort}: corridor-mean band power across trials"
                 f"{' (speed-residualised)' if kind == 'resid' else ''} -- {BLOCK}-trial blocks "
                 f"aligned to the learning point (red), mean +- SEM across animals (blocks with N >= 2)")
    _save(fig, f"lfp_trial_evolution_{cohort}{'_speedresid' if kind == 'resid' else ''}")


def main() -> None:
    config.FIGURES_DIR.mkdir(exist_ok=True)
    for cohort in ("task", "control"):
        data = load_cohort(cohort)
        for kind in ("raw", "resid"):
            for mouse, rec in data.items():
                plot_animal(mouse, rec, cohort, kind)
            plot_profiles(data, cohort, kind)
            plot_trial_evolution(data, cohort, kind)
        print(f"{cohort}: {len(data)} animals")


if __name__ == "__main__":
    main()
