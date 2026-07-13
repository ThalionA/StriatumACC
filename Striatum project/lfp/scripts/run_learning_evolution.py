"""LFP band-power evolution across learning -- scale-aware, artefact-masked.

PROVISIONAL / provenance-gated. Voltage scale, input band, referencing and the
exact sample<->ms offset are undocumented, so every result is RELATIVE (z-scored
log power within a session, or a dimensionless ratio). Running vs stationary is
split by *speed* (animals also run during dark), not by corridor/dark state.

Diagnostic override only::

    /opt/anaconda3/bin/python scripts/run_learning_evolution.py --allow-unaligned

Mouse 1212 is excluded because its voltage-probe identity is unverified. The
30-80 Hz band is excluded because the persistent ~74 Hz peak confounds it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tcca" / "src"))

from striatum_lfp import config, geometry, learning  # noqa: E402
from striatum_lfp.reader import DATASET  # noqa: E402
from striatum_tcca import config as tcfg  # noqa: E402
from striatum_tcca import dataio as tdataio  # noqa: E402

FS = config.FS
BANDS = {"theta": (4, 8), "beta": (15, 30)}
AREAS = ["DMS", "DLS", "ACC"]
AREA_COL = {"DMS": "#4C72B0", "DLS": "#55A868", "ACC": "#DD8452"}
RUN_SPEED = 2.0        # cm/s running gate (project standard)
_ANIMALS = {}


def animal_info(mouse_id, n_corr):
    if not _ANIMALS:
        for a in tdataio.load_animals(group="task"):
            _ANIMALS[a.animal_id] = a
    try:
        animal_id = config.TASK_MOUSE_IDS.index(mouse_id) + 1
    except ValueError as exc:
        raise KeyError(f"mouse {mouse_id} absent from TASK_MOUSE_IDS") from exc
    a = _ANIMALS[animal_id]
    if abs(a.n_trials - n_corr) > 1:
        raise ValueError(
            f"mouse {mouse_id}: raw corridor trials {n_corr} disagree with "
            f"preprocessed animal {animal_id} trials {a.n_trials}"
        )
    lp = tdataio.find_learning_point(a.zscored_lick_errors, tcfg.DEFAULT)
    cp = getattr(a, "change_point", None)
    cp = None if cp is None or not np.isfinite(cp) else float(cp)
    return lp, np.asarray(a.zscored_lick_errors, dtype=float), cp


def area_channels(mouse_id):
    depths = geometry.channel_depths()
    masks = geometry.channel_area_masks(depths, geometry.load_area_boundaries(mouse_id, "striatum"))
    return {ar: np.where(masks[ar])[0] for ar in AREAS if ar in masks}


def welch_all(seg):
    seg = seg - seg.mean(axis=0, keepdims=True)
    return welch(seg, fs=FS, nperseg=min(1024, seg.shape[0]), axis=0)


def band_integ(freqs, psd, band):
    m = (freqs >= band[0]) & (freqs < band[1])
    return np.trapz(psd[m], freqs[m], axis=0)


def frame_speed(pos, vt):
    """Per-frame speed (cm/s) from VR position; trial resets masked out."""
    speed = np.full(pos.shape, np.nan)
    dt, dp = np.diff(vt), np.diff(pos) * config.AU_TO_CM
    raw = np.abs(dp) / np.where(dt > 0, dt, np.nan)
    raw[np.abs(dp) > 50] = np.nan
    speed[1:] = raw
    return speed


def process(mouse_id):
    with h5py.File(config.RAW_MAT[mouse_id], "r") as f:
        VR = np.asarray(f["VR_data"][()])
        vt = np.asarray(f["VR_times_synched"][()]).ravel()
    pos, world, trial = VR[:, 1], VR[:, 4], VR[:, 6]
    frame_ms = np.round(vt * 1000).astype(int)
    speed = frame_speed(pos, vt)
    n_corr = int(np.unique(trial[world == 6]).size)
    lp, licks, change_point = animal_info(mouse_id, n_corr)
    traversals = learning.corridor_traversals(
        world, trial, frame_ms, min_frames=20
    )
    achan = area_channels(mouse_id)
    handle = h5py.File(config.LFP_DIR / config.FILE_BY_MOUSE[mouse_id], "r")
    lf = handle[DATASET]
    n = lf.shape[0]
    traversals = learning.valid_traversals(
        traversals, n_samples=n, min_samples=500
    )

    # --- per-traversal per-area band power (+ peak) ---
    trials, peaks = [], []
    area_band = {f"{ar}_{bd}": [] for ar in achan for bd in BANDS}
    for tv in traversals:
        a, b = tv.start_ms, tv.end_ms
        seg = np.asarray(lf[a:b, :], dtype=float)
        freqs, psd = welch_all(seg)
        trials.append(tv.trial)
        peaks.append(float(np.max(np.abs(seg))))
        for ar, chs in achan.items():
            for bd, band in BANDS.items():
                area_band[f"{ar}_{bd}"].append(float(np.median(band_integ(freqs, psd, band)[chs])))
    trials, peaks = np.array(trials), np.array(peaks)
    for k in area_band:
        area_band[k] = np.array(area_band[k])
    artefact = learning.robust_log_outliers(peaks, z_threshold=4.0)
    clean = ~artefact
    art_thr = 10 ** (np.nanmedian(np.log10(peaks[clean])) + 3) if clean.any() else np.inf

    # --- validation 1: per-area PSD from up to 40 clean traversals ---
    ci = np.where(clean)[0]
    sel = ci[np.linspace(0, len(ci) - 1, min(40, len(ci))).astype(int)] if len(ci) else []
    fref, psd_acc = None, {ar: [] for ar in achan}
    for i in sel:
        tv = traversals[i]
        seg = np.asarray(lf[max(0, tv.start_ms):min(n, tv.end_ms), :], dtype=float)
        fref, psd = welch_all(seg)
        for ar, chs in achan.items():
            psd_acc[ar].append(np.median(psd[:, chs], axis=1))
    area_psd = {ar: (np.median(np.array(v), axis=0) if v else None) for ar, v in psd_acc.items()}

    # --- validation 2: theta running vs stationary (speed-split 1 s windows) ---
    run = {ar: [] for ar in achan}
    sta = {ar: [] for ar in achan}
    for s in np.linspace(0, n - FS, 400).astype(int):
        seg = np.asarray(lf[s:s + FS, :], dtype=float)
        if np.max(np.abs(seg)) > art_thr:
            continue
        fmask = (frame_ms >= s) & (frame_ms < s + FS)
        if fmask.sum() < 3:
            continue
        spd = np.nanmean(speed[fmask])
        if not np.isfinite(spd):
            continue
        freqs, psd = welch_all(seg)
        th = band_integ(freqs, psd, BANDS["theta"])
        tgt = run if spd >= RUN_SPEED else sta
        for ar, chs in achan.items():
            tgt[ar].append(float(np.median(th[chs])))
    handle.close()
    runstat = {ar: dict(run=np.median(run[ar]) if run[ar] else np.nan,
                        sta=np.median(sta[ar]) if sta[ar] else np.nan,
                        n_run=len(run[ar]), n_sta=len(sta[ar])) for ar in achan}
    return dict(mouse=mouse_id, lp=lp, licks=licks, change_point=change_point,
                trials=trials, clean=clean, area_band=area_band, achan=achan,
                area_psd=area_psd, fref=fref, runstat=runstat)


MA_WINDOW = 10          # moving-average window for the evolution lines (<= 10)
EPOCH_ORDER = ["naive", "intermediate", "expert", "disengaged"]


def zlog_clean(power, clean):
    x = np.asarray(power, float)
    lx = np.full(x.shape, np.nan)
    lx[x > 0] = np.log10(x[x > 0])
    mu, sd = np.nanmean(lx[clean]), np.nanstd(lx[clean])
    return (lx - mu) / sd if sd > 0 else lx * 0.0


def build_epochs(res):
    """LP-aligned naive/intermediate/expert + a disengaged epoch (post change-point)."""
    n_trials = len(res["licks"]) if res["licks"] is not None else int(res["trials"].max())
    phase_idx, scheme = learning.phase_indices(n_trials, res["lp"])
    epochs = {k: phase_idx[k] for k in ("naive", "intermediate", "expert")}
    if res["change_point"] is not None:
        epochs["disengaged"] = np.arange(int(res["change_point"]) - 1, n_trials)
    return epochs, scheme, n_trials


def epoch_z_table(res, epochs):
    """{band: {area: {epoch: (mean, sem, n)}}} of z-scored log power (clean traversals)."""
    tr0 = res["trials"] - 1
    out = {bd: {ar: {} for ar in res["achan"]} for bd in BANDS}
    for bd in BANDS:
        for ar in res["achan"]:
            z = zlog_clean(res["area_band"][f"{ar}_{bd}"], res["clean"])
            for ep, idx in epochs.items():
                vals = z[res["clean"] & np.isin(tr0, idx) & np.isfinite(z)]
                out[bd][ar][ep] = (float(np.mean(vals)) if vals.size else np.nan,
                                   float(vals.std() / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
                                   int(vals.size))
    return out


def area_avg_epoch(table, band, epoch):
    """Mean z log-power across areas for a band/epoch (areas track together)."""
    vals = [table[band][ar][epoch][0] for ar in table[band] if epoch in table[band][ar]]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def plot_mouse(res, epoch_table, summary_rows):
    mid, achan, area_band = res["mouse"], res["achan"], res["area_band"]
    trials, clean, lp, licks, cp = res["trials"], res["clean"], res["lp"], res["licks"], res["change_point"]
    epochs, scheme, n_trials = build_epochs(res)
    ep_names = [e for e in EPOCH_ORDER if e in epochs]
    tr0 = trials - 1

    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(
        5, 3, height_ratios=[0.8, 1, 1, 1.05, 1.05], hspace=0.5, wspace=0.3
    )

    def marks(ax):
        if lp is not None:
            ax.axvline(lp, color="k", ls="--", lw=1.1, label="LP")
        if cp is not None:
            ax.axvline(cp, color="firebrick", ls=":", lw=1.3, label="disengage")

    # row 0: behaviour
    ax = fig.add_subplot(gs[0, :])
    if licks is not None:
        ax.plot(np.arange(1, len(licks) + 1), licks, color="0.35", lw=0.9)
        ax.axhline(tcfg.DEFAULT.lp_z_threshold, color="r", ls=":", lw=0.8)
    marks(ax)
    ax.set_ylabel("z lick error"); ax.set_xlim(0, n_trials + 1)
    ax.set_title(f"Mouse {mid} — behaviour (LP trial {lp if lp is not None else 'NA'}, "
                 f"disengage {int(cp) if cp else 'none'}); phases: {scheme}")
    ax.legend(fontsize=7, loc="upper right")

    # rows 1-3: band evolution (moving average window <= 10 + shaded SEM)
    for bi, bd in enumerate(BANDS):
        ax = fig.add_subplot(gs[1 + bi, :])
        for ar in achan:
            y = zlog_clean(area_band[f"{ar}_{bd}"], clean)
            tc = trials[clean]; order = np.argsort(tc)
            tcs = tc[order]
            mean, sem = learning.rolling_mean_sem(y[clean][order], window=MA_WINDOW)
            ax.plot(tcs, mean, color=AREA_COL[ar], lw=1.5, label=ar)
            ax.fill_between(tcs, mean - sem, mean + sem, color=AREA_COL[ar], alpha=0.2, lw=0)
        marks(ax)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_ylabel(f"{bd}\nz log-power"); ax.set_xlim(0, n_trials + 1)
        if bi == 0:
            ax.set_title(
                f"Band-power evolution ({MA_WINDOW}-trial moving average ± "
                "within-window SEM; descriptive)"
            )
            ax.legend(fontsize=8, ncol=len(achan) + 2, loc="upper right")

    # row 3: PSD + theta run/stationary
    ax = fig.add_subplot(gs[3, 0])
    for ar in achan:
        p = res["area_psd"][ar]
        if p is not None and res["fref"] is not None:
            b = (res["fref"] >= 1) & (res["fref"] <= 120)
            ax.loglog(res["fref"][b], p[b], color=AREA_COL[ar], lw=1.2, label=ar)
    for lo, hi, c in [(4, 8, "green"), (15, 30, "blue"), (30, 80, "purple")]:
        ax.axvspan(lo, hi, alpha=0.07, color=c)
    ax.set_xlabel("Hz"); ax.set_ylabel("PSD (stored u²/Hz)")
    ax.set_title("Per-area PSD (validation)"); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[3, 1])
    xs = np.arange(len(achan))
    ax.bar(xs, [res["runstat"][ar]["run"] / res["runstat"][ar]["sta"]
               if np.isfinite(res["runstat"][ar]["sta"]) and res["runstat"][ar]["sta"] > 0 else np.nan
               for ar in achan], color=[AREA_COL[ar] for ar in achan])
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(list(achan))
    ax.set_ylabel("theta running / stationary")
    ax.set_title("Theta: running vs stationary\n(>1 = higher when running)")
    fig.add_subplot(gs[3, 2]).axis("off")

    # row 4: approved diagnostic bands across epochs
    width = 0.8 / max(1, len(achan))
    for bi, bd in enumerate(BANDS):
        ax = fig.add_subplot(gs[4, bi])
        for ai, ar in enumerate(achan):
            means = [epoch_table[bd][ar][ep][0] for ep in ep_names]
            sems = [epoch_table[bd][ar][ep][1] for ep in ep_names]
            ax.bar(np.arange(len(ep_names)) + ai * width, means, width, yerr=sems,
                   color=AREA_COL[ar], label=ar, capsize=2)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xticks(np.arange(len(ep_names)) + width * (len(achan) - 1) / 2)
        ax.set_xticklabels(ep_names, rotation=20, fontsize=8)
        ax.set_ylabel("z log-power"); ax.set_title(f"{bd} across epochs")
        if bi == 0:
            ax.legend(fontsize=7)
    fig.add_subplot(gs[4, 2]).axis("off")

    for ext in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"learning_evolution_{mid}.{ext}", dpi=110)
    plt.close(fig)

    # summary rows (raw log10 per epoch incl. disengaged)
    for ar in achan:
        rs = res["runstat"][ar]
        ratio = rs["run"] / rs["sta"] if np.isfinite(rs["sta"]) and rs["sta"] > 0 else np.nan
        for bd in BANDS:
            v = area_band[f"{ar}_{bd}"]
            lv = np.full(len(v), np.nan); lv[v > 0] = np.log10(v[v > 0])
            def epm(name):
                if name not in epochs:
                    return np.nan
                s = clean & np.isin(tr0, epochs[name]) & np.isfinite(lv)
                return float(np.nanmean(lv[s])) if s.any() else np.nan
            nai, inter, exp, dis = epm("naive"), epm("intermediate"), epm("expert"), epm("disengaged")
            summary_rows.append(dict(
                mouse=mid, area=ar, band=bd, scheme=scheme,
                lp_trial=lp,
                disengage_trial=(int(cp) if cp else None), n_clean=int(clean.sum()),
                naive_log10=round(nai, 3) if np.isfinite(nai) else np.nan,
                intermediate_log10=round(inter, 3) if np.isfinite(inter) else np.nan,
                expert_log10=round(exp, 3) if np.isfinite(exp) else np.nan,
                disengaged_log10=round(dis, 3) if np.isfinite(dis) else np.nan,
                expert_over_naive_x=(round(10 ** (exp - nai), 3)
                                     if np.isfinite(exp) and np.isfinite(nai) else np.nan),
                theta_run_over_stationary=(round(ratio, 3) if bd == "theta" and np.isfinite(ratio) else ""),
            ))


def plot_summary_across_animals(recs):
    """recs = [(res, epoch_table)]. Cross-animal: per-band area-averaged z log-power
    across epochs (one line per animal + mean), and theta running/stationary per animal."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    mcol = {1212: "#8172B3", 614: "#4C72B0", 727: "#DD8452", 731: "#55A868"}
    xall = np.arange(len(EPOCH_ORDER))
    band_ax = {"theta": axes[0], "beta": axes[1]}
    for bd, ax in band_ax.items():
        stacks = {ep: [] for ep in EPOCH_ORDER}
        for res, tbl in recs:
            epochs = build_epochs(res)[0]
            ys = [area_avg_epoch(tbl, bd, ep) if ep in epochs else np.nan for ep in EPOCH_ORDER]
            caveat = res["mouse"] == 1212
            ax.plot(xall, ys, "--" if caveat else "-", marker="o", color=mcol[res["mouse"]],
                    lw=1.3, label=f"{res['mouse']}{' (caveat)' if caveat else ''}")
            for ep, y in zip(EPOCH_ORDER, ys):
                if np.isfinite(y) and not caveat:
                    stacks[ep].append(y)
        mean = [np.mean(stacks[ep]) if stacks[ep] else np.nan for ep in EPOCH_ORDER]
        ax.plot(xall, mean, "k-", lw=2.6, alpha=0.55, label="mean (excl 1212)")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xticks(xall); ax.set_xticklabels(EPOCH_ORDER, rotation=20)
        ax.set_ylabel("area-avg z log-power"); ax.set_title(f"{bd} across epochs")
        if bd == "theta":
            ax.legend(fontsize=7)
    ax = axes[2]
    for i, (res, _) in enumerate(recs):
        rr = [res["runstat"][ar]["run"] / res["runstat"][ar]["sta"] for ar in res["achan"]
              if np.isfinite(res["runstat"][ar]["sta"]) and res["runstat"][ar]["sta"] > 0]
        ax.bar(i, np.mean(rr) if rr else np.nan, color=mcol[res["mouse"]])
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(recs))); ax.set_xticklabels([str(r[0]["mouse"]) for r in recs])
    ax.set_ylabel("theta running / stationary"); ax.set_title("Theta run vs stationary (area-avg)")
    fig.suptitle("LFP band power across learning/engagement — across-animal summary (PROVISIONAL)", fontsize=13)
    for ext in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"learning_evolution_summary_animals.{ext}", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-unaligned",
        action="store_true",
        help="Run provisional analysis despite unresolved voltage/VR offset.",
    )
    args = parser.parse_args()
    if not args.allow_unaligned:
        raise SystemExit(
            "Refusing learning analysis: voltage-to-VR offset and source band are "
            "unverified. Re-run only for diagnostics with --allow-unaligned."
        )
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, recs = [], []
    analysis_mice = tuple(mid for mid in config.LFP_MICE if mid != 1212)
    print(
        "[learning] excluding mouse 1212: voltage-probe identity is unverified",
        flush=True,
    )
    for mid in analysis_mice:
        print(f"[learning] mouse {mid} ...", flush=True)
        res = process(mid)
        epochs = build_epochs(res)[0]
        table = epoch_z_table(res, epochs)
        rs = res["runstat"]
        print(f"           LP={res['lp']} disengage={res['change_point']} clean={int(res['clean'].sum())}/{len(res['trials'])} "
              f"theta run/stat: " + ", ".join(
                  f"{ar}={rs[ar]['run']/rs[ar]['sta']:.2f}" if np.isfinite(rs[ar]['sta']) and rs[ar]['sta'] > 0 else f"{ar}=NA"
                  for ar in res["achan"]), flush=True)
        plot_mouse(res, table, rows)
        recs.append((res, table))
    out = config.RESULTS_DIR / "learning_evolution_summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot_summary_across_animals(recs)
    print(f"[done] {out} + figures/learning_evolution_*.png + summary_animals")


if __name__ == "__main__":
    main()
