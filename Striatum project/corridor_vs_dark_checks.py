"""Two checks on the corridor-vs-dark figure (Fig S1), raised at the 2026-08-28 meeting.

1. Running velocity during the 5 s dark inter-trial period, per animal and epoch,
   against the corridor traversal. The dark period was assumed stationary; the VR
   position keeps integrating the wheel in the dark, so this is measurable.
2. What the z-scoring convention does to the dark-vs-corridor contrast. The
   figure z-scores each unit over corridor AND dark samples jointly ("common").
   The alternatives a reader might assume -- z-scoring each state on its own
   ("per-state"), or reusing SpatioTemporalActivityEvolution's corridor-only
   normalisation and the analogous dark-only one ("ST-style") -- are computed
   here so the figure can say which contrasts are convention-free.

Reads the v7.3 caches directly through h5py (only the small fields), so it does
not need MATLAB or the tens-of-GB full load. Epoch and learning-point rules are
ports of `epoch_indices.m` / `find_learning_points.m` (naive_split = 3,
trials_per_epoch = 10, LP = first sub-threshold trial with >= 7 of the next 10
below z = -2); the learner count and mean LP are printed and must match the
MATLAB run (14/16, mean LP 41.0 on 2026-08-27).

Outputs (figures/):
  CorridorVsDark_velocity.{svg,png}          dark vs corridor speed by epoch
  CorridorVsDark_zscore_variants.{svg,png}   corridor - dark contrast, 3 conventions
  corridor_vs_dark_velocity_by_animal.csv    animal x epoch speeds (cm/s)
  corridor_vs_dark_zscore_variants.csv       animal x area x epoch contrasts

Created 2026-09-07. Run from `Striatum project/`:
  /opt/anaconda3/bin/python corridor_vs_dark_checks.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
AU_TO_CM = 1.25                     # project_cfg.au_to_cm
TRIALS_PER_EPOCH = 10               # project_cfg.trials_per_epoch
NAIVE_SPLIT = 3                     # CorridorVsDarkActivity.m
LP_Z, LP_WIN, LP_MIN = -2.0, 10, 7  # project_cfg lp_*
EPOCHS = ["Trials 1-3", "Trials 4-10", "Intermediate", "Expert"]
AREAS = ["DMS", "DLS", "ACC", "V1", "CA1"]
AREA_FIELD = {"DMS": "is_dms", "DLS": "is_dls", "ACC": "is_acc", "V1": "is_v1", "CA1": "is_ca1"}
AREA_COL = {"DMS": "#1f77b4", "DLS": "#7fbf3f", "ACC": "#ff7f0e", "V1": "#9467bd", "CA1": "#d62728"}
MOVING_CM_S = 2.0                   # threshold for "moving" after 100 ms smoothing
GROUPS = [("Task", ROOT / "processed_data/preprocessed_data5cm.mat"),
          ("Control 1", ROOT / "processed_data/preprocessed_data_control5cm.mat")]


# ----------------------------------------------------------------------------- helpers
def find_learning_point(zerr: np.ndarray) -> float:
    """Port of find_learning_points.m for one animal. NaN = non-learner."""
    zerr = np.asarray(zerr, float).ravel()
    if zerr.size < LP_WIN:
        return np.nan
    passes = (zerr <= LP_Z)                     # NaN compares False, as in MATLAB
    # movsum(passes, [0, LP_WIN-1]): sum over [t, t+LP_WIN-1], truncated at the end
    win = np.array([passes[t:t + LP_WIN].sum() for t in range(passes.size)])
    hits = np.flatnonzero(passes & (win >= LP_MIN))
    return float(hits[0] + 1) if hits.size else np.nan   # 1-based like MATLAB


def epoch_indices(lp: float, n_trials: int) -> list[np.ndarray]:
    """Port of epoch_indices.m with naive_split = 3. 0-based trial indices."""
    w, s = TRIALS_PER_EPOCH, NAIVE_SPLIT
    idx = [np.array([], int)] * 4
    if n_trials >= s:
        idx[0] = np.arange(0, s)
    if n_trials >= w:
        idx[1] = np.arange(s, w)
    if np.isnan(lp) or lp > n_trials:
        return idx
    lp = int(lp)
    pre = (lp - w, lp - 1)                      # 1-based inclusive
    if pre[0] >= 1 and pre[1] <= n_trials:
        idx[2] = np.arange(pre[0] - 1, pre[1])
    post = (lp, lp + w - 1)
    if post[1] <= n_trials:
        idx[3] = np.arange(post[0] - 1, post[1])
    return idx


def cell_vec(h, ref) -> np.ndarray:
    return np.asarray(h[ref]).ravel().astype(float)


def trial_speed(pos: np.ndarray, t_ms: np.ndarray) -> tuple[float, float]:
    """Mean speed (cm/s) over the segment and fraction of time moving."""
    if pos.size < 3 or t_ms[-1] <= t_ms[0]:
        return np.nan, np.nan
    dist_cm = np.abs(np.diff(pos)).sum() * AU_TO_CM
    dur_s = (t_ms[-1] - t_ms[0]) / 1000.0
    # instantaneous speed on a 100 ms grid for the moving fraction
    grid = np.arange(t_ms[0], t_ms[-1], 100.0)
    if grid.size < 2:
        return dist_cm / dur_s, np.nan
    p = np.interp(grid, t_ms, pos)
    inst = np.abs(np.diff(p)) * AU_TO_CM / 0.1
    return dist_cm / dur_s, float(np.mean(inst > MOVING_CM_S))


def nanmean(x):
    x = np.asarray(x, float)
    return np.nan if np.all(np.isnan(x)) else np.nanmean(x)


def nansem(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.nan if x.size < 2 else x.std(ddof=1) / np.sqrt(x.size)


def zscore_rows(a: np.ndarray) -> np.ndarray:
    """Per-row z-score over all finite entries (unit x samples)."""
    mu = np.nanmean(a, axis=1, keepdims=True)
    sd = np.nanstd(a, axis=1, ddof=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = np.nan
    return (a - mu) / sd


def save_pair(fig, stem: str, max_px: int = 1600) -> None:
    w_in, h_in = fig.get_size_inches()
    dpi = int(min(150, max_px / max(w_in, h_in)))
    fig.savefig(FIG / f"{stem}.svg")
    fig.savefig(FIG / f"{stem}.png", dpi=dpi)
    print(f"  saved figures/{stem}.svg + .png ({int(w_in * dpi)}x{int(h_in * dpi)} px)")


# ----------------------------------------------------------------------------- main
def main() -> None:
    FIG.mkdir(exist_ok=True)
    vel_rows: list[dict] = []
    z_rows: list[dict] = []
    avg_task_lp = np.nan

    for gname, path in GROUPS:
        print(f"--- {gname}: {path.name}")
        with h5py.File(path, "r") as h:
            pd = h["preprocessed_data"]
            n_animals = pd["darkData"].shape[0]

            if gname == "Task":
                lps = np.array([find_learning_point(cell_vec(h, pd["zscored_lick_errors"][i, 0]))
                                for i in range(n_animals)])
                avg_task_lp = float(np.round(np.nanmean(lps)))
                print(f"  {np.isfinite(lps).sum()}/{n_animals} task learners, mean LP {avg_task_lp:.1f}"
                      "  (MATLAB 2026-08-27: 14/16, 41.0)")
            else:
                lps = np.full(n_animals, avg_task_lp)

            for i in range(n_animals):
                dd = h[pd["darkData"][i, 0]]
                cd = h[pd["corridorData"][i, 0]]
                n_tr = dd["trial_position"].shape[0]
                sp_d, sp_c, mv_d, mv_c = (np.full(n_tr, np.nan) for _ in range(4))
                for t in range(n_tr):
                    sp_d[t], mv_d[t] = trial_speed(cell_vec(h, dd["trial_position"][t, 0]),
                                                   cell_vec(h, dd["trial_times"][t, 0]))
                    sp_c[t], mv_c[t] = trial_speed(cell_vec(h, cd["trial_position"][t, 0]),
                                                   cell_vec(h, cd["trial_times"][t, 0]))

                # rates: h5 stores MATLAB (units x bins x trials) as (trials x bins x units)
                corr = np.asarray(h[pd["spatial_binned_fr_all"][i, 0]]).transpose(2, 1, 0)
                dark = np.asarray(h[pd["temp_binned_dark_fr"][i, 0]]).transpose(2, 1, 0)
                n_use = min(corr.shape[2], dark.shape[2], n_tr)
                corr, dark = corr[:, :, :n_use], dark[:, :, :n_use]
                pt_c = np.nanmean(corr, axis=1)          # units x trials
                pt_d = np.nanmean(dark, axis=1)
                # (i) common: per unit over both states' trial means (the figure's rule)
                zc_com = zscore_rows(np.concatenate([pt_c, pt_d], axis=1))
                z_c_common, z_d_common = zc_com[:, :n_use], zc_com[:, n_use:]
                # (ii) per-state on the same trial means
                z_c_sep, z_d_sep = zscore_rows(pt_c), zscore_rows(pt_d)
                # (iii) ST-style: normalise over ALL bins x trials of each state separately,
                #       then take the per-trial mean (what SpatioTemporal does for corridor)
                z_c_st = np.nanmean(zscore_rows(corr.reshape(corr.shape[0], -1)).reshape(corr.shape), axis=1)
                z_d_st = np.nanmean(zscore_rows(dark.reshape(dark.shape[0], -1)).reshape(dark.shape), axis=1)

                masks = {a: np.asarray(h[pd[AREA_FIELD[a]][i, 0]]).ravel().astype(bool)
                         for a in AREAS if AREA_FIELD[a] in pd}
                idx = epoch_indices(lps[i], n_use)
                for e, tr in enumerate(idx):
                    if tr.size == 0:
                        continue
                    vel_rows.append(dict(group=gname, animal=i + 1, epoch=EPOCHS[e], n_trials=tr.size,
                                         dark_speed_cm_s=nanmean(sp_d[tr]), corridor_speed_cm_s=nanmean(sp_c[tr]),
                                         dark_moving_frac=nanmean(mv_d[tr]), corridor_moving_frac=nanmean(mv_c[tr])))
                    for a, m in masks.items():
                        if m.sum() == 0:
                            continue
                        raw_diff = np.nanmean(pt_c[m][:, tr], axis=1) - np.nanmean(pt_d[m][:, tr], axis=1)
                        com_diff = np.nanmean(z_c_common[m][:, tr], axis=1) - np.nanmean(z_d_common[m][:, tr], axis=1)
                        sep_diff = np.nanmean(z_c_sep[m][:, tr], axis=1) - np.nanmean(z_d_sep[m][:, tr], axis=1)
                        st_diff = np.nanmean(z_c_st[m][:, tr], axis=1) - np.nanmean(z_d_st[m][:, tr], axis=1)
                        ok = np.isfinite(raw_diff) & np.isfinite(com_diff) & (raw_diff != 0)
                        z_rows.append(dict(group=gname, animal=i + 1, area=a, epoch=EPOCHS[e], n_units=int(m.sum()),
                                           raw_diff_hz=nanmean(raw_diff), common_z_diff=nanmean(com_diff),
                                           perstate_z_diff=nanmean(sep_diff), ststyle_z_diff=nanmean(st_diff),
                                           frac_sign_agree_raw_vs_common=float(np.mean(np.sign(raw_diff[ok]) == np.sign(com_diff[ok]))) if ok.any() else np.nan))
                print(f"  animal {i + 1:2d}: lp={lps[i]:>4}, {n_use} trials, "
                      f"dark {nanmean(sp_d):5.1f} vs corridor {nanmean(sp_c):5.1f} cm/s, "
                      f"dark moving {100 * nanmean(mv_d):4.0f}%")

    # ------------------------------------------------------------------ tables
    for name, rows in [("corridor_vs_dark_velocity_by_animal.csv", vel_rows),
                       ("corridor_vs_dark_zscore_variants.csv", z_rows)]:
        with open(FIG / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote figures/{name} ({len(rows)} rows)")

    # ------------------------------------------------------------------ velocity figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for gi, (gname, _) in enumerate(GROUPS):
        ls = "-" if gi == 0 else "--"
        for state, col in [("dark", "0.35"), ("corridor", "#1f77b4")]:
            mu = [nanmean([r[f"{state}_speed_cm_s"] for r in vel_rows if r["group"] == gname and r["epoch"] == e]) for e in EPOCHS]
            se = [nansem([r[f"{state}_speed_cm_s"] for r in vel_rows if r["group"] == gname and r["epoch"] == e]) for e in EPOCHS]
            axes[0].errorbar(range(4), mu, se, fmt="o" + ls, color=col, capsize=3, lw=1.8,
                             label=f"{gname}, {state}")
            mu = [nanmean([r[f"{state}_moving_frac"] for r in vel_rows if r["group"] == gname and r["epoch"] == e]) for e in EPOCHS]
            se = [nansem([r[f"{state}_moving_frac"] for r in vel_rows if r["group"] == gname and r["epoch"] == e]) for e in EPOCHS]
            axes[1].errorbar(range(4), mu, se, fmt="o" + ls, color=col, capsize=3, lw=1.8,
                             label=f"{gname}, {state}")
        xs = [r["corridor_speed_cm_s"] for r in vel_rows if r["group"] == gname]
        ys = [r["dark_speed_cm_s"] for r in vel_rows if r["group"] == gname]
        axes[2].scatter(xs, ys, s=22, alpha=0.75, label=gname, marker="o" if gi == 0 else "s")
    n_task = len({r["animal"] for r in vel_rows if r["group"] == "Task"})
    n_ctrl = len({r["animal"] for r in vel_rows if r["group"] == "Control 1"})
    for ax in axes[:2]:
        ax.set_xticks(range(4), EPOCHS, rotation=25)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Mean speed (cm/s)")
    axes[0].set_title(f"Running speed by epoch (animal mean ± SEM; N = {n_task} task, {n_ctrl} control)", fontsize=10)
    axes[1].set_ylabel(f"Fraction of time moving (> {MOVING_CM_S:g} cm/s)")
    axes[1].set_title("Time spent moving (100 ms grid)", fontsize=10)
    axes[1].set_ylim(0, 1.02)
    lim = max(np.nanmax(xs) if xs else 0, 60)
    axes[2].plot([0, lim], [0, lim], "k:", lw=1, label="identity")
    axes[2].set_xlabel("Corridor speed (cm/s)")
    axes[2].set_ylabel("Dark-period speed (cm/s)")
    axes[2].set_title("Animal × epoch means", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle("Fig S1 check: the animal keeps running in the 5 s dark period (VR position integrates the wheel in the dark)", fontsize=11)
    fig.tight_layout()
    save_pair(fig, "CorridorVsDark_velocity")

    # ------------------------------------------------------------------ z-score variants figure
    variants = [("raw_diff_hz", "Raw FR difference (Hz)"),
                ("common_z_diff", "Common z (figure's rule)"),
                ("perstate_z_diff", "Per-state z (trial means)"),
                ("ststyle_z_diff", "State-wise z over bins × trials (ST-style)")]
    for gname, _ in GROUPS:
        fig, axes = plt.subplots(len(variants), len(AREAS), figsize=(3.1 * len(AREAS), 2.6 * len(variants)),
                                 sharex=True)
        for vi, (key, vlabel) in enumerate(variants):
            for ai, a in enumerate(AREAS):
                ax = axes[vi, ai]
                vals = [[r[key] for r in z_rows if r["group"] == gname and r["area"] == a and r["epoch"] == e] for e in EPOCHS]
                n_an = max(len(v) for v in vals)
                if n_an == 0:
                    ax.axis("off")
                    continue
                mu = [nanmean(v) for v in vals]
                se = [nansem(v) for v in vals]
                ax.axhline(0, color="0.6", lw=0.8)
                ax.errorbar(range(4), mu, se, fmt="o-", color=AREA_COL[a], capsize=3, lw=1.8,
                            label="corridor − dark")
                if vi == 0:
                    ax.set_title(f"{a} (N = {n_an} mice)", fontsize=10)
                if ai == 0:
                    ax.set_ylabel(vlabel, fontsize=8)
                if vi == len(variants) - 1:
                    ax.set_xticks(range(4), EPOCHS, rotation=25, fontsize=8)
                if vi == 0 and ai == 0:
                    ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f"{gname}: corridor − dark contrast under four normalisations (animal mean ± SEM)", fontsize=11)
        fig.tight_layout()
        stem = "CorridorVsDark_zscore_variants" + ("" if gname == "Task" else "_control1")
        save_pair(fig, stem)

    # ------------------------------------------------------------------ console summary
    print("\nSummary (animal-level means over epochs):")
    for gname, _ in GROUPS:
        rs = [r for r in vel_rows if r["group"] == gname]
        print(f"  {gname}: dark {nanmean([r['dark_speed_cm_s'] for r in rs]):.1f} cm/s, "
              f"corridor {nanmean([r['corridor_speed_cm_s'] for r in rs]):.1f} cm/s, "
              f"dark moving {100 * nanmean([r['dark_moving_frac'] for r in rs]):.0f}%, "
              f"corridor moving {100 * nanmean([r['corridor_moving_frac'] for r in rs]):.0f}%")
        zs = [r for r in z_rows if r["group"] == gname]
        print(f"  {gname}: sign agreement raw vs common-z per unit = "
              f"{100 * nanmean([r['frac_sign_agree_raw_vs_common'] for r in zs]):.1f}%; "
              f"|per-state z diff| max = {np.nanmax(np.abs([r['perstate_z_diff'] for r in zs])):.2e}; "
              f"|ST-style z diff| mean = {nanmean(np.abs([r['ststyle_z_diff'] for r in zs])):.3f}")


if __name__ == "__main__":
    main()
