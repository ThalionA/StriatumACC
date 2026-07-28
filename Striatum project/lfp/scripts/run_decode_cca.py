"""Position decoding + cross-area CCA on LFP band power (PROVISIONAL, gated).

Crosses the README provenance gate deliberately, with two framings:
  * position decoding is run as an ALIGNMENT TEST (decodes above chance only if
    the export offset is roughly right);
  * cross-area CCA is offset-robust but volume-conduction limited, so it is
    reported against a within-area split-half VC ceiling and a trial-shuffle null.
All scale-relative and provisional.

Run:  cd "Striatum project/lfp" && python3 scripts/run_decode_cca.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tcca" / "src"))

from striatum_lfp import config, decode, features, geometry, learning  # noqa: E402
from striatum_lfp.reader import DATASET  # noqa: E402

FS = config.FS
BANDS = {"theta": (4, 8), "beta": (15, 30), "low_gamma": (30, 80)}
AREAS = ["DMS", "DLS", "ACC"]
AREA_COL = {"DMS": "#4C72B0", "DLS": "#55A868", "ACC": "#DD8452"}
N_POS = 20                      # corridor position bins (12.5 cm)
PAIRS = [("DMS", "ACC"), ("DMS", "DLS"), ("DLS", "ACC")]
CCA_BANDS = ["theta", "beta"]
SOS = {bd: features.design_band_sos(band, fs=FS) for bd, band in BANDS.items()}


def area_channels(mouse_id):
    depths = geometry.channel_depths()
    masks = geometry.channel_area_masks(depths, geometry.load_area_boundaries(mouse_id, "striatum"))
    return {ar: np.where(masks[ar])[0] for ar in AREAS if ar in masks}


def collect(mouse_id):
    """Per (clean traversal x position bin): area-median band power (decoding
    features) and per-channel band power (CCA), for every band."""
    with h5py.File(config.RAW_MAT[mouse_id], "r") as f:
        VR = np.asarray(f["VR_data"][()])
        vt = np.asarray(f["VR_times_synched"][()]).ravel()
    pos_au, world, trial = VR[:, 1], VR[:, 4], VR[:, 6]
    frame_ms = np.round(vt * 1000).astype(int)
    trav = learning.corridor_traversals(world, trial, frame_ms, min_frames=20)
    achan = area_channels(mouse_id)
    handle = h5py.File(config.LFP_DIR / config.FILE_BY_MOUSE[mouse_id], "r")
    lf = handle[DATASET]
    n = lf.shape[0]

    per_trav = []   # dict(trial, peak, {band: [N_POS x 384]})
    for tv in trav:
        a, b = max(0, tv.start_ms), min(n, tv.end_ms)
        if b - a < 500:
            continue
        seg = np.asarray(lf[a:b, :], dtype=float)
        samp_ms = np.arange(a, b)
        pos_cm = np.interp(samp_ms, frame_ms, pos_au) * config.AU_TO_CM
        binned = {}
        for bd in BANDS:
            env = features.band_envelope(seg, SOS[bd])
            binned[bd] = decode.bin_by_position(env, pos_cm, N_POS, config.CORRIDOR_CM)
        per_trav.append(dict(trial=int(tv.trial), peak=float(np.max(np.abs(seg))), binned=binned))
    handle.close()

    peaks = np.array([p["peak"] for p in per_trav])
    clean = ~learning.robust_log_outliers(peaks, z_threshold=4.0)
    kept = [p for p, c in zip(per_trav, clean) if c]

    # stack rows = (traversal x position bin)
    pos_centres = (np.arange(N_POS) + 0.5) / N_POS * config.CORRIDOR_CM
    trial_row, posbin_row, pos_row = [], [], []
    feat_rows = []                                   # decoding features (area x band)
    band_chan = {bd: [] for bd in BANDS}             # per band: rows x 384
    for p in kept:
        for b in range(N_POS):
            row = [np.nanmedian(p["binned"][bd][b, achan[ar]]) for ar in achan for bd in BANDS]
            feat_rows.append(row)
            trial_row.append(p["trial"]); posbin_row.append(b); pos_row.append(pos_centres[b])
            for bd in BANDS:
                band_chan[bd].append(p["binned"][bd][b])
    return dict(
        mouse=mouse_id, achan=achan, feat_names=[f"{ar}_{bd}" for ar in achan for bd in BANDS],
        X=np.array(feat_rows), y=np.array(pos_row), groups=np.array(trial_row),
        posbin=np.array(posbin_row), band_chan={bd: np.array(v) for bd, v in band_chan.items()},
        n_clean=len(kept))


def decode_analysis(res, rng):
    """Position decoding: all areas + each area alone, real vs shuffled."""
    X, y, g, names = res["X"], res["y"], res["groups"], res["feat_names"]
    out = {}
    subsets = {"all": np.arange(len(names))}
    for ar in res["achan"]:
        subsets[ar] = np.array([i for i, nm in enumerate(names) if nm.startswith(ar + "_")])
    for key, cols in subsets.items():
        r2, mae, _ = decode.ridge_cv_decode(X[:, cols], y, g)
        sh = [decode.ridge_cv_decode(X[:, cols], rng.permutation(y), g)[0] for _ in range(15)]
        out[key] = dict(r2=r2, mae=mae, shuf_r2_mean=float(np.mean(sh)), shuf_r2_p95=float(np.percentile(sh, 95)))
    return out


def cca_analysis(res, rng):
    """Cross-area vs within-area (VC ceiling) vs trial-shuffle CCA per pair/band."""
    posbin = res["posbin"]
    out = {}
    for bd in CCA_BANDS:
        M = res["band_chan"][bd]                      # rows x 384
        Mr = decode.residualise_by_group(M, posbin)   # residual fluctuations
        for x, yb in PAIRS:
            if x not in res["achan"] or yb not in res["achan"]:
                continue
            cx, cy = res["achan"][x], res["achan"][yb]
            cross = decode.heldout_cca(Mr[:, cx], Mr[:, cy], k=5)
            half = len(cx) // 2
            within = decode.heldout_cca(Mr[:, cx[:half]], Mr[:, cx[half:]], k=5) if half >= 2 else np.nan
            sh_rows = rng.permutation(len(Mr))
            shuf = decode.heldout_cca(Mr[:, cx], Mr[sh_rows][:, cy], k=5)
            out[f"{x}-{yb}_{bd}"] = dict(cross=cross, within=within, shuffle=shuf)
    return out


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dec_rows, cca_rows, all_dec, all_cca = [], [], {}, {}
    for mid in config.LFP_MICE:
        print(f"[decode/cca] mouse {mid} ...", flush=True)
        res = collect(mid)
        dec = decode_analysis(res, rng)
        cca = cca_analysis(res, rng)
        all_dec[mid], all_cca[mid] = dec, cca
        print(f"    decode R2 all={dec['all']['r2']:.3f} (shuf p95 {dec['all']['shuf_r2_p95']:.3f}) "
              f"MAE={dec['all']['mae']:.1f} cm | n_clean={res['n_clean']}", flush=True)
        for key, d in dec.items():
            dec_rows.append(dict(mouse=mid, feature_set=key, r2=round(d["r2"], 4),
                                 mae_cm=round(d["mae"], 2), shuf_r2_p95=round(d["shuf_r2_p95"], 4),
                                 above_chance=bool(d["r2"] > d["shuf_r2_p95"])))
        for key, c in cca.items():
            cca_rows.append(dict(mouse=mid, pair_band=key,
                                 cross=round(c["cross"], 3) if np.isfinite(c["cross"]) else np.nan,
                                 within_area_VC=round(c["within"], 3) if np.isfinite(c["within"]) else np.nan,
                                 shuffle=round(c["shuffle"], 3) if np.isfinite(c["shuffle"]) else np.nan))
    with (config.RESULTS_DIR / "decode_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dec_rows[0])); w.writeheader(); w.writerows(dec_rows)
    with (config.RESULTS_DIR / "cca_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cca_rows[0])); w.writeheader(); w.writerows(cca_rows)
    plot_decode(all_dec)
    plot_cca(all_cca)
    print("[done] results/decode_summary.csv, cca_summary.csv + figures/decode_*.png, cca_*.png")


def plot_decode(all_dec):
    mice = list(all_dec)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    keys = ["all", "DMS", "DLS", "ACC"]
    x = np.arange(len(mice)); w = 0.2
    for j, key in enumerate(keys):
        r2 = [all_dec[m].get(key, {}).get("r2", np.nan) for m in mice]
        axes[0].bar(x + j * w, r2, w, label=key)
    for i, m in enumerate(mice):
        p95 = all_dec[m]["all"]["shuf_r2_p95"]
        axes[0].plot([x[i] - w / 2, x[i] + 3.5 * w], [p95, p95], "k--", lw=0.8)
    axes[0].set_xticks(x + 1.5 * w); axes[0].set_xticklabels(mice)
    axes[0].set_ylabel("CV R² (position)"); axes[0].axhline(0, color="0.6", lw=0.6)
    axes[0].set_title("Position decoding from LFP band power\n(dashed = shuffle 95th pct)")
    axes[0].legend(fontsize=8, title="features")
    mae = [all_dec[m]["all"]["mae"] for m in mice]
    axes[1].bar(x, mae, color="0.5")
    axes[1].axhline(config.CORRIDOR_CM / np.sqrt(12), color="r", ls="--", lw=0.9, label="chance MAE (uniform)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(mice)
    axes[1].set_ylabel("CV MAE (cm)"); axes[1].set_title("Decoding error (all areas)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Position decoding = alignment test (PROVISIONAL)", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"decode_position.{ext}", dpi=110)
    plt.close(fig)


def plot_cca(all_cca):
    mice = list(all_cca)
    fig, axes = plt.subplots(len(CCA_BANDS), len(mice), figsize=(15, 7), constrained_layout=True,
                             squeeze=False)
    for bi, bd in enumerate(CCA_BANDS):
        for mi, m in enumerate(mice):
            ax = axes[bi][mi]
            pairs = [f"{x}-{y}" for x, y in PAIRS if f"{x}-{y}_{bd}" in all_cca[m]]
            xs = np.arange(len(pairs)); w = 0.25
            for j, kind in enumerate(["cross", "within", "shuffle"]):
                vals = [all_cca[m][f"{p}_{bd}"][kind] for p in pairs]
                ax.bar(xs + j * w, vals, w, label=kind,
                       color={"cross": "#C44E52", "within": "#4C72B0", "shuffle": "0.6"}[kind])
            ax.set_xticks(xs + w); ax.set_xticklabels(pairs, fontsize=7, rotation=15)
            ax.set_ylim(0, 1)
            if mi == 0:
                ax.set_ylabel(f"{bd}\nheld-out CC1")
            if bi == 0:
                ax.set_title(f"mouse {m}", fontsize=10)
            if bi == 0 and mi == len(mice) - 1:
                ax.legend(fontsize=7)
    fig.suptitle("Cross-area LFP CCA vs within-area VC ceiling vs trial-shuffle (PROVISIONAL; "
                 "volume-conduction limited)", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"cca_cross_area.{ext}", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
