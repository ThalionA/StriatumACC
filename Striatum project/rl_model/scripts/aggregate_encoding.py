"""Aggregate the per-neuron RL-latent encoding results into citable tables.

Reads results/encoding_v6/M*.npz (one per mouse, written by
run_neural_encoding.py) and writes:

  results/encoding_summary_by_animal.csv
      one row per (mouse, area, model, latent): the mouse's mean unique dR2
      and the fraction of its neurons significant against the trial-shuffle
      null. Animals-as-n, so downstream tests use the honest unit.

  results/encoding_summary_by_area.csv
      one row per (area, model, latent): median across animals of the above,
      plus a one-sample Wilcoxon of the per-animal significant fraction
      against the alpha=0.05 chance rate, BH-corrected within model x area.

The dR2 readout is the unique cross-validated contribution of each latent
after behaviour (licks, velocity) and a slow-drift basis are controlled;
'beh_spatial' additionally removes per-bin mean firing, so a latent is
credited only for trial-by-trial modulation beyond stationary spatial tuning.

Run:  ./.venv/bin/python -m scripts.aggregate_encoding
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCDIR = os.path.join(HERE, "results", "encoding_v6")
OUTDIR = os.path.join(HERE, "results")
LATENTS = ("value", "rpe", "precision")
MODELS = ("beh", "beh_spatial")
ALPHA = 0.05


def bh(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if not ok.any():
        return out
    pf = p[ok]
    n = pf.size
    order = np.argsort(pf)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, pf[idx] * n / k)
        adj[idx] = prev
    out[ok] = adj
    return out


def main() -> None:
    files = sorted(f for f in os.listdir(ENCDIR) if f.endswith(".npz"))
    if not files:
        raise SystemExit(f"no per-mouse files in {ENCDIR}")

    per_animal = []
    for fn in files:
        mid = fn.split(".")[0]
        d = np.load(os.path.join(ENCDIR, fn), allow_pickle=True)
        areas = d["area"].astype(str)
        for area in sorted(set(areas)):
            sel = areas == area
            n_cells = int(sel.sum())
            if n_cells == 0:
                continue
            for mdl in MODELS:
                for lat in LATENTS:
                    dr2 = np.asarray(d[f"dR2_{mdl}_{lat}"], float)[sel]
                    pv = np.asarray(d[f"pval_{mdl}_{lat}"], float)[sel]
                    finite = np.isfinite(dr2)
                    if not finite.any():
                        continue
                    per_animal.append({
                        "mouse": mid, "area": area, "model": mdl, "latent": lat,
                        "n_cells": n_cells,
                        "mean_dR2": round(float(np.nanmean(dr2)), 6),
                        "median_dR2": round(float(np.nanmedian(dr2)), 6),
                        "frac_sig": round(float(np.nanmean(pv < ALPHA)), 4),
                    })

    with open(os.path.join(OUTDIR, "encoding_summary_by_animal.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_animal[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(per_animal)

    # --- per area x model x latent, animals as n --------------------------
    grouped = defaultdict(list)
    for r in per_animal:
        grouped[(r["area"], r["model"], r["latent"])].append(r)

    rows = []
    for (area, mdl, lat), rs in sorted(grouped.items()):
        fr = np.array([r["frac_sig"] for r in rs], float)
        dr = np.array([r["mean_dR2"] for r in rs], float)
        n = fr.size
        # Is the per-animal significant fraction above the alpha chance rate?
        if n >= 5 and np.any(fr != ALPHA):
            p = stats.wilcoxon(fr - ALPHA).pvalue
        else:
            p = np.nan
        rows.append({
            "area": area, "model": mdl, "latent": lat, "n_animals": n,
            "total_cells": int(sum(r["n_cells"] for r in rs)),
            "median_mean_dR2": round(float(np.nanmedian(dr)), 6),
            "median_frac_sig": round(float(np.nanmedian(fr)), 4),
            "p_frac_vs_chance": round(float(p), 5) if np.isfinite(p) else "",
        })

    # BH within (model, area) family across the three latents
    fam = defaultdict(list)
    for i, r in enumerate(rows):
        fam[(r["model"], r["area"])].append(i)
    for idxs in fam.values():
        ps = [rows[i]["p_frac_vs_chance"] for i in idxs]
        ps = [p if p != "" else np.nan for p in ps]
        for i, q in zip(idxs, bh(ps)):
            rows[i]["p_BH"] = round(float(q), 5) if np.isfinite(q) else ""
            rows[i]["sig_BH"] = int(np.isfinite(q) and q < 0.05)

    with open(os.path.join(OUTDIR, "encoding_summary_by_area.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"{len(files)} mice · {len(per_animal)} animal-area-model-latent cells")
    print("\nPer area (animals as n), model = beh_spatial "
          "(trial-by-trial modulation beyond stationary tuning):")
    print(f"{'area':6s} {'latent':10s} {'n':>3s} {'cells':>6s} "
          f"{'med dR2':>9s} {'med frac_sig':>12s} {'p':>8s} {'BH':>8s}")
    for r in rows:
        if r["model"] != "beh_spatial":
            continue
        print(f"{r['area']:6s} {r['latent']:10s} {r['n_animals']:3d} "
              f"{r['total_cells']:6d} {r['median_mean_dR2']:9.5f} "
              f"{r['median_frac_sig']:12.3f} "
              f"{str(r['p_frac_vs_chance']):>8s} {str(r.get('p_BH','')):>8s}"
              f"{'  *' if r.get('sig_BH') else ''}")
    print("\nwrote encoding_summary_by_animal.csv, encoding_summary_by_area.csv")


if __name__ == "__main__":
    main()
