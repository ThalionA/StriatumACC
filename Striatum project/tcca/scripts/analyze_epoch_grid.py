"""Analyse the 2026-08-11 epoch grid: bin {25,10 ms} x FS {excl,incl} x {partial,plain}.

Consumes the epoch_metrics_* and epoch_lagcurves_* CSVs written by run_epochs.py
(tags _b25 / _b10) and answers, per config:

  A. STRENGTH   -- per-pair, per-animal cc1 medians by epoch; paired Wilcoxon
                   naive->expert (animals-as-n), BH across the striatal triangle.
  B. IFI x WINDOW -- IFI recomputed from the exported CC1 lag curves at every
                   integration half-width w (|lag| <= w bins), per animal:
                   (i) existence: session-pooled (epoch-mean) IFI vs 0 per window,
                   (ii) change: naive->expert IFI delta per window,
                   both Wilcoxon, BH across windows within pair.
  C. FS CONTRAST -- matched (animal, pair, epoch) cells: cc1 FS-incl vs FS-excl.
  D. PLAIN-PARTIAL -- matched cells: cc1 plain vs partial (shared-drive uplift).

Outputs: results/grid_summary.csv, results/grid_ifi_windows.csv, and a printed
summary. Registered priors: PREDICTIONS.md 2026-08-11 (P1-P5).

Run:  python3 scripts/analyze_epoch_grid.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from striatum_tcca import config, lagged, paired_stats  # noqa: E402

RESULTS = config.RESULTS_DIR
TRIANGLE = ["DMS-DLS", "DMS-ACC", "DLS-ACC"]
EPOCHS = ["naive", "intermediate", "expert"]
BINS = {"_b25": 25, "_b10": 10}


def _suffix(fs: bool, plain: bool, tag: str) -> str:
    return ("_fsincl" if fs else "") + ("_plain" if plain else "") + tag


def _rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _configs():
    for tag, bin_ms in BINS.items():
        for fs in (False, True):
            for plain in (False, True):
                yield {"tag": tag, "bin_ms": bin_ms, "fs": fs, "plain": plain,
                       "suffix": _suffix(fs, plain, tag),
                       "name": (f"b{bin_ms}"
                                + ("/fsincl" if fs else "/fsexcl")
                                + ("/plain" if plain else "/partial"))}


def load_metrics(suffix: str) -> dict[tuple, dict]:
    """{(animal, pair, epoch): row} for one config."""
    out = {}
    for r in _rows(RESULTS / f"epoch_metrics{suffix}.csv"):
        out[(r["animal"], r["pair"], r["epoch"])] = r
    return out


def load_curves(suffix: str) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """{(animal, pair, epoch): (lags, cc1_curve)} for one config."""
    acc = defaultdict(list)
    for r in _rows(RESULTS / f"epoch_lagcurves{suffix}.csv"):
        v = float(r["cc1"]) if r["cc1"] != "" else np.nan
        acc[(r["animal"], r["pair"], r["epoch"])].append((int(r["lag_bin"]), v))
    out = {}
    for k, pts in acc.items():
        pts.sort()
        out[k] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def per_animal_cc1(metrics: dict, pair: str) -> dict[str, dict[str, float]]:
    """{animal: {epoch: cc1}} for one pair."""
    out = defaultdict(dict)
    for (animal, p, epoch), r in metrics.items():
        if p == pair:
            out[animal][epoch] = float(r["cc1"])
    return out


def strength_tests(metrics: dict) -> list[dict]:
    rows = []
    pvals = []
    for pair in TRIANGLE:
        pa = per_animal_cc1(metrics, pair)
        med = {e: float(np.median([v[e] for v in pa.values() if e in v]))
               if any(e in v for v in pa.values()) else np.nan for e in EPOCHS}
        deltas = [v["expert"] - v["naive"] for v in pa.values()
                  if "expert" in v and "naive" in v]
        _, _, _, p = paired_stats.wilcoxon_signed(np.array(deltas))
        rows.append({"pair": pair, "n_animals": len(deltas),
                     "cc1_med_naive": round(med["naive"], 4),
                     "cc1_med_inter": round(med["intermediate"], 4),
                     "cc1_med_expert": round(med["expert"], 4),
                     "delta_median": round(float(np.median(deltas)), 4) if deltas else np.nan,
                     "p_naive_vs_expert": round(p, 4) if np.isfinite(p) else np.nan})
        pvals.append(p)
    sig = paired_stats.fdr_bh(np.array([p if np.isfinite(p) else 1.0 for p in pvals]))
    for r, s in zip(rows, sig):
        r["bh_sig"] = int(bool(s))
    return rows


def ifi_window_tests(curves: dict, bin_ms: int) -> list[dict]:
    """Per pair x window: existence (pooled) and change (naive->expert) tests."""
    rows = []
    for pair in TRIANGLE:
        # per animal: epoch -> per-window IFI vector
        per_animal = defaultdict(dict)
        max_w = 0
        for (animal, p, epoch), (lags, curve) in curves.items():
            if p != pair:
                continue
            wv = lagged.ifi_by_window(lags, curve)
            per_animal[animal][epoch] = wv
            max_w = max(max_w, wv.size)
        if not per_animal or max_w == 0:
            continue
        p_exist, p_change = [], []
        med_exist, med_change, n_e, n_c = [], [], [], []
        for w in range(max_w):
            pooled = [np.nanmean([v[e][w] for e in v if w < v[e].size])
                      for v in per_animal.values()
                      if any(w < v[e].size for e in v)]
            pooled = [x for x in pooled if np.isfinite(x)]
            deltas = [v["expert"][w] - v["naive"][w] for v in per_animal.values()
                      if "expert" in v and "naive" in v
                      and w < v["expert"].size and w < v["naive"].size]
            deltas = [x for x in deltas if np.isfinite(x)]
            pe = paired_stats.wilcoxon_signed(np.array(pooled))[3] if len(pooled) >= 5 else np.nan
            pc = paired_stats.wilcoxon_signed(np.array(deltas))[3] if len(deltas) >= 5 else np.nan
            p_exist.append(pe); p_change.append(pc)
            med_exist.append(np.median(pooled) if pooled else np.nan)
            med_change.append(np.median(deltas) if deltas else np.nan)
            n_e.append(len(pooled)); n_c.append(len(deltas))
        sig_e = paired_stats.fdr_bh(np.array([p if np.isfinite(p) else 1.0 for p in p_exist]))
        sig_c = paired_stats.fdr_bh(np.array([p if np.isfinite(p) else 1.0 for p in p_change]))
        for w in range(max_w):
            rows.append({"pair": pair, "window_bins": w + 1,
                         "window_ms": (w + 1) * bin_ms,
                         "ifi_pooled_median": round(float(med_exist[w]), 4),
                         "n_exist": n_e[w],
                         "p_exist": round(float(p_exist[w]), 4) if np.isfinite(p_exist[w]) else np.nan,
                         "bh_exist": int(bool(sig_e[w])),
                         "ifi_delta_median": round(float(med_change[w]), 4),
                         "n_change": n_c[w],
                         "p_change": round(float(p_change[w]), 4) if np.isfinite(p_change[w]) else np.nan,
                         "bh_change": int(bool(sig_c[w]))})
    return rows


def matched_contrast(m_a: dict, m_b: dict, label_a: str, label_b: str) -> dict:
    """cc1 contrast over matched (animal, pair, epoch) cells (triangle only).

    frac_positive/median_diff are cell-level (the registered P2/P3 readout);
    the p-value is animal-level (Wilcoxon over per-animal median differences)
    so the inferential unit stays animals-as-n.
    """
    keys = [k for k in m_a if k in m_b and k[1] in TRIANGLE]
    d = np.array([float(m_b[k]["cc1"]) - float(m_a[k]["cc1"]) for k in keys])
    if d.size == 0:
        return {"contrast": f"{label_b}-{label_a}", "n_cells": 0}
    by_animal = defaultdict(list)
    for k, v in zip(keys, d):
        by_animal[k[0]].append(v)
    animal_meds = np.array([np.median(v) for v in by_animal.values()])
    n_an, med_an, _, p = paired_stats.wilcoxon_signed(animal_meds)
    return {"contrast": f"{label_b}-{label_a}", "n_cells": int(d.size),
            "frac_positive": round(float(np.mean(d > 0)), 3),
            "median_diff": round(float(np.median(d)), 4),
            "n_animals": int(n_an),
            "p_animal": round(float(p), 6) if np.isfinite(p) else np.nan}


def main():
    summary, ifi_rows = [], []
    loaded = {}
    for c in _configs():
        m = load_metrics(c["suffix"])
        if not m:
            print(f"[missing] {c['name']} (epoch_metrics{c['suffix']}.csv)")
            continue
        loaded[(c["tag"], c["fs"], c["plain"])] = m
        print(f"\n=== {c['name']} ({len(m)} cells) ===")
        for r in strength_tests(m):
            r2 = {"config": c["name"], "test": "strength", **r}
            summary.append(r2)
            print(f"  {r['pair']:8s} cc1 med N/I/E {r['cc1_med_naive']:.3f}/"
                  f"{r['cc1_med_inter']:.3f}/{r['cc1_med_expert']:.3f}  "
                  f"dNE {r['delta_median']:+.3f}  p={r['p_naive_vs_expert']}"
                  f"{' *BH' if r['bh_sig'] else ''} (n={r['n_animals']})")
        curves = load_curves(c["suffix"])
        for r in ifi_window_tests(curves, c["bin_ms"]):
            ifi_rows.append({"config": c["name"], **r})

    # C. FS contrast and D. plain-vs-partial, within matched frames
    print("\n=== matched-cell contrasts (striatal triangle) ===")
    for tag in BINS:
        for plain in (False, True):
            a, b = loaded.get((tag, False, plain)), loaded.get((tag, True, plain))
            if a and b:
                r = matched_contrast(a, b, "fsexcl", "fsincl")
                r["config"] = f"b{BINS[tag]}" + ("/plain" if plain else "/partial")
                r["test"] = "fs_incl_minus_excl"
                summary.append(r)
                print(f"  {r['config']:12s} FSincl-FSexcl: {r.get('frac_positive', 'NA')} pos, "
                      f"median {r.get('median_diff', 'NA')}, p_animal={r.get('p_animal', 'NA')} "
                      f"(cells={r['n_cells']}, n={r.get('n_animals', 0)})")
        for fs in (False, True):
            a, b = loaded.get((tag, fs, False)), loaded.get((tag, fs, True))
            if a and b:
                r = matched_contrast(a, b, "partial", "plain")
                r["config"] = f"b{BINS[tag]}" + ("/fsincl" if fs else "/fsexcl")
                r["test"] = "plain_minus_partial"
                summary.append(r)
                print(f"  {r['config']:12s} plain-partial: {r.get('frac_positive', 'NA')} pos, "
                      f"median {r.get('median_diff', 'NA')}, p_animal={r.get('p_animal', 'NA')} "
                      f"(cells={r['n_cells']}, n={r.get('n_animals', 0)})")
    for fs in (False, True):
        for plain in (False, True):
            a, b = loaded.get(("_b25", fs, plain)), loaded.get(("_b10", fs, plain))
            if a and b:
                r = matched_contrast(a, b, "b25", "b10")
                r["config"] = (("fsincl" if fs else "fsexcl")
                               + ("/plain" if plain else "/partial"))
                r["test"] = "b10_minus_b25"
                summary.append(r)
                print(f"  {r['config']:12s} b10-b25:       {r.get('frac_positive', 'NA')} pos, "
                      f"median {r.get('median_diff', 'NA')}, p_animal={r.get('p_animal', 'NA')} "
                      f"(cells={r['n_cells']}, n={r.get('n_animals', 0)})")

    # persist
    def _dump(path, rows):
        keys = sorted({k for r in rows for k in r}, key=lambda s: s)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    _dump(RESULTS / "grid_summary.csv", summary)
    _dump(RESULTS / "grid_ifi_windows.csv", ifi_rows)

    # headline IFI-window findings
    hits = [r for r in ifi_rows if r["bh_exist"] or r["bh_change"]]
    print(f"\nIFI x window: {len(hits)} BH-surviving (pair x window x config) cells "
          f"of {len(ifi_rows)}")
    for r in hits[:20]:
        kind = "exist" if r["bh_exist"] else "change"
        print(f"  {r['config']:14s} {r['pair']:8s} w=±{r['window_ms']}ms "
              f"[{kind}] med {r['ifi_pooled_median' if kind == 'exist' else 'ifi_delta_median']:+.3f} "
              f"p={r['p_exist' if kind == 'exist' else 'p_change']}")
    print(f"\nwrote grid_summary.csv ({len(summary)}), grid_ifi_windows.csv ({len(ifi_rows)})")


if __name__ == "__main__":
    main()
