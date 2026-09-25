#!/usr/bin/env python3
"""**Never compare the raw ``mi`` column across epochs.**

The Tort modulation index is positively biased at small sample sizes, and this
driver's epochs were 3, 7, 10 and 10 trials before 2026-09-18. Measured then
(2026-09-17): Spearman(mi, n_trials) = -0.49, p = 1.6e-90 -- raw MI runs 0.00033
at three trials against 0.00014 at ten, so "Trials 1-3" beats "Expert" by a
factor of two on sample size alone and a naive contrast returns p = 0.0000 with
no coupling change whatever. ``mi_corrected`` (= ``mi - mi_surrogate_mean``)
carries none of it: Spearman = +0.011, p = 0.68. On size-matched epochs
(Intermediate vs Expert, both ten trials) the learning change is p = 0.63.

Theta and gamma coupling between areas, and theta-gamma coupling within and between.

Reads the voltage exports once per animal (via `area_signals`), then computes,
for both referencing schemes:

* **same-frequency amplitude coupling** between every area pair, in theta, low
  gamma and high gamma -- the raw envelope correlation AND the orthogonalised one
  that removes the zero-lag component;
* **theta-gamma phase-amplitude coupling WITHIN each area** -- theta phase and
  gamma amplitude from the same signal;
* **theta-gamma phase-amplitude coupling BETWEEN areas** -- theta phase from one,
  gamma amplitude from the other, in both directions, which are different
  questions and are kept apart.

Two tables:

``lfp_coupling_trials_<cohort>.csv``
    One row per trial, bipolar only. This is the "across trials" view; bipolar
    because it is the reference that survives the shared field.
``lfp_coupling_epochs_<cohort>.csv``
    One row per epoch, BOTH references, with surrogate statistics for the PAC
    measures. This is what the statistical claims rest on.

Everything here is shaped by 2026-09-17: cross-area coupling on this probe is a
shared field and has no direction. A raw envelope correlation between two
electrodes in one field is large and meaningless, and between-area PAC is
spurious in a way that looks specific -- one source carrying theta-gamma coupling
seen by two electrodes gives theta phase at one predicting gamma amplitude at the
other. Hence bipolar first, orthogonalised first, and the within-area value
reported beside the between-area one.

    /opt/anaconda3/bin/python scripts/run_lfp_coupling.py --cohort task
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import analysis, area_signals, config, coupling, filtering  # noqa: E402

THETA = (4.0, 8.0)
AMP_BANDS = {"low_gamma": (30.0, 80.0), "high_gamma": (80.0, 150.0)}
SAME_BANDS = {"theta": THETA, **AMP_BANDS}
EPOCHS = ("All", "Naive", "Intermediate", "Expert")
#: One theta cycle is ~150-250 ms; a modulation index wants many of them.
MIN_SAMPLES = 4000
N_SURROGATES = 100
REFS = ("monopolar", "bipolar")


def _filtered(sig: np.ndarray, sos_cache: dict) -> dict:
    """``{band: envelope}`` plus ``('phase', 'theta')``, filtered once per signal."""
    out = {("env", b): filtering.band_envelope(sig, sos_cache[b]) for b in SAME_BANDS}
    out[("phase", "theta")] = filtering.band_phase(sig, sos_cache["theta"])
    return out


def _trial_rows(feats, chans, base) -> list[dict]:
    """Per-trial measures, bipolar only — the evolution view."""
    rows = []
    areas = sorted(chans)
    for a1, a2 in combinations(areas, 2):
        for band in SAME_BANDS:
            e1, e2 = feats[(a1, "bipolar")][("env", band)], feats[(a2, "bipolar")][("env", band)]
            if np.std(e1) == 0 or np.std(e2) == 0:
                continue
            rows.append({**base, "measure": "same_freq", "band": band,
                         "area_a": a1, "area_b": a2,
                         "value": float(np.corrcoef(e1, e2)[0, 1])})
    for area in areas:
        for band in AMP_BANDS:
            rows.append({**base, "measure": "pac_within", "band": band,
                         "area_a": area, "area_b": area,
                         "value": coupling.modulation_index(
                             feats[(area, "bipolar")][("phase", "theta")],
                             feats[(area, "bipolar")][("env", band)])})
    for a1, a2 in permutations(areas, 2):
        for band in AMP_BANDS:
            rows.append({**base, "measure": "pac_between", "band": band,
                         "area_a": a1, "area_b": a2,
                         "value": coupling.modulation_index(
                             feats[(a1, "bipolar")][("phase", "theta")],
                             feats[(a2, "bipolar")][("env", band)])})
    return rows


def _epoch_rows(per_trial, trials, chans, base, sos_cache) -> list[dict]:
    """Per-epoch measures on concatenated trials, both references, with surrogates."""
    rows = []
    areas = sorted(chans)
    joined = {(a, r): np.concatenate([per_trial[t][(a, r)] for t in trials])
              for a in areas for r in REFS}
    for ref in REFS:
        for a1, a2 in combinations(areas, 2):
            for band, edges in SAME_BANDS.items():
                x, y = joined[(a1, ref)], joined[(a2, ref)]
                rows.append({**base, "reference": ref, "measure": "same_freq",
                             "band": band, "area_a": a1, "area_b": a2,
                             "raw_r": coupling.envelope_correlation(x, y, sos_cache[band]),
                             "orth_r": coupling.orthogonalised_envelope_correlation(
                                 x, y, sos_cache[band])})
        for area in areas:
            for band, edges in AMP_BANDS.items():
                out = coupling.modulation_index_with_surrogates(
                    joined[(area, ref)], None, fs=config.FS, phase_band=THETA,
                    amp_band=edges, n_surrogates=N_SURROGATES, seed=int(base["mouse_id"]))
                out["mi_corrected"] = out["mi"] - out["mi_surrogate_mean"]
                rows.append({**base, "reference": ref, "measure": "pac_within",
                             "band": band, "area_a": area, "area_b": area, **out})
        for a1, a2 in permutations(areas, 2):
            for band, edges in AMP_BANDS.items():
                out = coupling.modulation_index_with_surrogates(
                    joined[(a1, ref)], joined[(a2, ref)], fs=config.FS,
                    phase_band=THETA, amp_band=edges,
                    n_surrogates=N_SURROGATES, seed=int(base["mouse_id"]))
                out["mi_corrected"] = out["mi"] - out["mi_surrogate_mean"]
                rows.append({**base, "reference": ref, "measure": "pac_between",
                             "band": band, "area_a": a1, "area_b": a2, **out})
    return rows


def run_one(cache: Path, cohort_name: str):
    z = np.load(cache, allow_pickle=False)
    mouse, probe = int(z["mouse_id"]), str(z["probe"])
    ch = config.get_cohort(cohort_name)
    chans, per_trial, elapsed = area_signals.read_trial_signals(
        z, ch, min_samples=MIN_SAMPLES)
    if len(chans) < 2 or not per_trial:
        print(f"[coup] {mouse}/{probe}: fewer than two usable areas or no long trial, skipped",
              flush=True)
        return [], []

    sos_cache = {b: filtering.design_band_sos(e, fs=int(config.FS))
                 for b, e in SAME_BANDS.items()}
    lp = analysis.cohort_learning_points(ch).get(mouse)
    tag = {"cohort": cohort_name, "mouse_id": mouse, "probe": probe,
           "learning_point": lp}

    trial_rows = []
    for t in sorted(per_trial):
        feats = {(a, "bipolar"): _filtered(per_trial[t][(a, "bipolar")], sos_cache)
                 for a in chans}
        trial_rows += _trial_rows(feats, chans, {**tag, "trial": t + 1})

    n_trials = int(min(analysis.cohort_trial_counts(ch).get(mouse, len(per_trial)),
                       len(per_trial)))
    # The project's THREE-epoch scheme: Naive = trials 1-10, Expert = the ten
    # trials from the learning point. Ten trials each, so every across-epoch
    # contrast is count-matched by construction -- which the Tort index REQUIRES,
    # being positively biased at small n (Spearman(mi, n_trials) = -0.49 on the
    # four-epoch scheme this replaces, whose Naive was three trials).
    dp = analysis.disengagement_points(ch).get(mouse, np.nan)
    usable = sorted(per_trial)
    if np.isfinite(dp):
        usable = [t for t in usable if t + 1 <= dp]
    windows = {"All": usable}
    for name, tr in zip(("Naive", "Intermediate", "Expert"),
                        analysis.epoch_indices(lp, n_trials)):
        keep = [t for t in (np.asarray(tr, int) - 1) if t in per_trial]
        # An LP-relative epoch can run past disengagement when LP and DP are close
        # (418: LP = 26, DP = 27). Drop it rather than compare engaged with
        # disengaged -- that confound killed a headline result on 2026-09-18.
        if keep and np.isfinite(dp) and max(keep) + 1 > dp:
            keep = []
        windows[name] = keep

    epoch_rows = []
    for epoch in EPOCHS:
        trials = windows.get(epoch, [])
        if not trials:
            continue
        epoch_rows += _epoch_rows(per_trial, trials, chans,
                                  {**tag, "epoch": epoch, "n_trials": len(trials)},
                                  sos_cache)

    print(f"[coup] {cohort_name[:4]:<4} {mouse}/{probe:9s} {len(chans)} areas "
          f"{sorted(chans)}, {len(per_trial):3d} trials -> {len(trial_rows):5d} trial rows, "
          f"{len(epoch_rows):4d} epoch rows  (read {elapsed:.0f}s)", flush=True)
    return trial_rows, epoch_rows


def write(rows, path: Path) -> None:
    if not rows:
        print(f"[coup] nothing to write to {path.name}")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[coup] wrote {path.name} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="task", choices=sorted(config.COHORTS))
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    in_dir = config.RESULTS_DIR / f"lfp_band_trials_{args.cohort}"
    files = sorted(in_dir.glob("*.npz"))
    if args.only:
        want = {m.strip() for m in args.only.split(",")}
        files = [f for f in files if f.name.split("_")[0] in want]
    if not files:
        print(f"[coup] no caches in {in_dir}")
        return
    print(f"[coup] cohort={args.cohort}: {len(files)} files, "
          f"{N_SURROGATES} surrogates per PAC cell")

    trials, epochs = [], []
    for f in files:
        a, b = run_one(f, args.cohort)
        trials += a
        epochs += b
    write(trials, config.RESULTS_DIR / f"lfp_coupling_trials_{args.cohort}.csv")
    write(epochs, config.RESULTS_DIR / f"lfp_coupling_epochs_{args.cohort}.csv")

    for measure in ("pac_within", "pac_between"):
        for ref in REFS:
            sel = [r for r in epochs if r["measure"] == measure
                   and r["reference"] == ref and r.get("epoch") == "All"
                   and np.isfinite(r.get("p", np.nan))]
            if not sel:
                continue
            p = np.array([r["p"] for r in sel])
            print(f"[coup] {measure:12s} {ref:10s} epoch=All: {len(sel)} cells, "
                  f"p < 0.05 in {int((p < 0.05).sum())} ({(p < 0.05).mean():.0%})")


if __name__ == "__main__":
    main()
