#!/usr/bin/env python3
"""Is there a direction of communication in the LFP, and does it move with learning?

Three panels, answering the three questions from the 2026-09-09 meeting in order.

(a) Is there a direction at all? Per-animal phase-slope index for every area pair
    and band, over all trials. Positive means the first-named area leads. Shown
    for both referencing schemes: monopolar (the area's mean channel, maximum
    exposure to the shared field) and bipolar (adjacent-channel differences,
    which cancel the far field to first order). A direction present monopolar and
    absent bipolar is the field, not an interaction.

(b) Does it change with learning? The same index across the four epochs, bipolar
    only -- the conservative reference.

(c) Does it differ between task and yoked control? Both cohorts throughout.

The animal is the unit of analysis. Each per-animal value is itself a z-score
against a jackknife over segments; the population test is a Wilcoxon signed-rank
against zero across animals, BH-FDR within each panel's family.

    /opt/anaconda3/bin/python scripts/plot_lfp_psi.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config, figstyle, trials  # noqa: E402

BANDS = ("theta", "beta", "low_gamma", "high_gamma")
EPOCHS = trials.EPOCHS
COHORTS = (("task", "Task", "#1f4e79"), ("control", "Control 1", "#e69f00"))
MIN_MICE = 3


def _load(cohort: str) -> list[dict]:
    path = config.RESULTS_DIR / f"lfp_psi_{cohort}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("cohort", "probe", "epoch", "area_a", "area_b", "reference", "band"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = np.nan
    return rows


def _per_animal(rows, *, pair, band, ref, epoch):
    by_mouse = defaultdict(list)
    for r in rows:
        if (r["area_a"], r["area_b"]) != pair or r["band"] != band:
            continue
        if r["reference"] != ref or r["epoch"] != epoch:
            continue
        if np.isfinite(r["z"]):
            by_mouse[int(r["mouse_id"])].append(r["z"])
    return {m: float(np.mean(v)) for m, v in by_mouse.items()}


def _test(vals: np.ndarray):
    """(mean, sem, p, floor). p is nan when the signed-rank test cannot reach 0.05."""
    n = vals.size
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    p = float(stats.wilcoxon(vals).pvalue) if n >= 6 else np.nan
    return mean, sem, p, 2.0 / 2 ** n


def main() -> None:
    data = {k: _load(k) for k, _, _ in COHORTS}
    if not any(data.values()):
        print("[plot] no PSI tables; run scripts/run_lfp_psi.py first")
        return
    all_rows = [r for rows in data.values() for r in rows]
    pairs = sorted({(r["area_a"], r["area_b"]) for r in all_rows})

    # constrained layout: tight_layout cannot handle a gridspec with these
    # height ratios and warns that the result may be wrong.
    fig = plt.figure(figsize=(6.0 + 2.6 * len(pairs), 11.5), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 0.9])
    lines: list[str] = []

    # ---------------- (a) is there a direction at all? -----------------------
    axa = fig.add_subplot(gs[0])
    labels, xs = [], []
    x = 0
    for pair in pairs:
        for band in BANDS:
            drew = False
            for ref, marker, dx in (("monopolar", "o", -0.16), ("bipolar", "s", 0.16)):
                for key, label, colour in COHORTS:
                    vals = _per_animal(data.get(key, []), pair=pair, band=band,
                                       ref=ref, epoch="All")
                    if len(vals) < MIN_MICE:
                        continue
                    v = np.array(list(vals.values()))
                    mean, sem, p, floor = _test(v)
                    off = dx + (0.06 if key == "control" else -0.06)
                    axa.errorbar(x + off, mean, yerr=sem, fmt=marker, ms=7,
                                 color=colour, capsize=3,
                                 mfc=colour if ref == "bipolar" else "white",
                                 mew=1.6, zorder=3)
                    drew = True
                    verdict = (f"p = {p:.3f}" if np.isfinite(p)
                               else f"UNDERPOWERED (n = {v.size}, floor {floor:.3f})")
                    lines.append(f"  {pair[0]}->{pair[1]:4s} {band:11s} {ref:10s} "
                                 f"{label:9s} z = {mean:+6.2f} ± {sem:4.2f} "
                                 f"(N = {v.size} mice, {verdict})")
            if drew:
                labels.append(f"{pair[0]}→{pair[1]}\n{figstyle.BAND_LABEL[band].split()[0]}")
                xs.append(x)
                x += 1
    axa.axhline(0, color="k", lw=1.0)
    axa.set_xticks(xs)
    axa.set_xticklabels(labels, fontsize=7)
    axa.set_ylabel("phase-slope index (z)\npositive = first area leads")
    axa.set_title("(a) Is there a direction? Open = monopolar (sees the shared field), "
                  "filled = bipolar (far field cancelled). Blue = task, orange = control.",
                  fontsize=10)

    # ---------------- (b) does it change with learning? ----------------------
    axb = fig.add_subplot(gs[1])
    xe = np.arange(len(EPOCHS))
    # Only the pairs a cohort can actually test. The hippocampal pairs rest on
    # three animals, where a signed-rank test cannot reach 0.05 at all, and
    # plotting them here would fill the panel with lines that carry no evidence.
    testable = [pr for pr in pairs
                if max((len(_per_animal(data.get(k, []), pair=pr, band=b,
                                        ref="bipolar", epoch="All"))
                        for k, _, _ in COHORTS for b in BANDS), default=0) >= 6]
    for pi, pair in enumerate(testable):
        for key, label, colour in COHORTS:
            m, e = [], []
            for ep in EPOCHS:
                # Average an animal's bands FIRST, then aggregate over animals.
                # Pooling the four bands straight into one array would let one
                # mouse contribute four values and shrink the error bar by ~2x
                # on replication that is not there.
                by_mouse = defaultdict(list)
                for band in BANDS:
                    for mouse, val in _per_animal(data.get(key, []), pair=pair,
                                                  band=band, ref="bipolar",
                                                  epoch=ep).items():
                        by_mouse[mouse].append(val)
                per_animal = np.array([np.mean(v) for v in by_mouse.values()])
                if per_animal.size < MIN_MICE:
                    m.append(np.nan)
                    e.append(np.nan)
                else:
                    m.append(float(per_animal.mean()))
                    e.append(float(per_animal.std(ddof=1) / np.sqrt(per_animal.size)))
            if np.all(np.isnan(m)):
                continue
            axb.errorbar(xe + 0.06 * pi + (0.02 if key == "control" else -0.02), m,
                         yerr=e, marker="o", ms=4, lw=1.4, capsize=3, color=colour,
                         alpha=0.55 + 0.45 * (pi == 0),
                         label=f"{label}, {pair[0]}→{pair[1]}")
    axb.axhline(0, color="k", lw=1.0)
    axb.set_xticks(xe)
    axb.set_xticklabels(["1-3", "4-10", "Inter", "Expert"])
    axb.set_ylabel("phase-slope index (z), bipolar")
    axb.set_title("(b) Does it change with learning? Bipolar reference; an animal's bands "
                  "are averaged before the animals are.\nOnly pairs with N >= 6 mice are "
                  "drawn — the hippocampal pairs have three. Control epochs are matched TIME "
                  "windows.", fontsize=10)
    axb.legend(fontsize=6.5, frameon=False, ncol=2)

    # ---------------- (c) what the two references say about each other -------
    axc = fig.add_subplot(gs[2])
    mono, bip = [], []
    for pair in pairs:
        for band in BANDS:
            for key, _, _ in COHORTS:
                a = _per_animal(data.get(key, []), pair=pair, band=band,
                                ref="monopolar", epoch="All")
                b = _per_animal(data.get(key, []), pair=pair, band=band,
                                ref="bipolar", epoch="All")
                for m in set(a) & set(b):
                    mono.append(a[m])
                    bip.append(b[m])
    if mono:
        mono, bip = np.array(mono), np.array(bip)
        axc.scatter(mono, bip, s=18, alpha=0.65, color="#40567a")
        lim = float(np.nanmax(np.abs(np.concatenate([mono, bip])))) * 1.1
        axc.plot([-lim, lim], [-lim, lim], "k:", lw=1, label="equal")
        axc.axhline(0, color="0.6", lw=0.8)
        axc.axvline(0, color="0.6", lw=0.8)
        axc.set_xlim(-lim, lim)
        axc.set_ylim(-lim, lim)
        r = float(np.corrcoef(mono, bip)[0, 1])
        keep = float(np.mean(np.sign(mono) == np.sign(bip)))
        axc.set_title(f"(c) The same cells under both references: r = {r:+.2f}, "
                      f"sign agrees in {keep:.0%}. Points on the diagonal mean the "
                      f"far field was not what produced the direction.", fontsize=10)
        axc.set_xlabel("monopolar z")
        axc.set_ylabel("bipolar z")
        axc.legend(fontsize=8, frameon=False)
        lines.append(f"\n  monopolar vs bipolar over {mono.size} cells: "
                     f"r = {r:+.2f}, sign agreement {keep:.0%}")

    fig.suptitle("Direction of communication from the LFP (phase-slope index)\n"
                 "PSI is blind to instantaneous mixing by construction, which is why it is "
                 "usable here at all:\nthe distance control showed cross-area coupling on this "
                 "probe is a shared field.", fontsize=12)
    figstyle.save_pair(fig, "lfp_psi_direction")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
