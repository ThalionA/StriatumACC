"""Task vs Control 1 contrasts on the LFP arms — the test the control group exists for.

Comparing "significant in task" against "not significant in control" is not a
comparison; with 5 control animals almost nothing will reach significance there,
so that reasoning would turn low power into evidence of absence. What is asked
here instead is whether the two groups *differ*: an exact two-sample permutation
test on the per-animal quantity (``striatum_lfp.stats``), BH-corrected over the
area x band family of each (arm, metric). Each row carries the test's floor and
whether 0.05 was reachable at those group sizes.

**Pre-registered primary test (2026-09-25, before the re-run):** evolution,
``delta_log_corridor`` -- Expert minus Naive log10 band power, the log taken of
the linear power averaged over the epoch's bins and trials. Log power is already
within-session relative and needs no normaliser; the z-scored version divides by
a session SD that differs between groups, and carried the earlier p_FDR 0.049
DLS-theta result. Everything else in the table is a sensitivity check.

Four contrasts:

* **evolution** -- per-animal naive-to-expert delta. A change present in yoked
  controls is time in the apparatus, not learning.
* **decoding** -- per-animal R2 above its own rotated-label null. Control 1 runs
  the same corridor, so position information should survive; a task/control gap
  would be about reward contingency, not about the corridor.
* **reliability** -- per-animal split-half reliability, raw and with the linear
  speed component removed first: if the task/control gap is behavioural (task
  animals run the corridor more stereotypically), it shrinks after removal.
* **moving reliability** -- per-animal Expert-epoch reliability, raw. Minus its
  trial shuffle it measures drift, not single-trial reliability: a perfectly
  reliable stationary profile scores ~0 after subtraction.
* **cca** -- per-animal held-out CC1.

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/run_lfp_group_contrast.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config, results_io, stats, trials  # noqa: E402
from striatum_lfp.results_io import load_arms  # noqa: E402

NAIVE_EPOCH, EXPERT_EPOCH = trials.EPOCHS[0], trials.EPOCHS[-1]
PRIMARY_EVOLUTION_METRIC = "log_corridor"


def _mouse(row) -> int:
    """Animal id, whichever column this table calls it.

    The moving-reliability rollup uses ``animal`` because its columns mirror
    ``figures/stability_by_animal.csv``; every other arm uses ``mouse_id``.
    """
    return int(row["mouse_id"] if "mouse_id" in row else row["animal"])


def per_animal_evolution(rows, metric):
    """{(area, band): {mouse: expert - naive}} for one cohort."""
    naive, expert = defaultdict(dict), defaultdict(dict)
    for r in rows:
        key = (r["area"], r["band"])
        v = r.get(metric, np.nan)
        if not (isinstance(v, float) and np.isfinite(v)):
            continue
        if r["epoch"] == NAIVE_EPOCH:
            naive[key][_mouse(r)] = v
        elif r["epoch"] == EXPERT_EPOCH:
            expert[key][_mouse(r)] = v
    out = {}
    for key in set(naive) & set(expert):
        shared = set(naive[key]) & set(expert[key])
        if shared:
            out[key] = {m: expert[key][m] - naive[key][m] for m in shared}
    return out


def per_animal_simple(rows, value, minus=None, where=None):
    """{(area, band): {mouse: value [- minus]}} filtered by ``where``."""
    out = defaultdict(dict)
    for r in rows:
        if where and not where(r):
            continue
        v = r.get(value, np.nan)
        if not (isinstance(v, float) and np.isfinite(v)):
            continue
        if minus is not None:
            m = r.get(minus, np.nan)
            if not (isinstance(m, float) and np.isfinite(m)):
                continue
            v = v - m
        out[(r["area"], r["band"])][_mouse(r)] = v
    return out


def per_animal_pairs(rows, value, minus=None):
    """Same, keyed by area pair instead of area (for the CCA arm)."""
    out = defaultdict(dict)
    for r in rows:
        v, m = r.get(value, np.nan), r.get(minus, 0.0) if minus else 0.0
        if not (isinstance(v, float) and np.isfinite(v)):
            continue
        out[(f"{r['area_a']}-{r['area_b']}", r["band"])][_mouse(r)] = v - m
    return out


def contrast(task_map, ctrl_map, arm, metric, primary=False):
    """Permutation task-vs-control per cell, then BH over this (arm, metric)."""
    cells = []
    for key in sorted(set(task_map) | set(ctrl_map)):
        t = np.array(list(task_map.get(key, {}).values()))
        c = np.array(list(ctrl_map.get(key, {}).values()))
        row = {
            "arm": arm, "metric": metric, "area": key[0], "band": key[1],
            "n_task": t.size, "n_control": c.size,
            "task_mean": float(t.mean()) if t.size else np.nan,
            "task_sem": float(t.std(ddof=1) / np.sqrt(t.size)) if t.size > 1 else np.nan,
            "control_mean": float(c.mean()) if c.size else np.nan,
            "control_sem": float(c.std(ddof=1) / np.sqrt(c.size)) if c.size > 1 else np.nan,
            "difference": float(t.mean() - c.mean()) if t.size and c.size else np.nan,
            "p_raw": stats.permutation_test_two_sample(t, c),
            "p_floor": stats.two_sample_floor(t.size, c.size),
            "primary": primary,
        }
        row["reachable"] = stats.can_reach(row["p_floor"])
        cells.append(row)
    p = np.array([r["p_raw"] for r in cells])
    adjusted, reject = stats.fdr_bh(p, q=0.05)
    for r, a, k in zip(cells, adjusted, reject):
        r["p_fdr"] = float(a) if np.isfinite(a) else np.nan
        r["differs"] = bool(k)
        r["family_size"] = int(np.isfinite(p).sum())
    return cells


def main() -> None:
    out_rows = []

    for metric in ("log_corridor", "z_corridor", "z_corridor_speed_resid",
                   "frac_of_total_corridor"):
        t = per_animal_evolution(load_arms("evolution", "task"), metric)
        c = per_animal_evolution(load_arms("evolution", "control"), metric)
        out_rows += contrast(t, c, "evolution", f"delta_{metric}",
                             primary=metric == PRIMARY_EVOLUTION_METRIC)

    def all_window(r):
        return r["window"] == "All"

    for arm, value, minus in (("decoding", "r2", "null_r2_median"),
                              ("reliability", "split_half_r", None),
                              ("reliability", "split_half_r_speed_resid", None)):
        t = per_animal_simple(load_arms(arm, "task"), value, minus, where=all_window)
        c = per_animal_simple(load_arms(arm, "control"), value, minus, where=all_window)
        out_rows += contrast(t, c, arm, value if minus is None else f"{value}_minus_null")

    def expert(r):
        return r["epoch"] == EXPERT_EPOCH

    t = per_animal_simple(load_arms("moving_reliability_epochs", "task"), "reliability",
                          where=expert)
    c = per_animal_simple(load_arms("moving_reliability_epochs", "control"), "reliability",
                          where=expert)
    out_rows += contrast(t, c, "moving_reliability", "reliability_expert")

    # Behaviour: one value per animal, so a single "cell" per measure.
    for measure in ("speed_profile_split_half_r", "mean_speed_cm_s", "speed_bin_cv"):
        tb = {("behaviour", measure): {_mouse(r): r[measure]
                                       for r in load_arms("behaviour", "task")
                                       if np.isfinite(r.get(measure, np.nan))}}
        cb = {("behaviour", measure): {_mouse(r): r[measure]
                                       for r in load_arms("behaviour", "control")
                                       if np.isfinite(r.get(measure, np.nan))}}
        out_rows += contrast(tb, cb, "behaviour", measure)

    t = per_animal_pairs(load_arms("cca", "task"), "heldout_cc1")
    c = per_animal_pairs(load_arms("cca", "control"), "heldout_cc1")
    out_rows += contrast(t, c, "cca", "heldout_cc1")

    out = config.RESULTS_DIR / "lfp_group_contrast.csv"
    results_io.write_rows(out_rows, out)

    for arm in dict.fromkeys(r["arm"] for r in out_rows):
        sub = [r for r in out_rows if r["arm"] == arm]
        sig = [r for r in sub if r["differs"]]
        tested = sum(r["reachable"] for r in sub)
        print(f"[contrast] {arm:<20} {len(sig)}/{len(sub)} cells differ between groups "
              f"(BH q=0.05; {tested} could reach 0.05)")
        for r in sig:
            print(f"              {r['area']:<8}{r['band']:<11} task {r['task_mean']:+.4f} "
                  f"vs control {r['control_mean']:+.4f}  p={r['p_raw']:.4f} "
                  f"p_FDR={r['p_fdr']:.3f}  (n {r['n_task']} vs {r['n_control']})")
    print(f"[contrast] wrote {out.name} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
