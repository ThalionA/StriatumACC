"""Loading and hierarchical aggregation of the per-cohort arm tables.

Hoisted out of the plotting drivers so the task/control comparison and the
per-cohort figures read the same rows through the same code. Aggregation is
hierarchical everywhere: the animal is the unit of analysis, because channels
within an area correlate at r = 0.83-0.96 and a channel-level error bar would be
a small fraction of the real uncertainty.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from . import config

#: Columns that stay strings when a table is loaded.
TEXT_FIELDS = frozenset({
    "cohort", "group", "probe", "area", "band", "epoch", "window", "lp_source",
    "area_a", "area_b", "trial_rel_lp", "metric", "survives_fdr",
    "reachable", "primary", "differs",
})


def load_arms(name: str, cohort_name: str = "task") -> list[dict]:
    """Rows of ``results/lfp_arms_<name>_<cohort>.csv``, numerics coerced.

    Returns ``[]`` when the table does not exist, so a figure can simply skip a
    cohort that has not been run rather than crashing half way through a set.
    """
    path = config.RESULTS_DIR / f"lfp_arms_{name}_{cohort_name}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r.setdefault("cohort", cohort_name)
        for k, v in r.items():
            if k in TEXT_FIELDS:
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = np.nan
    return rows


def load_both(name: str) -> dict[str, list[dict]]:
    """``{cohort_name: rows}`` for every cohort whose table exists."""
    return {c: rows for c in config.COHORTS
            if (rows := load_arms(name, c))}


def hierarchical(rows, key_fields, value_field):
    """``{key: (mean, sem, n_animals)}`` with the ANIMAL as the unit of analysis.

    One value per animal first (a repeated animal overwrites, which is what we
    want when an animal contributes several probes to the same key), then the
    mean and standard error across animals. ``nan`` values are dropped, so an
    area an animal does not have simply lowers that cell's ``n``.
    """
    by_key = defaultdict(dict)
    for r in rows:
        v = r.get(value_field, np.nan)
        if isinstance(v, float) and np.isfinite(v):
            by_key[tuple(r[f] for f in key_fields)][int(r["mouse_id"])] = v
    out = {}
    for key, per_animal in by_key.items():
        vals = np.array(list(per_animal.values()))
        n = vals.size
        out[key] = (float(vals.mean()),
                    float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan, n)
    return out
