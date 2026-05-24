"""Round-8 parameter-sweep definitions.

``build_sweep(name)`` returns the list of ``(tag, data_path, cfg)`` triples for
a named sweep. Defined in one place so run_stage2.py, run_stage3.py and the
summary all agree on the grid.

SPATIAL sweep = the FULL Cartesian product of every analysis knob (round 8,
"do all the combinations"):

    bin width      x  AXIS_BINS      (2.5 cm / 5 cm)
    CCA type       x  AXIS_CCA       (residual / signal)
    fast-spiking   x  AXIS_FS        (excluded / included)
    z-scoring      x  AXIS_Z         (on / off)
    min units      x  AXIS_MIN_UNITS (4 / 6 / 10)
    LP criterion   x  AXIS_LP        (7 / 8 consecutive)
    PC-count rule  x  AXIS_KRULE     (samples 15/25/40, fixed 3/5/10/20/30,
                                      variance 75/85/95 %)

Total = 2*2*2*2*3*2*11 = 1056 configs. Edit the AXIS_* lists below to trim.

TEMPORAL sweep = 40 ms and 20 ms (signal CCA, FS-excluded, disengaged
traversals excluded; 10 ms is computationally intractable here).
"""

from __future__ import annotations

import dataclasses

from . import config

# --- spatial sweep axes (edit to trim the grid) ------------------------------
AXIS_BINS = (("s2p5", None, 10), ("s5cm", config.PREPROCESSED_DATA_5CM, 5))
AXIS_CCA = ((True, "res"), (False, "sig"))
AXIS_FS = ((True, "fsX"), (False, "fsI"))
AXIS_Z = ((True, "z1"), (False, "z0"))
AXIS_MIN_UNITS = (4, 6, 10)
AXIS_LP = (7, 8)
# PC-count rule: (tag, cfg-override dict).
AXIS_KRULE = (
    ("samp15", {"k_mode": "samples", "samples_per_pc": 15}),
    ("samp25", {"k_mode": "samples", "samples_per_pc": 25}),
    ("samp40", {"k_mode": "samples", "samples_per_pc": 40}),
    ("fix03", {"k_mode": "fixed", "k_fixed": 3}),
    ("fix05", {"k_mode": "fixed", "k_fixed": 5}),
    ("fix10", {"k_mode": "fixed", "k_fixed": 10}),
    ("fix20", {"k_mode": "fixed", "k_fixed": 20}),
    ("fix30", {"k_mode": "fixed", "k_fixed": 30}),
    ("var75", {"k_mode": "variance", "k_variance": 0.75}),
    ("var85", {"k_mode": "variance", "k_variance": 0.85}),
    ("var95", {"k_mode": "variance", "k_variance": 0.95}),
)


def _spatial() -> list[tuple]:
    base = config.DEFAULT
    out: list[tuple] = []
    for btag, path, lag in AXIS_BINS:
        for resid, ctag in AXIS_CCA:
            for fs, ftag in AXIS_FS:
                for z, ztag in AXIS_Z:
                    for mu in AXIS_MIN_UNITS:
                        for lpc in AXIS_LP:
                            for krtag, kover in AXIS_KRULE:
                                cfg = dataclasses.replace(
                                    base, max_lag_bins=lag,
                                    subtract_trial_mean=resid,
                                    exclude_fast_spiking=fs, zscore_units=z,
                                    min_units=mu, lp_min_consecutive=lpc,
                                    **kover)
                                tag = (f"{btag}_{ctag}_{ftag}_{ztag}"
                                       f"_mu{mu:02d}_lp{lpc}_{krtag}")
                                out.append((tag, path, cfg))
    out.sort(key=lambda t: t[1] is not None)        # group by data file
    return out


def _temporal() -> list[tuple]:
    # 10 ms is intractable here; 40/20 ms, signal CCA, FS-excluded, disengaged
    # traversals excluded. +/-200 ms lag scan.
    base = config.DEFAULT
    out: list[tuple] = []
    for ms, lag in ((40, 5), (20, 10)):
        cfg = dataclasses.replace(
            base, bin_mode="temporal", temporal_bin_ms=ms, max_lag_bins=lag,
            subtract_trial_mean=False, exclude_fast_spiking=True,
            zscore_units=True)
        out.append((f"t{ms:02d}", None, cfg))
    return out


def build_sweep(name: str) -> list[tuple]:
    """``(tag, data_path, cfg)`` triples for sweep ``name``.

    ``name`` is "spatial", "temporal" or "all"; ``data_path`` is None for the
    default 2.5 cm file.
    """
    if name == "spatial":
        return _spatial()
    if name == "temporal":
        return _temporal()
    if name == "all":
        return _spatial() + _temporal()
    raise ValueError(f"unknown sweep: {name!r}")
