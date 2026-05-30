"""Recovery benchmark: do standard analyses recover the configured ground truth?

For each scenario we simulate the populations, reduce each area to its top
principal components (as a real analysis would, never touching the latents), and
run the lag-0 / lagged / partial CCA from :mod:`popsim.metrics` on the *neural*
data. Each scenario carries a qualitative expectation; :func:`run_benchmark`
returns one :class:`RecoveryRow` per scenario with the recovered numbers and a
PASS/FAIL verdict, and :func:`format_table` renders them.

This is the end-to-end check that the generator and the analysis agree: the
coupling we dialed in at the latent level is the coupling a population-level
method reads back out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from . import scenarios
from .metrics import (
    canonical_variates,
    cca,
    lag_of_peak_xcorr,
    partial_cca,
    pca_reduce,
)
from .simulate import SimConfig, simulate

__all__ = ["RecoveryRow", "analyse_recovery", "run_benchmark", "format_table"]


@dataclass
class RecoveryRow:
    """Recovered metrics + verdict for one scenario."""

    scenario: str
    cca1: float            # top canonical corr, neural CCA(A, B)
    partial_cca1: float    # top canonical corr, partial CCA(A, B | C)
    drop_frac: float       # fractional drop from cca1 to partial_cca1
    peak_lag: int          # lag (bins) of top canonical variate pair, A vs B
    peak_lag2: int         # lag of the 2nd canonical variate pair
    n_strong: int          # canonical corrs >= strong_threshold
    latent_cca1: float     # CCA on the (hidden) latents -- ground-truth coupling
    pop_corr: float        # |corr| of population-mean activity (shared noise)
    epoch_lags: list[int]  # per-epoch peak lags (empty if no epochs)
    expected: str
    passed: bool


def analyse_recovery(
    result, k: int = 5, max_lag: int = 50, strong_threshold: float = 0.3
) -> dict:
    """Recover coupling metrics from one simulation's neural data."""
    a, b, c = "A", "B", "C"
    Xa = pca_reduce(result.neural[a], k)
    Xb = pca_reduce(result.neural[b], k)
    Xc = pca_reduce(result.neural[c], k)

    corrs, _, _ = cca(Xa, Xb)
    cca1 = float(corrs[0])
    n_strong = int(np.sum(corrs >= strong_threshold))

    pcorrs, _, _ = partial_cca(Xa, Xb, Xc)
    partial_cca1 = float(pcorrs[0])
    drop_frac = (cca1 - partial_cca1) / cca1 if cca1 > 1e-6 else 0.0

    _, u0, v0 = canonical_variates(Xa, Xb, 0)
    peak_lag = lag_of_peak_xcorr(u0, v0, max_lag)
    _, u1, v1 = canonical_variates(Xa, Xb, 1)
    peak_lag2 = lag_of_peak_xcorr(u1, v1, max_lag)

    latent_cca1 = float(cca(result.latents[a], result.latents[b])[0][0])
    mean_a = result.neural[a].mean(axis=1)
    mean_b = result.neural[b].mean(axis=1)
    pop_corr = abs(float(np.corrcoef(mean_a, mean_b)[0, 1]))

    epoch_lags: list[int] = []
    bounds = result.config.epoch_boundaries
    if bounds:
        edges = [0, *bounds, len(u0)]
        for i in range(len(edges) - 1):
            seg = slice(edges[i], edges[i + 1])
            epoch_lags.append(lag_of_peak_xcorr(u0[seg], v0[seg], max_lag))

    return {
        "cca1": cca1,
        "partial_cca1": partial_cca1,
        "drop_frac": float(drop_frac),
        "peak_lag": int(peak_lag),
        "peak_lag2": int(peak_lag2),
        "n_strong": n_strong,
        "latent_cca1": latent_cca1,
        "pop_corr": pop_corr,
        "epoch_lags": epoch_lags,
    }


# Per-scenario expectation: human description + a predicate over the recovered
# metrics dict. Keeping these next to the scenarios documents what "correct
# recovery" means for each ground truth.
def _expectations() -> dict[str, tuple[str, Callable[[dict], bool]]]:
    return {
        "no_coupling": (
            "A-B uncoupled (cca1 low)",
            lambda r: r["cca1"] < 0.3,
        ),
        "zero_lag": (
            "A->B, peak lag ~ 0",
            lambda r: r["cca1"] > 0.4 and abs(r["peak_lag"]) <= 3,
        ),
        "lagged": (
            "A leads B by ~+10",
            lambda r: r["cca1"] > 0.4 and abs(r["peak_lag"] - 10) <= 4,
        ),
        "mediated": (
            "A-B collapses given C",
            lambda r: r["cca1"] > 0.3 and r["drop_frac"] > 0.5,
        ),
        "epoch_varying": (
            "direction reverses across epochs",
            lambda r: (
                len(r["epoch_lags"]) == 3
                and r["epoch_lags"][0] > 0
                and r["epoch_lags"][1] < 0
            ),
        ),
        "bidirectional": (
            "reciprocal: +ve and -ve lags",
            lambda r: r["cca1"] > 0.3 and (r["peak_lag"] * r["peak_lag2"] < 0),
        ),
        "common_input": (
            "A-B collapses given C",
            lambda r: r["cca1"] > 0.3 and r["drop_frac"] > 0.5,
        ),
        "rotated_subspace": (
            "rank-2 subspace recovered",
            lambda r: r["n_strong"] >= 2,
        ),
        "partial_mediation": (
            "survives partialling C (direct path)",
            lambda r: (
                r["cca1"] > 0.4
                and r["drop_frac"] < 0.3
                and r["partial_cca1"] > 0.4
            ),
        ),
        "noise_correlation": (
            "latents independent, pops correlated",
            lambda r: r["latent_cca1"] < 0.25 and r["pop_corr"] > 0.3,
        ),
    }


def run_benchmark(
    names: list[str] | None = None,
    n_timesteps: int = 6000,
    k: int = 5,
    max_lag: int = 50,
    builder_kwargs: dict | None = None,
) -> list[RecoveryRow]:
    """Run every scenario and score recovered coupling against ground truth.

    Parameters
    ----------
    names:
        Subset of scenario names (default: all).
    n_timesteps:
        Session length per scenario.
    k:
        PCs retained per area before CCA.
    max_lag:
        Cross-correlogram half-width (bins).
    builder_kwargs:
        Extra kwargs passed to each scenario builder (e.g. ``dynamics``).
    """
    names = names or list(scenarios.SCENARIOS)
    builder_kwargs = builder_kwargs or {}
    expectations = _expectations()
    rows: list[RecoveryRow] = []
    for name in names:
        cfg: SimConfig = scenarios.SCENARIOS[name](
            n_timesteps=n_timesteps, **builder_kwargs
        )
        result = simulate(cfg)
        rec = analyse_recovery(result, k=k, max_lag=max_lag)
        desc, predicate = expectations.get(name, ("(no expectation)", lambda r: True))
        rows.append(
            RecoveryRow(
                scenario=name, expected=desc, passed=bool(predicate(rec)), **rec
            )
        )
    return rows


def format_table(rows: list[RecoveryRow]) -> str:
    """Render benchmark rows as a fixed-width text table."""
    header = (
        f"{'scenario':<18}{'cca1':>6}{'pcca1':>7}{'drop':>6}{'lag':>5}"
        f"{'lag2':>6}{'rank':>5}{'latCC':>7}{'popρ':>6}  {'expected / verdict'}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        verdict = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.scenario:<18}{r.cca1:>6.2f}{r.partial_cca1:>7.2f}"
            f"{r.drop_frac:>6.2f}{r.peak_lag:>5d}{r.peak_lag2:>6d}"
            f"{r.n_strong:>5d}{r.latent_cca1:>7.2f}{r.pop_corr:>6.2f}  "
            f"[{verdict}] {r.expected}"
        )
    n_pass = sum(r.passed for r in rows)
    lines.append("-" * len(header))
    lines.append(f"{n_pass}/{len(rows)} scenarios recovered as expected")
    return "\n".join(lines)


def rows_to_dicts(rows: list[RecoveryRow]) -> list[dict]:
    """JSON-serialisable view of benchmark rows."""
    return [asdict(r) for r in rows]
