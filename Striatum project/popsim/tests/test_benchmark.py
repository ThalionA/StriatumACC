"""Tests for the recovery benchmark and the metrics helpers it relies on."""

from __future__ import annotations

import numpy as np

from popsim.benchmark import format_table, run_benchmark
from popsim.metrics import canonical_variates, pca_reduce


def test_pca_reduce_shape_and_orthogonal_scores():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 20))
    scores = pca_reduce(X, 5)
    assert scores.shape == (2000, 5)
    # PC scores are uncorrelated across components.
    c = np.corrcoef(scores.T)
    off = c - np.diag(np.diag(c))
    assert np.abs(off).max() < 1e-6


def test_canonical_variates_match_top_corr():
    rng = np.random.default_rng(1)
    shared = rng.standard_normal((3000, 2))
    X = shared @ rng.standard_normal((2, 5)) + 0.1 * rng.standard_normal((3000, 5))
    Y = shared @ rng.standard_normal((2, 4)) + 0.1 * rng.standard_normal((3000, 4))
    corr, u, v = canonical_variates(X, Y, 0)
    assert corr > 0.9
    # The returned variates realise that correlation.
    assert abs(abs(np.corrcoef(u, v)[0, 1]) - corr) < 1e-6


def test_benchmark_runs_all_scenarios_and_table_renders():
    rows = run_benchmark(n_timesteps=4000, k=5)
    names = {r.scenario for r in rows}
    assert names == set(__import__("popsim").scenarios.SCENARIOS)
    table = format_table(rows)
    assert "scenario" in table and "PASS" in table or "FAIL" in table


def test_benchmark_all_verdicts_pass():
    # Every scenario must recover as configured at a reasonable session length.
    rows = run_benchmark(n_timesteps=6000, k=5)
    failed = [r.scenario for r in rows if not r.passed]
    assert not failed, f"benchmark verdicts failed: {failed}"


def test_benchmark_lagged_recovers_positive_lag():
    rows = {r.scenario: r for r in run_benchmark(n_timesteps=6000, k=5)}
    assert abs(rows["lagged"].peak_lag - 10) <= 4
    # Mediated coupling really does collapse when C is partialled out.
    assert rows["mediated"].drop_frac > 0.5
    # Partial mediation does not fully collapse.
    assert rows["partial_mediation"].partial_cca1 > 0.25
