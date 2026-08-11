"""Plumbing tests for the continuous-regime runner (build_present, fit_window)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from striatum_tcca import config, dataio, runner, subspace_window

CFG = dataclasses.replace(config.DEFAULT, min_units=5)


def _streams(n_bins, n_units, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    spikes = rng.normal(size=(n_bins, n_units)).astype(np.float32)
    vel = np.full(n_bins, 10.0)                       # all running
    trial_idx = np.repeat(np.arange(n_trials), n_bins // n_trials)[:n_bins].astype(float)
    return dataio.TemporalStreams(spikes=spikes, vel=vel, trial_idx=trial_idx.astype(float))


def _animal(n_units, areas_units):
    """areas_units: {area: slice/indices} -> boolean masks; all MSN (no FS)."""
    masks = {a: np.zeros(n_units, dtype=bool) for a in config.AREAS}
    for area, ix in areas_units.items():
        masks[area][ix] = True
    return dataio.Animal(animal_id=1, neurontypes=np.ones((n_units, 5)),
                         area_masks=masks, change_point=np.nan,
                         zscored_lick_errors=np.zeros(40), n_trials=40,
                         corridor_path=Path("/dev/null"))


def test_build_present_filters_by_min_units_and_zscores():
    n_units = 20
    a = _animal(n_units, {"DMS": range(0, 10), "ACC": range(10, 18), "V1": range(18, 20)})
    s = _streams(400, n_units, n_trials=20)
    run = dataio.running_mask(s, CFG)
    present = runner.build_present(s, run, a, CFG)
    assert set(present) == {"DMS", "ACC"}             # V1 has 2 < min_units 5
    assert present["DMS"].shape == (run.sum(), 10)
    # z-scored over the running reference (all bins run here) -> ~0 mean, ~1 std.
    np.testing.assert_allclose(present["DMS"].mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(present["DMS"].std(0), 1, atol=1e-6)


def test_fit_window_none_below_min_bins():
    n_units = 20
    a = _animal(n_units, {"DMS": range(0, 10), "ACC": range(10, 20)})
    s = _streams(400, n_units, n_trials=20)
    run = dataio.running_mask(s, CFG)
    present = runner.build_present(s, run, a, CFG)
    tids = s.trial_idx[run]
    out = runner.fit_window(present, tids, "DMS", "ACC", np.array([0, 1]), CFG,
                            k=5, max_lag=2, n_shuffles=3, n_folds=3, min_cell_bins=10_000)
    assert out is None                               # window far below min_cell_bins


def test_fit_window_returns_readout_and_partialled_shapes():
    n_units = 30
    a = _animal(n_units, {"DMS": range(0, 10), "ACC": range(10, 20), "DLS": range(20, 30)})
    s = _streams(2000, n_units, n_trials=20, seed=1)
    run = dataio.running_mask(s, CFG)
    present = runner.build_present(s, run, a, CFG)
    tids = s.trial_idx[run]
    ws = runner.fit_window(present, tids, "DMS", "ACC", np.arange(20), CFG,
                           k=5, max_lag=2, n_shuffles=5, n_folds=3, min_cell_bins=30)
    assert isinstance(ws, subspace_window.WindowSubspace)
    assert ws.weights_x.shape[0] == 10               # DMS units
    assert ws.weights_y.shape[0] == 10               # ACC units
    # Third area (DLS) was available as the partial-out Z (no crash, finite Gini).
    assert np.isfinite(ws.gini_x) and np.isfinite(ws.gini_y)


def test_cross_window_runs_on_two_fits():
    n_units = 20
    a = _animal(n_units, {"DMS": range(0, 10), "ACC": range(10, 20)})
    run_a = dataio.running_mask(_streams(2000, n_units, 20, seed=2), CFG)
    s2 = _streams(2000, n_units, 20, seed=3)
    run2 = dataio.running_mask(s2, CFG)
    p1 = runner.build_present(_streams(2000, n_units, 20, seed=2), run_a, a, CFG)
    p2 = runner.build_present(s2, run2, a, CFG)
    t1 = _streams(2000, n_units, 20, seed=2).trial_idx[run_a]
    t2 = s2.trial_idx[run2]
    w1 = runner.fit_window(p1, t1, "DMS", "ACC", np.arange(20), CFG,
                           k=5, max_lag=2, n_shuffles=5, n_folds=3, min_cell_bins=30)
    w2 = runner.fit_window(p2, t2, "DMS", "ACC", np.arange(20), CFG,
                           k=5, max_lag=2, n_shuffles=5, n_folds=3, min_cell_bins=30)
    cw = runner.cross_window(w1, w2)
    assert 0.0 <= cw.jaccard_x <= 1.0
    assert np.isfinite(cw.rot_x_cc1) and np.isfinite(cw.floor_x_cc1)


def test_fit_window_no_partial_keeps_common_input():
    # A third area (DLS) drives both DMS and ACC. Partialling it out must
    # remove the shared drive; partial_z=False (plain CCA) must keep it.
    n_units = 30
    a = _animal(n_units, {"DMS": range(0, 10), "ACC": range(10, 20),
                          "DLS": range(20, 30)})
    rng = np.random.default_rng(7)
    n_bins, n_trials = 4000, 20
    common = rng.normal(size=(n_bins, 1))
    spikes = rng.normal(size=(n_bins, n_units)).astype(np.float32)
    spikes[:, 0:30] += (1.2 * common).astype(np.float32)
    vel = np.full(n_bins, 10.0)
    trial_idx = np.repeat(np.arange(n_trials), n_bins // n_trials).astype(float)
    s = dataio.TemporalStreams(spikes=spikes, vel=vel, trial_idx=trial_idx)
    run = dataio.running_mask(s, CFG)
    present = runner.build_present(s, run, a, CFG)
    tids = s.trial_idx[run]
    trials = np.arange(n_trials)
    kw = dict(k=5, max_lag=2, n_shuffles=3, n_folds=4, min_cell_bins=100)
    ws_partial = runner.fit_window(present, tids, "DMS", "ACC", trials, CFG, **kw)
    ws_plain = runner.fit_window(present, tids, "DMS", "ACC", trials, CFG,
                                 partial_z=False, **kw)
    assert ws_partial is not None and ws_plain is not None
    assert ws_plain.cc[0] > ws_partial.cc[0] + 0.1
