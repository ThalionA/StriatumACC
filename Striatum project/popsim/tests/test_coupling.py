"""Tests for inter-area coupling resolution."""

from __future__ import annotations

import numpy as np
import pytest

from popsim.coupling import CouplingEdge, resolve_latents, topological_order


def test_edge_rejects_negative_lag():
    with pytest.raises(ValueError):
        CouplingEdge("A", "B", lag=-1)


def test_edge_rejects_zero_lag_self_loop():
    with pytest.raises(ValueError):
        CouplingEdge("A", "A", lag=0)


def test_topological_order_simple_chain():
    edges = [CouplingEdge("A", "C", lag=0), CouplingEdge("C", "B", lag=0)]
    order = topological_order(["A", "B", "C"], edges)
    assert order.index("A") < order.index("C") < order.index("B")


def test_topological_order_detects_cycle():
    edges = [CouplingEdge("A", "B", lag=0), CouplingEdge("B", "A", lag=0)]
    with pytest.raises(ValueError):
        topological_order(["A", "B"], edges)


def test_lagged_cycle_is_allowed():
    # A lag>0 cycle is fine: it references past values only.
    edges = [CouplingEdge("A", "B", lag=0), CouplingEdge("B", "A", lag=5)]
    order = topological_order(["A", "B"], edges)
    assert set(order) == {"A", "B"}


def test_zero_lag_injection_is_exact():
    n = 200
    a = np.random.default_rng(0).standard_normal((n, 2))
    intrinsic = {"A": a, "B": np.zeros((n, 2))}
    edges = [CouplingEdge("A", "B", gain=2.0, lag=0, matrix=np.eye(2))]
    out = resolve_latents(intrinsic, edges)
    np.testing.assert_allclose(out["B"], 2.0 * a)


def test_lagged_injection_shifts_in_time():
    n = 100
    a = np.random.default_rng(1).standard_normal((n, 1))
    intrinsic = {"A": a, "B": np.zeros((n, 1))}
    lag = 7
    out = resolve_latents(intrinsic, [CouplingEdge("A", "B", lag=lag, matrix=np.eye(1))])
    np.testing.assert_allclose(out["B"][lag:], a[:-lag])
    np.testing.assert_allclose(out["B"][:lag], 0.0)


def test_mediation_chain_composes():
    n = 50
    a = np.random.default_rng(2).standard_normal((n, 1))
    intrinsic = {"A": a, "C": np.zeros((n, 1)), "B": np.zeros((n, 1))}
    edges = [
        CouplingEdge("A", "C", lag=0, matrix=np.eye(1)),
        CouplingEdge("C", "B", lag=0, matrix=np.eye(1)),
    ]
    out = resolve_latents(intrinsic, edges)
    np.testing.assert_allclose(out["C"], a)
    np.testing.assert_allclose(out["B"], a)


def test_epoch_mask_restricts_injection():
    n = 100
    intrinsic = {"A": np.ones((n, 1)), "B": np.zeros((n, 1))}
    edges = [CouplingEdge("A", "B", lag=0, matrix=np.eye(1), epochs=[(40, 60)])]
    out = resolve_latents(intrinsic, edges)
    assert np.all(out["B"][:40] == 0.0)
    assert np.all(out["B"][40:60] == 1.0)
    assert np.all(out["B"][60:] == 0.0)
