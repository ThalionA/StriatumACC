"""Tests for partial CCA -- regressing a confound out before the CCA."""

from __future__ import annotations

import numpy as np

from striatum_cca import config, core, partial

CFG = config.DEFAULT


# ---------------------------------------------------------------------------
# partial_out
# ---------------------------------------------------------------------------
def test_partial_out_removes_a_linear_confound():
    rng = np.random.default_rng(0)
    confound = rng.standard_normal((200, 3))
    target = confound @ rng.standard_normal((3, 4))    # exact linear function
    resid = partial.partial_out(target, confound)
    assert np.allclose(resid, 0.0, atol=1e-8)


def test_partial_out_keeps_the_orthogonal_part():
    rng = np.random.default_rng(1)
    confound = rng.standard_normal((300, 2))
    independent = rng.standard_normal((300, 2))
    target = confound @ rng.standard_normal((2, 2)) + independent
    resid = partial.partial_out(target, confound)
    # the confound-predictable part is gone; the independent part survives
    assert np.corrcoef(resid[:, 0], independent[:, 0])[0, 1] > 0.9


# ---------------------------------------------------------------------------
# partial_out_tensor
# ---------------------------------------------------------------------------
def test_partial_out_tensor_preserves_shape():
    rng = np.random.default_rng(2)
    tensor = rng.standard_normal((10, 50, 6))
    confound = rng.standard_normal((10, 50, 4))
    assert partial.partial_out_tensor(tensor, confound).shape == (10, 50, 6)


def test_partial_out_tensor_removes_the_confound():
    rng = np.random.default_rng(3)
    confound = rng.standard_normal((8, 40, 3))
    flat_z = confound.reshape(8 * 40, 3)
    tensor = (flat_z @ rng.standard_normal((3, 5))).reshape(8, 40, 5)
    out = partial.partial_out_tensor(tensor, confound)
    assert np.allclose(out, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# partial_cca_cv
# ---------------------------------------------------------------------------
def test_partial_cca_cv_collapses_z_mediated_coupling():
    # X and Y share structure only through Z -> partialling Z out kills the CC.
    rng = np.random.default_rng(4)
    n_tr, n_bins, k = 12, 50, 5
    z = rng.standard_normal((n_tr, n_bins, k))
    x = z + 0.3 * rng.standard_normal((n_tr, n_bins, k))
    y = z + 0.3 * rng.standard_normal((n_tr, n_bins, k))
    plain = core.cca_cv(x, y, CFG).held_out_r[0]
    part = partial.partial_cca_cv(x, y, z, CFG).held_out_r[0]
    assert plain > 0.6
    assert part < 0.3
