"""Tests for partial CCA -- regressing a confound out before the CCA."""

from __future__ import annotations

import numpy as np

from striatum_tcca import config, core, partial

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
# partial_out_cv (train-only coefficients — leak-free pre-CV residualisation)
# ---------------------------------------------------------------------------
def test_partial_out_cv_fits_on_train_rows_only():
    # The confound->target map differs between train and test halves. Coefficients
    # fit on the train half must residualise train ~exactly but NOT the test half
    # (whose different map the train fit never saw) — that is the leak-free
    # property: the held-out rows do not inform their own residualisation.
    rng = np.random.default_rng(11)
    n = 400
    confound = rng.standard_normal((n, 3))
    train = np.zeros(n, dtype=bool); train[: n // 2] = True
    b_train = rng.standard_normal((3, 2))
    b_test = rng.standard_normal((3, 2))                # different relationship
    target = np.where(train[:, None], confound @ b_train, confound @ b_test)
    resid = partial.partial_out_cv(target, confound, train)
    assert np.allclose(resid[train], 0.0, atol=1e-8)        # train map removed
    assert np.linalg.norm(resid[~train]) > 1.0              # test map NOT removed


def test_partial_out_cv_all_train_equals_partial_out():
    rng = np.random.default_rng(12)
    confound = rng.standard_normal((150, 2))
    target = confound @ rng.standard_normal((2, 3)) + rng.standard_normal((150, 3))
    allrows = np.ones(150, dtype=bool)
    assert np.allclose(partial.partial_out_cv(target, confound, allrows),
                       partial.partial_out(target, confound))


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


def test_partial_out_cv_has_an_intercept():
    """Ported from tom_cca 9e03883. A constant offset in target and a nonzero-mean
    confound: the train residual must be mean-zero and orthogonal to the confound.
    Without an intercept the residual keeps a multiple of (I - P_Z)*1 -- the
    sub-window trap, and this port's epoch windows are exactly that."""
    rng = np.random.default_rng(21)
    n = 500
    confound = rng.standard_normal((n, 3)) + np.array([2.0, -1.0, 0.5])
    target = 5.0 + confound @ rng.standard_normal((3, 2)) + 0.1 * rng.standard_normal((n, 2))
    train = np.zeros(n, dtype=bool)
    train[:350] = True
    resid = partial.partial_out_cv(target, confound, train)
    assert np.allclose(resid[train].mean(axis=0), 0.0, atol=1e-10)
    zc = confound[train] - confound[train].mean(axis=0)
    assert np.allclose(zc.T @ resid[train], 0.0, atol=1e-8)
    tc = target[train] - target[train].mean(axis=0)
    ref = tc - zc @ np.linalg.lstsq(zc, tc, rcond=None)[0]
    assert np.allclose(resid[train], ref, atol=1e-8)


def test_partial_out_cv_intercept_is_train_only():
    """The intercept is estimated on the train rows and applied to held-out rows, so a
    held-out block with a DIFFERENT offset keeps that difference."""
    rng = np.random.default_rng(22)
    n = 400
    confound = rng.standard_normal((n, 2))
    target = confound @ rng.standard_normal((2, 2)) + 0.05 * rng.standard_normal((n, 2))
    train = np.zeros(n, dtype=bool)
    train[:300] = True
    target[~train] += 3.0                       # held-out rows sit 3 higher
    resid = partial.partial_out_cv(target, confound, train)
    assert np.allclose(resid[train].mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(resid[~train].mean(axis=0), 3.0, atol=0.05)


def test_the_intercept_free_form_creates_a_spurious_shared_channel():
    """The trap itself, pinned. X and Y are INDEPENDENT quiet units on a sub-window
    with nonzero column means; without an intercept both residuals keep the same
    (I - P_Z)*1 image and CCA reads a channel that is not there."""
    rng = np.random.default_rng(7)
    n, kx, ky, kz = 4000, 12, 12, 6
    Z = rng.standard_normal((n, kz)) + rng.standard_normal(kz) * 2.0
    X = 0.05 * rng.standard_normal((n, kx)) + rng.standard_normal(kx) * 2.0
    Y = 0.05 * rng.standard_normal((n, ky)) + rng.standard_normal(ky) * 2.0
    train = np.zeros(n, dtype=bool)
    train[: n // 2] = True

    design = np.column_stack([Z, np.ones(n)])
    def resid(M, D):
        coef, *_ = np.linalg.lstsq(D[train], M[train], rcond=None)
        return M - D @ coef
    def cc1(A, B):
        A = A - A.mean(0); B = B - B.mean(0)
        qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
        return float(np.linalg.svd(qa.T @ qb, compute_uv=False)[0])

    bad = cc1(resid(X, Z)[~train], resid(Y, Z)[~train])
    good = cc1(resid(X, design)[~train], resid(Y, design)[~train])
    assert bad > 0.9, f"the trap must reproduce; got cc1 = {bad:.3f}"
    assert good < bad - 0.3, (
        f"the intercept must remove it; {bad:.3f} -> {good:.3f}")
