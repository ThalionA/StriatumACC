"""Ground-truth tests for the information-theoretic estimators.

These mirror Lemke, Celotto, Maffulli, Ganguly & Panzeri (2024), *Information
flow between motor cortex and striatum reverses during skill learning*, Curr Biol
34:1831-1843 — mutual information about a behavioural feature, a partial
information decomposition of what two areas carry, and a directed measure
(transfer entropy, and the feature-specific transfer built on top of it).

Every estimator is pinned on a case whose answer is known analytically:

* **XOR** — each source alone carries nothing, the pair carries one bit. All of
  it is synergy.
* **COPY** — one source IS the target, the other is independent. All of it is
  unique to the first.
* **COMMON** — both sources are the target. All of it is redundant.
* **Independence** — zero, and specifically zero AFTER bias handling, because the
  plug-in estimate of mutual information is biased upward and is never zero on
  finite samples.

The bias point is the one that matters most in practice: with 5 bins for the
neural variable and 3 for the feature (the paper's scheme), a plug-in MI on a
few hundred trials is positive for independent variables. Lemke et al. subtract
the mean of a shuffled distribution; the existing MATLAB arm in this repo uses a
Miller-Madow analytic correction AND a shuffle null. Both are implemented so the
two can be compared rather than argued about.

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np
import pytest

from striatum_info import estimators as est


# --------------------------------------------------------------------------
# discretisation
# --------------------------------------------------------------------------

def test_equipopulated_bins_are_equally_occupied():
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=9_000)                 # strongly skewed on purpose
    b = est.equipopulated_bins(x, 3)
    counts = np.bincount(b, minlength=3)
    assert set(np.unique(b)) == {0, 1, 2}
    assert counts.max() - counts.min() <= 1, f"bins should be balanced, got {counts}"


def test_equipopulated_bins_are_monotone_in_the_value():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5_000)
    b = est.equipopulated_bins(x, 5)
    for lo in range(4):
        assert x[b == lo].max() <= x[b == lo + 1].min() + 1e-9


def test_zero_aware_bins_keep_zeros_together():
    """The repo's existing MATLAB convention: exact zeros get their own bin.

    Spike counts and lick rates are sparse; an equipopulated split would scatter
    the zeros across bins and destroy the distinction that carries the signal.
    """
    rng = np.random.default_rng(2)
    x = np.concatenate([np.zeros(500), rng.lognormal(size=500)])
    b = est.zero_aware_bins(x, 3)
    assert len(set(b[:500])) == 1, "every exact zero belongs in one bin"
    assert b[500:].min() != b[0], "non-zeros must not share the zero bin"


# --------------------------------------------------------------------------
# mutual information
# --------------------------------------------------------------------------

def test_mutual_information_of_a_deterministic_copy_is_the_entropy():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 4, size=20_000)
    assert est.mutual_information(x, x) == pytest.approx(2.0, abs=0.02)


def test_mutual_information_of_independent_variables_is_zero_in_the_limit():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 3, size=200_000)
    y = rng.integers(0, 5, size=200_000)
    assert est.mutual_information(x, y) == pytest.approx(0.0, abs=0.005)


def test_plugin_mutual_information_is_biased_up_on_small_samples():
    """The reason a bias correction is not optional.

    Independent variables, 5 x 3 bins, 200 trials -- the paper's scheme and a
    plausible trial count. The plug-in estimate must come out clearly positive,
    which is what makes an uncorrected 'information' value meaningless.
    """
    rng = np.random.default_rng(5)
    vals = [est.mutual_information(rng.integers(0, 5, 200), rng.integers(0, 3, 200))
            for _ in range(200)]
    assert np.mean(vals) > 0.02, (
        f"plug-in MI on independent data should be visibly positive, got {np.mean(vals):.4f}")


def test_shuffle_subtraction_removes_that_bias():
    rng = np.random.default_rng(6)
    vals = [est.shuffle_subtracted_mi(rng.integers(0, 5, 200), rng.integers(0, 3, 200),
                                      n_shuffles=50, seed=k)["mi_corrected"]
            for k in range(60)]
    assert abs(np.mean(vals)) < 0.01, (
        f"after shuffle subtraction independent data should sit at zero, "
        f"got {np.mean(vals):+.4f}")


def test_shuffle_subtraction_keeps_a_real_dependence():
    rng = np.random.default_rng(7)
    x = rng.integers(0, 3, size=600)
    y = (x + (rng.random(600) < 0.1).astype(int)) % 3      # noisy copy
    out = est.shuffle_subtracted_mi(x, y, n_shuffles=50, seed=0)
    assert out["mi_corrected"] > 0.5
    assert out["p"] < 0.05


def test_miller_madow_also_reduces_the_bias():
    """The repo's existing MATLAB correction, for comparison with the paper's."""
    rng = np.random.default_rng(8)
    plug = [est.mutual_information(rng.integers(0, 5, 200), rng.integers(0, 3, 200))
            for _ in range(150)]
    mm = [est.mutual_information(rng.integers(0, 5, 200), rng.integers(0, 3, 200),
                                 correction="miller_madow") for _ in range(150)]
    assert abs(np.mean(mm)) < abs(np.mean(plug))


# --------------------------------------------------------------------------
# partial information decomposition
# --------------------------------------------------------------------------

def test_xor_is_pure_synergy():
    rng = np.random.default_rng(9)
    x = rng.integers(0, 2, size=60_000)
    y = rng.integers(0, 2, size=60_000)
    s = x ^ y
    pid = est.pid_imin(x, y, s)
    assert est.mutual_information(x, s) == pytest.approx(0.0, abs=0.01)
    assert est.mutual_information(y, s) == pytest.approx(0.0, abs=0.01)
    assert pid["synergy"] == pytest.approx(1.0, abs=0.05)
    assert pid["redundancy"] == pytest.approx(0.0, abs=0.02)
    assert pid["unique_x"] == pytest.approx(0.0, abs=0.02)


def test_a_copied_source_carries_unique_information():
    rng = np.random.default_rng(10)
    x = rng.integers(0, 2, size=60_000)
    y = rng.integers(0, 2, size=60_000)
    s = x                                   # target IS x; y is irrelevant
    pid = est.pid_imin(x, y, s)
    assert pid["unique_x"] == pytest.approx(1.0, abs=0.05)
    assert pid["unique_y"] == pytest.approx(0.0, abs=0.02)
    assert pid["redundancy"] == pytest.approx(0.0, abs=0.02)


def test_two_copies_of_the_target_are_pure_redundancy():
    rng = np.random.default_rng(11)
    s = rng.integers(0, 2, size=60_000)
    pid = est.pid_imin(s, s, s)
    assert pid["redundancy"] == pytest.approx(1.0, abs=0.05)
    assert pid["synergy"] == pytest.approx(0.0, abs=0.02)
    assert pid["unique_x"] == pytest.approx(0.0, abs=0.02)


def test_pid_atoms_sum_to_the_joint_information():
    rng = np.random.default_rng(12)
    x = rng.integers(0, 3, size=40_000)
    y = rng.integers(0, 3, size=40_000)
    s = (x + y) % 3
    pid = est.pid_imin(x, y, s)
    total = pid["redundancy"] + pid["unique_x"] + pid["unique_y"] + pid["synergy"]
    joint = est.mutual_information(est.pair_code(x, y), s)
    assert total == pytest.approx(joint, rel=0.02), "the four atoms must partition I(X,Y;S)"


# --------------------------------------------------------------------------
# directed measures
# --------------------------------------------------------------------------

def test_transfer_entropy_finds_the_driver():
    """X drives Y one step later; TE must be asymmetric in the right direction."""
    rng = np.random.default_rng(13)
    n = 60_000
    x = rng.integers(0, 2, size=n)
    y = np.zeros(n, dtype=int)
    y[1:] = np.where(rng.random(n - 1) < 0.85, x[:-1], rng.integers(0, 2, n - 1))
    te_xy = est.transfer_entropy(x, y, lag=1)
    te_yx = est.transfer_entropy(y, x, lag=1)
    assert te_xy > 0.2, f"X->Y should be clear, got {te_xy:.3f}"
    assert te_xy > 5 * max(te_yx, 1e-6), f"and much larger than Y->X ({te_yx:.3f})"


def test_transfer_entropy_is_zero_for_independent_series():
    rng = np.random.default_rng(14)
    n = 60_000
    x = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    assert est.transfer_entropy(x, y, lag=1) == pytest.approx(0.0, abs=0.01)


def test_transfer_entropy_discounts_the_receivers_own_past():
    """A self-predictive Y that X merely echoes must NOT look like X -> Y.

    This is the whole point of conditioning on Y's past, and the case that
    separates transfer entropy from a lagged correlation.
    """
    rng = np.random.default_rng(15)
    n = 80_000
    y = np.zeros(n, dtype=int)
    for t in range(1, n):
        y[t] = y[t - 1] if rng.random() < 0.9 else 1 - y[t - 1]
    x = np.roll(y, 1)                      # x is just y delayed: adds nothing new
    te = est.transfer_entropy(x, y, lag=1)
    assert te < 0.05, f"an echo of Y's own past is not transfer, got {te:.3f}"


def test_conditional_mutual_information_matches_mi_when_z_is_constant():
    rng = np.random.default_rng(16)
    x = rng.integers(0, 3, size=30_000)
    y = (x + (rng.random(30_000) < 0.2).astype(int)) % 3
    z = np.zeros_like(x)
    assert est.conditional_mutual_information(x, y, z) == pytest.approx(
        est.mutual_information(x, y), rel=0.02)


# --------------------------------------------------------------------------
# the vectorised path
# --------------------------------------------------------------------------

def test_vectorised_mi_matches_the_scalar_estimator():
    """The fast path exists only for speed; it must give the same numbers."""
    rng = np.random.default_rng(20)
    n_samples = 800
    spikes = (rng.random((6, n_samples)) < 0.3).astype(int)
    # Three feature variants, one of them genuinely coupled to unit 0.
    codes = rng.integers(0, 3, size=(n_samples, 3))
    codes[:, 0] = np.where(spikes[0] == 1, 0, rng.integers(1, 3, n_samples))

    fast = est.mi_binary_vs_categorical(spikes, codes, n_codes=3)
    for u in range(spikes.shape[0]):
        for v in range(codes.shape[1]):
            slow = est.mutual_information(spikes[u], codes[:, v])
            assert fast[u, v] == pytest.approx(slow, abs=1e-12), (
                f"unit {u}, variant {v}: fast {fast[u, v]:.6f} vs slow {slow:.6f}")


def test_vectorised_mi_handles_a_silent_unit():
    """A unit that never fires carries no information and must not produce NaN."""
    rng = np.random.default_rng(21)
    spikes = np.zeros((2, 500), dtype=int)
    spikes[1] = (rng.random(500) < 0.5).astype(int)
    codes = rng.integers(0, 3, size=(500, 1))
    out = est.mi_binary_vs_categorical(spikes, codes, n_codes=3)
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_vectorised_mi_shapes_follow_the_inputs():
    rng = np.random.default_rng(22)
    spikes = (rng.random((4, 300)) < 0.4).astype(int)
    assert est.mi_binary_vs_categorical(spikes, rng.integers(0, 3, 300), 3).shape == (4, 1)
    assert est.mi_binary_vs_categorical(spikes, rng.integers(0, 3, (300, 7)), 3).shape == (4, 7)


# --------------------------------------------------------------------------
# which summary statistic may be compared across epochs
# --------------------------------------------------------------------------

def _window_summaries(n_trials, *, n_units=40, n_time=150, pool=5, shift=5,
                      n_shuffles=30, n_bins=2, seed=0):
    """Peak and mean over time windows, on data with NO information at all.

    Mirrors the driver in ``scripts/run_mi.py``: binary spikes, a feature split
    into ``n_bins``, responses pooled over non-overlapping windows, each window
    shuffle-subtracted.
    """
    rng = np.random.default_rng(seed)
    spikes = (rng.random((n_units, n_time, n_trials)) < 0.045).astype(int)
    codes = est.equipopulated_bins(rng.normal(size=n_trials), n_bins)
    variants = np.empty((n_trials, 1 + n_shuffles), int)
    variants[:, 0] = codes
    for s in range(n_shuffles):
        variants[:, s + 1] = rng.permutation(codes)
    rep = np.tile(variants, (pool, 1))
    starts = np.arange(0, n_time - pool + 1, shift)
    mi = np.empty((n_units, starts.size, 1 + n_shuffles))
    for wi, w in enumerate(starts):
        blk = spikes[:, w:w + pool, :]
        mi[:, wi, :] = est.mi_binary_vs_categorical(
            blk.reshape(n_units, -1), rep, n_bins)
    corrected = mi[:, :, 0] - mi[:, :, 1:].mean(axis=2)
    return corrected.max(axis=1).mean(), corrected.mean(axis=1).mean()


def test_the_peak_over_windows_is_biased_even_after_shuffle_subtraction():
    """Shuffle subtraction fixes each window; taking the max re-breaks it.

    The bug this pins (2026-09-17): ``peak_mi_corrected`` was used as the effect
    size for Naive-vs-Expert contrasts. On data carrying zero information it
    comes back at roughly +0.05 bits -- ten times the size of the "effects" that
    were being reported off it.
    """
    peak, mean = _window_summaries(10, seed=1)
    assert peak > 0.02, (
        f"the max over windows must show its selection bias, got {peak:.5f}")
    assert abs(mean) < 0.002, (
        f"the mean over windows must be unbiased, got {mean:+.5f}")


def test_the_peaks_bias_depends_on_sample_size_and_the_means_does_not():
    """Why the peak cannot be compared between epochs of unequal n.

    ``lick_error_z`` is NaN on trial 1, and trial 1 lives in the Naive epoch, so
    Naive had 8.85 usable trials against Expert's 9.81. That one trial alone
    produced a spurious negative contrast through the peak.
    """
    peak_9, mean_9 = _window_summaries(9, seed=2)
    peak_10, mean_10 = _window_summaries(10, seed=2)
    assert peak_9 - peak_10 > 0.001, (
        "the smaller sample must carry the larger peak bias; got "
        f"{peak_9:.5f} at n=9 vs {peak_10:.5f} at n=10")
    assert abs(mean_9 - mean_10) < 0.0005, (
        f"the mean must not care about one trial; got {mean_9:+.5f} vs {mean_10:+.5f}")


# --------------------------------------------------------------------------
# the general vectorised path, used by the LFP arm
# --------------------------------------------------------------------------

def test_vectorised_multilevel_mi_matches_the_scalar_estimator():
    rng = np.random.default_rng(30)
    n = 900
    x = rng.integers(0, 3, size=n)
    variants = rng.integers(0, 2, size=(n, 5))
    variants[:, 0] = (x > 0).astype(int)          # one genuinely coupled column
    fast = est.mi_codes_vs_variants(x, variants, n_x=3, n_y=2)
    for v in range(variants.shape[1]):
        slow = est.mutual_information(x, variants[:, v])
        assert fast[v] == pytest.approx(slow, abs=1e-12)


def test_conditional_vectorised_mi_matches_the_scalar_estimator():
    rng = np.random.default_rng(31)
    n = 1_200
    z = rng.integers(0, 2, size=n)
    x = rng.integers(0, 3, size=n)
    variants = np.stack([(x + z) % 2, rng.integers(0, 2, n)], axis=1)
    fast = est.cmi_codes_vs_variants(x, variants, z, n_x=3, n_y=2)
    for v in range(variants.shape[1]):
        slow = est.conditional_mutual_information(x, variants[:, v], z)
        assert fast[v] == pytest.approx(slow, abs=1e-9)


def test_conditioning_on_the_confound_removes_a_spurious_dependence():
    """The case the LFP arm exists to catch.

    Speed drives BOTH band power and the behavioural feature. The raw mutual
    information is then clearly positive while the conditional information is
    zero -- the whole apparent coupling was the confound.
    """
    rng = np.random.default_rng(32)
    n = 40_000
    speed = rng.integers(0, 2, size=n)
    power = (speed + (rng.random(n) < 0.15).astype(int)) % 2
    feature = (speed + (rng.random(n) < 0.15).astype(int)) % 2
    raw = est.mi_codes_vs_variants(power, feature, n_x=2, n_y=2)[0]
    cond = est.cmi_codes_vs_variants(power, feature, speed, n_x=2, n_y=2)[0]
    assert raw > 0.1, f"the confound must create apparent coupling, got {raw:.4f}"
    assert cond < 0.01, f"conditioning must remove it, got {cond:.4f}"


# --------------------------------------------------------------------------
# tie handling — the split must encode the feature, not the array order
# --------------------------------------------------------------------------

def test_equipopulated_bins_splits_ties_by_position():
    """The defect this guards against, pinned so it cannot be forgotten.

    A near-constant feature still returns a perfectly balanced split, and among
    the tied samples the bin tracks the INDEX — so for a trial-ordered array the
    "feature" becomes trial order.
    """
    v = np.ones(20)
    v[[3, 11]] = 0.0
    c = est.equipopulated_bins(v, 2)
    assert np.bincount(c).min() >= 9, "it still looks balanced, which is the trap"
    tied = c[v == 1]
    assert np.array_equal(tied, np.sort(tied)), "tied samples are split by position"


def test_value_boundary_split_refuses_a_near_constant_feature():
    v = np.ones(200)
    v[:4] = 0.0                       # 2% zeros — honest split exists but is useless
    assert est.value_boundary_split(v) is None


def test_value_boundary_split_cuts_only_at_a_real_value_change():
    rng = np.random.default_rng(40)
    v = rng.integers(0, 5, size=400).astype(float)     # ties, but well spread
    c = est.value_boundary_split(v)
    assert c is not None
    assert v[c == 0].max() < v[c == 1].min(), "no tied value may straddle the cut"


def test_value_boundary_split_matches_the_median_on_continuous_data():
    rng = np.random.default_rng(41)
    v = rng.normal(size=500)
    c = est.value_boundary_split(v)
    assert c is not None
    assert abs(c.mean() - 0.5) < 0.01
    assert np.array_equal(c, est.equipopulated_bins(v, 2)), (
        "with no ties it must agree with the equipopulated split")


def test_value_boundary_split_refuses_a_constant_feature():
    assert est.value_boundary_split(np.full(50, 3.0)) is None
