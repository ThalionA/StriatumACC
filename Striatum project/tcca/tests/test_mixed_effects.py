"""Tests for tom_cca.mixed_effects.

A linear mixed model with a per-animal random intercept is the correct way to
test the epoch effect on landmark-level data: it accounts for the within-animal
correlation across landmarks instead of treating each (animal, landmark) as an
independent sample (the pseudoreplication that inflates the pooled Wilcoxon).
"""

from __future__ import annotations

import numpy as np

from striatum_tcca import mixed_effects


def _make_records(rng, n_animals, n_landmarks, epoch_effect,
                  animal_sd=0.5, noise_sd=0.05, landmark_sd=0.1):
    """Synthetic landmark-level records with a per-animal random intercept.

    ``epoch_effect`` maps epoch -> true mean shift. The animal baseline is the
    big variance component; landmarks add a fixed per-landmark offset and each
    observation gets small noise.
    """
    animal_base = rng.normal(0, animal_sd, n_animals)
    landmark_off = rng.normal(0, landmark_sd, n_landmarks)
    recs = []
    for a in range(n_animals):
        for lm in range(n_landmarks):
            for epoch, eff in epoch_effect.items():
                val = (animal_base[a] + landmark_off[lm] + eff
                       + rng.normal(0, noise_sd))
                recs.append({"animal_id": a, "landmark": lm,
                             "epoch": epoch, "value": val})
    return recs


def test_detects_real_per_animal_effect():
    rng = np.random.default_rng(0)
    recs = _make_records(rng, n_animals=8, n_landmarks=6,
                         epoch_effect={"naive": 0.0, "intermediate": 0.15,
                                       "expert": 0.30})
    out = mixed_effects.lmm_epoch_contrasts(recs)
    en = out["expert-naive"]
    assert en["ok"]
    assert en["n_animals"] == 8
    assert abs(en["estimate"] - 0.30) < 0.08      # recovers the true shift
    assert en["p"] < 0.05                          # and calls it significant


def test_null_effect_not_significant():
    rng = np.random.default_rng(1)
    recs = _make_records(rng, n_animals=8, n_landmarks=6,
                         epoch_effect={"naive": 0.0, "intermediate": 0.0,
                                       "expert": 0.0})
    out = mixed_effects.lmm_epoch_contrasts(recs)
    assert out["expert-naive"]["ok"]
    assert out["expert-naive"]["p"] > 0.05


def test_zero_mean_with_per_animal_slopes_not_significant():
    # Each animal has its OWN learning slope, shared across that animal's 6
    # landmarks, but the slopes average to ~zero across animals. A pooled test
    # that treats all 48 (animal, landmark) deltas as independent is tempted to
    # call it significant; the LMM tests the effect against BETWEEN-animal
    # variation (n = animals), so a zero mean stays non-significant.
    rng = np.random.default_rng(2)
    animal_slope = rng.normal(0.0, 0.3, 8)            # mean ~0, big spread
    animal_base = rng.normal(0, 0.5, 8)
    recs = []
    for a in range(8):
        for lm in range(6):
            for epoch in ("naive", "expert"):
                eff = animal_slope[a] if epoch == "expert" else 0.0
                recs.append({"animal_id": a, "landmark": lm, "epoch": epoch,
                             "value": animal_base[a] + eff
                             + rng.normal(0, 0.02)})
    out = mixed_effects.lmm_epoch_contrasts(
        recs, contrasts=(("expert", "naive"),))
    assert out["expert-naive"]["ok"]
    assert out["expert-naive"]["p"] > 0.05


def test_no_landmark_term_when_absent():
    # Arm A has no landmark dimension: one value per (animal, epoch).
    rng = np.random.default_rng(3)
    recs = []
    base = rng.normal(0, 0.5, 10)
    for a in range(10):
        for epoch, eff in {"naive": 0.0, "expert": 0.4}.items():
            recs.append({"animal_id": a, "epoch": epoch,
                         "value": base[a] + eff + rng.normal(0, 0.05)})
    out = mixed_effects.lmm_epoch_contrasts(recs, landmark=None,
                                            contrasts=(("expert", "naive"),))
    assert out["expert-naive"]["ok"]
    assert out["expert-naive"]["p"] < 0.05


def test_insufficient_animals_returns_not_ok():
    recs = [{"animal_id": 0, "landmark": 0, "epoch": e, "value": v}
            for e, v in (("naive", 0.1), ("expert", 0.5))]
    out = mixed_effects.lmm_epoch_contrasts(recs)
    assert not out["expert-naive"]["ok"]
    assert np.isnan(out["expert-naive"]["p"])


def test_lmm_slope_detects_population_trend():
    # Each animal has windows along an axis with a positive mean slope.
    rng = np.random.default_rng(20)
    recs = []
    for a in range(8):
        base = rng.normal(0, 0.5)
        slope = 0.4 + rng.normal(0, 0.1)        # positive across animals
        for x in np.linspace(0, 1, 8):
            recs.append({"animal_id": a, "axis": float(x),
                         "value": base + slope * x + rng.normal(0, 0.05)})
    out = mixed_effects.lmm_slope(recs)
    assert out["ok"] and out["n_animals"] == 8
    assert out["estimate"] > 0.2 and out["p"] < 0.05


def test_lmm_slope_rejects_random_slope_below_four_animals():
    # A continuous random slope needs a 2x2 RE covariance (3 params) -> with only
    # 3 animals it is unidentifiable and the Wald p is anticonservative; the guard
    # must return ok=False / NaN p (mirrors lmm_epoch_contrasts) instead of fitting.
    rng = np.random.default_rng(22)
    recs = []
    for a in range(3):                              # only 3 animals
        for x in np.linspace(0, 1, 8):
            recs.append({"animal_id": a, "axis": float(x),
                         "value": 0.4 * x + rng.normal(0, 0.05)})
    out = mixed_effects.lmm_slope(recs)
    assert not out["ok"]
    assert np.isnan(out["p"])
    assert out["n_animals"] == 3
    assert out["reason"] == "too_few_animals_for_random_slope"


def test_lmm_slope_null_not_significant():
    rng = np.random.default_rng(21)
    recs = []
    for a in range(8):
        base = rng.normal(0, 0.5)
        slope = rng.normal(0, 0.3)              # mean ~0 across animals
        for x in np.linspace(0, 1, 8):
            recs.append({"animal_id": a, "axis": float(x),
                         "value": base + slope * x + rng.normal(0, 0.05)})
    out = mixed_effects.lmm_slope(recs)
    assert out["ok"]
    assert out["p"] > 0.05


def test_degenerate_fit_not_trusted():
    # Constant values -> zero residual variance -> the random-slope fit cannot
    # converge. The result must be ok=False with a NaN p (never a spurious tiny
    # p from a collapsed variance component), so small-N pairs don't masquerade
    # as significant.
    recs = [{"animal_id": a, "landmark": lm, "epoch": e, "value": 1.0}
            for a in range(4) for lm in range(6)
            for e in ("naive", "intermediate", "expert")]
    out = mixed_effects.lmm_epoch_contrasts(recs)
    assert not out["expert-naive"]["ok"]
    assert np.isnan(out["expert-naive"]["p"])


def test_all_three_contrasts_present():
    rng = np.random.default_rng(4)
    recs = _make_records(rng, n_animals=6, n_landmarks=6,
                         epoch_effect={"naive": 0.0, "intermediate": 0.1,
                                       "expert": 0.2})
    out = mixed_effects.lmm_epoch_contrasts(recs)
    assert set(out) == {"expert-naive", "expert-intermediate",
                        "intermediate-naive"}


# --------------------------------------------------------------------------
# collapsed_epoch_contrasts (robust small-N companion)
# --------------------------------------------------------------------------

def test_collapsed_recovers_per_animal_median():
    rng = np.random.default_rng(5)
    recs = _make_records(rng, n_animals=8, n_landmarks=6,
                         epoch_effect={"naive": 0.0, "intermediate": 0.1,
                                       "expert": 0.3})
    out = mixed_effects.collapsed_epoch_contrasts(recs)
    en = out["expert-naive"]
    assert en["ok"] and en["n_animals"] == 8
    assert abs(en["estimate"] - 0.30) < 0.08       # median per-animal delta


def test_collapsed_underpowered_at_n4():
    # 4 animals, all with a clear positive delta, still cannot beat the
    # signed-rank p-floor of 0.125 at n=4 -- the honest power limit.
    rng = np.random.default_rng(6)
    recs = _make_records(rng, n_animals=4, n_landmarks=6, noise_sd=0.01,
                         epoch_effect={"naive": 0.0, "expert": 0.3})
    out = mixed_effects.collapsed_epoch_contrasts(
        recs, contrasts=(("expert", "naive"),))
    en = out["expert-naive"]
    assert en["n_animals"] == 4
    assert en["estimate"] > 0.2                     # clearly positive...
    assert en["p"] >= 0.125                         # ...yet cannot reach sig
