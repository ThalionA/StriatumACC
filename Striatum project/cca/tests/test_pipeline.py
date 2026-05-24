"""End-to-end tests for the per-(animal, pair) pipeline, on synthetic animals."""

from __future__ import annotations

import dataclasses

import numpy as np

from striatum_cca import config, core, dataio, pipeline

CFG = config.DEFAULT


def synthetic_animal(
    units_per_area: dict[str, int],
    *,
    shared_strength: float,
    noise: float,
    n_trials: int = 120,
    n_bins: int = config.N_BINS,
    n_shared: int = 3,
    seed: int = 0,
) -> dataio.Animal:
    """An animal whose area residuals share ``n_shared`` latent dimensions.

    Each unit = a fixed spatial tuning profile (removed by residualisation)
    plus ``shared_strength`` x (shared latents) plus independent noise. With
    ``shared_strength > 0`` the areas have genuine trial-to-trial communication;
    with ``shared_strength == 0`` they do not.
    """
    rng = np.random.default_rng(seed)
    n_units = sum(units_per_area.values())

    profile = rng.standard_normal((1, n_bins, n_units)) * 2.0
    activity = np.repeat(profile, n_trials, axis=0)

    # One shared latent set, used by every area -> genuine communication.
    shared_latents = rng.standard_normal((n_trials, n_bins, n_shared))

    masks = {a: np.zeros(n_units, dtype=bool) for a in config.AREAS}
    start = 0
    for area in config.AREAS:
        count = units_per_area.get(area, 0)
        if count and shared_strength > 0:
            loadings = rng.standard_normal((n_shared, count))
            activity[:, :, start : start + count] += shared_strength * (
                shared_latents @ loadings
            )
        masks[area][start : start + count] = True
        start += count
    activity += noise * rng.standard_normal((n_trials, n_bins, n_units))

    return dataio.Animal(
        animal_id=1,
        spatial_fr=activity,
        neurontypes=np.full((n_units, 5), 5.0),
        area_masks=masks,
        change_point=np.nan,
        zscored_lick_errors=np.zeros(n_trials),
        n_trials=n_trials,
    )


LEARNER = dataio.CohortEntry(animal_id=1, role="learner", lp=42, raw_lp=42)


def test_fit_pair_recovers_shared_communication():
    animal = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0, noise=0.3)
    fit = pipeline.fit_pair(animal, "DMS", "ACC", LEARNER, CFG)
    assert isinstance(fit, pipeline.PairFit)
    assert set(fit.epochs) == set(config.EPOCH_NAMES)
    for epoch in config.EPOCH_NAMES:
        # Genuine shared latents -> held-out CC1 generalises high.
        assert fit.epochs[epoch].held_out_r[0] > 0.6


def test_fit_pair_no_communication_stays_near_zero():
    animal = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=0.0, noise=1.0)
    fit = pipeline.fit_pair(animal, "DMS", "ACC", LEARNER, CFG)
    assert isinstance(fit, pipeline.PairFit)
    for epoch in config.EPOCH_NAMES:
        # No shared latent -> held-out CC1 ~ 0 even though in-sample is biased.
        assert abs(fit.epochs[epoch].held_out_r[0]) < 0.3
        assert fit.epochs[epoch].in_sample_r[0] > fit.epochs[epoch].held_out_r[0]


def test_fit_pair_skips_area_with_too_few_units():
    animal = synthetic_animal(
        {"DMS": 25, "ACC": 25, "DLS": 3}, shared_strength=1.0, noise=0.3
    )
    result = pipeline.fit_pair(animal, "DMS", "DLS", LEARNER, CFG)
    assert isinstance(result, pipeline.SkippedPair)
    assert "too few units" in result.reason


def test_fit_pair_skips_when_epochs_invalid():
    animal = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0, noise=0.3)
    too_early = dataio.CohortEntry(animal_id=1, role="learner", lp=5, raw_lp=5)
    result = pipeline.fit_pair(animal, "DMS", "ACC", too_early, CFG)
    assert isinstance(result, pipeline.SkippedPair)
    assert "no valid epochs" in result.reason


def test_fit_pair_k_is_fixed_across_epochs():
    animal = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0, noise=0.3)
    fit = pipeline.fit_pair(animal, "DMS", "ACC", LEARNER, CFG)
    assert isinstance(fit, pipeline.PairFit)
    ks = {cv.k for cv in fit.epochs.values()}
    assert ks == {fit.k}


def test_fit_pair_is_bin_count_agnostic():
    # The pipeline auto-detects n_bins; 100-bin (2.5 cm) data must work too.
    animal = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0,
                              noise=0.3, n_bins=100)
    assert animal.n_bins == 100
    fit = pipeline.fit_pair(animal, "DMS", "ACC", LEARNER, CFG)
    assert isinstance(fit, pipeline.PairFit)
    for epoch in config.EPOCH_NAMES:
        assert fit.epochs[epoch].n_samples == CFG.trials_per_epoch * 100
        assert fit.epochs[epoch].held_out_r[0] > 0.6


def test_zscore_units_flag_runs_and_recovers_communication():
    # With zscore_units on, the pipeline still runs and recovers the shared
    # latent (z-scoring only re-weights units for the PCA step).
    animal = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0, noise=0.3)
    zcfg = dataclasses.replace(CFG, zscore_units=True)
    fit = pipeline.fit_pair(animal, "DMS", "ACC", LEARNER, zcfg)
    assert isinstance(fit, pipeline.PairFit)
    for epoch in config.EPOCH_NAMES:
        assert fit.epochs[epoch].held_out_r[0] > 0.6


def test_zscore_invariant_to_global_unit_rescaling():
    # Whole-unit z-scoring removes each unit's scale, so multiplying one unit's
    # activity by a constant across *all* trials leaves the fit unchanged.
    base = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0, noise=0.3)
    scaled = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0, noise=0.3)
    scaled.spatial_fr[:, :, 0] *= 100.0
    # k_fixed < n_units so PCA genuinely reduces -- otherwise (k == n_units)
    # a full-rank PCA makes the result invariant to z-scoring trivially.
    zcfg = dataclasses.replace(CFG, zscore_units=True, k_mode="fixed",
                               k_fixed=10)
    fit_base = pipeline.fit_pair(base, "DMS", "ACC", LEARNER, zcfg)
    fit_scaled = pipeline.fit_pair(scaled, "DMS", "ACC", LEARNER, zcfg)
    for epoch in config.EPOCH_NAMES:
        assert np.allclose(fit_base.epochs[epoch].held_out_r,
                           fit_scaled.epochs[epoch].held_out_r, atol=1e-8)


def test_prepare_pair_partial_removes_third_area_mediated_coupling():
    # All three areas share the SAME latents, so DMS-ACC coupling runs through
    # DLS. Partialling DLS out should substantially reduce the DMS-ACC
    # canonical correlation (not to zero -- DLS is only a noisy proxy for the
    # latent, so some shared variance survives; that is correct partial-
    # correlation behaviour).
    animal = synthetic_animal({"DMS": 25, "ACC": 25, "DLS": 25},
                              shared_strength=1.0, noise=0.3)
    plain = pipeline.prepare_pair(animal, "DMS", "ACC", LEARNER, CFG)
    part = pipeline.prepare_pair_partial(animal, "DMS", "ACC", LEARNER, CFG)
    assert isinstance(plain, pipeline.PreparedPair)
    assert isinstance(part, pipeline.PreparedPair)
    for epoch in config.EPOCH_NAMES:
        plain_cc = core.cca_cv(plain.scores_x[epoch],
                               plain.scores_y[epoch], CFG).held_out_r[0]
        part_cc = core.cca_cv(part.scores_x[epoch],
                              part.scores_y[epoch], CFG).held_out_r[0]
        assert plain_cc > 0.6                   # genuine coupling
        assert plain_cc - part_cc > 0.2         # ... substantially via DLS


def test_prepare_pair_partial_skips_when_no_other_area():
    animal = synthetic_animal({"DMS": 25, "ACC": 25},
                              shared_strength=1.0, noise=0.3)
    result = pipeline.prepare_pair_partial(animal, "DMS", "ACC", LEARNER, CFG)
    assert isinstance(result, pipeline.SkippedPair)
    assert "other area" in result.reason


def test_prepare_pair_partial_returns_a_valid_prepared_pair():
    animal = synthetic_animal({"DMS": 25, "ACC": 25, "DLS": 25},
                              shared_strength=1.0, noise=0.3)
    part = pipeline.prepare_pair_partial(animal, "DMS", "ACC", LEARNER, CFG)
    assert isinstance(part, pipeline.PreparedPair)
    assert set(part.scores_x) == set(config.EPOCH_NAMES)
    for epoch in config.EPOCH_NAMES:
        assert part.scores_x[epoch].shape[2] == part.k
        assert part.scores_y[epoch].shape[2] == part.k


def test_zscore_normalisation_spans_whole_engaged_period():
    # Z-scoring is computed over the entire engaged period before epoch
    # slicing. Scaling one unit in the *expert* trials only therefore shifts
    # that unit's global std and so perturbs even the *naive* epoch's fit --
    # which the old per-epoch z-scoring would have left bit-identical.
    base = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0, noise=0.3)
    expert_only = synthetic_animal({"DMS": 25, "ACC": 25},
                                   shared_strength=1.0, noise=0.3)
    expert = np.arange(LEARNER.lp, LEARNER.lp + CFG.trials_per_epoch)
    expert_only.spatial_fr[expert, :, 0] *= 50.0
    # k_fixed < n_units so PCA genuinely reduces (z-scoring is a no-op when
    # k == n_units: a full-rank PCA + scale-invariant CCA).
    zcfg = dataclasses.replace(CFG, zscore_units=True, k_mode="fixed",
                               k_fixed=10)
    fit_base = pipeline.fit_pair(base, "DMS", "ACC", LEARNER, zcfg)
    fit_expert = pipeline.fit_pair(expert_only, "DMS", "ACC", LEARNER, zcfg)
    assert not np.allclose(fit_base.epochs["naive"].held_out_r,
                           fit_expert.epochs["naive"].held_out_r, atol=1e-6)
