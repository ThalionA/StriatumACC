"""End-to-end tests for the per-(animal, pair) pipeline, on synthetic animals."""

from __future__ import annotations

import numpy as np

from striatum_cca import config, dataio, pipeline

CFG = config.DEFAULT


def synthetic_animal(
    units_per_area: dict[str, int],
    *,
    shared_strength: float,
    noise: float,
    n_trials: int = 120,
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
    n_bins = config.N_BINS

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
    too_early = dataio.CohortEntry(animal_id=1, role="learner", lp=15, raw_lp=15)
    result = pipeline.fit_pair(animal, "DMS", "ACC", too_early, CFG)
    assert isinstance(result, pipeline.SkippedPair)
    assert "no valid epochs" in result.reason


def test_fit_pair_k_is_fixed_across_epochs():
    animal = synthetic_animal({"DMS": 25, "ACC": 25}, shared_strength=1.0, noise=0.3)
    fit = pipeline.fit_pair(animal, "DMS", "ACC", LEARNER, CFG)
    assert isinstance(fit, pipeline.PairFit)
    ks = {cv.k for cv in fit.epochs.values()}
    assert ks == {fit.k}
