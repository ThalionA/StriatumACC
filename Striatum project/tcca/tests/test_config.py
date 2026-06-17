"""Config sanity: areas, pairs, mappings, and report-faithful defaults."""

from __future__ import annotations

from striatum_tcca import config


def test_pairs_reference_known_areas():
    for x, y in config.PAIRS:
        assert x in config.AREAS, x
        assert y in config.AREAS, y
        assert x != y


def test_area_field_and_colours_cover_areas():
    assert set(config.AREA_FIELD) == set(config.AREAS)
    assert set(config.AREA_COLOURS) == set(config.AREAS)


def test_epoch_metadata():
    assert config.EPOCH_NAMES == ("naive", "intermediate", "expert")
    assert set(config.EPOCH_COLOURS) == set(config.EPOCH_NAMES)
    assert set(config.ENGAGEMENT_COLOURS) == {"engaged", "disengaged"}


def test_report_faithful_defaults():
    cfg = config.DEFAULT
    # Report Section 2.2: smooth sigma 2.5 ms, 10 ms primary bin.
    assert cfg.gaussian_sd_ms == 2.5
    assert cfg.temporal_bin_ms == 10
    # Headline IFI window +/-50 ms.
    assert config.lag_window_ms(cfg) == 50
    # Residual + partial pipeline, z-scored, FS-excluded, circshift null.
    assert cfg.subtract_trial_mean and cfg.zscore_units and cfg.exclude_fast_spiking
    assert cfg.null_type == "circshift"
    assert cfg.velocity_thresh_cm_s == 2.0


def test_data_paths_by_group():
    assert set(config.DATA_BY_GROUP) == {"task", "control"}
    assert config.DATA_BY_GROUP["task"].name.endswith(".mat")
