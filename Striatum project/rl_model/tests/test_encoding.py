"""Pytest coverage for the per-neuron encoding analysis (`neural_encoding.py`).

Promotes the synthetic ground-truth gate that previously lived only inside
`scripts/run_neural_encoding.py` into the test suite, and checks the
`spatial_demean_var_ratio` diagnostic that flags near-stationary latents which
cannot be tested under the per-bin-demeaned ('beh_spatial') model.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.neural_encoding import encode_mouse, spatial_demean_var_ratio

ALPHA = 0.05


def _synthetic_fr():
    """Synthetic firing tensor with known per-neuron encoding + confounds."""
    rng = np.random.default_rng(0)
    T, B = 130, 50
    bins = np.arange(B)
    bump = np.exp(-0.5 * ((bins - 28) / 4.0) ** 2)
    ramp = 1.0 / (1.0 + np.exp(-(np.arange(T) - 30) / 8.0))
    value = ramp[:, None] * bump[None, :] + 0.15 * rng.standard_normal((T, B))
    rpe = (np.exp(-np.arange(T) / 25.0)[:, None]
           * (-np.exp(-0.5 * ((bins - 26) / 3.0) ** 2))[None, :]
           + 0.15 * rng.standard_normal((T, B)))
    precision = ((0.5 + 0.5 * np.exp(-0.5 * ((bins - 20) / 6.0) ** 2))[None, :]
                 * np.ones((T, 1)) + 0.15 * rng.standard_normal((T, B)))
    licks = 0.7 * value + 0.6 * rng.standard_normal((T, B))        # confound
    velocity = 0.5 * precision + 0.6 * rng.standard_normal((T, B))
    mask = np.ones((T, B))
    lat = dict(value=value, rpe=rpe, precision=precision)

    def z(a):
        return (a - a.mean()) / a.std()

    def noise():
        return 0.8 * rng.standard_normal((T, B))

    fr = np.stack([z(value) + noise(),       # n0 — encodes value
                   z(licks) + noise(),       # n1 — encodes a nuisance (licks)
                   z(rpe) + noise(),         # n2 — encodes RPE
                   z(precision) + noise(),   # n3 — encodes precision
                   noise()], axis=-1)        # n4 — nothing
    return fr, lat, licks, velocity, mask


def test_encoding_recovers_known_neurons_and_rejects_confounds():
    fr, lat, licks, velocity, mask = _synthetic_fr()
    res = encode_mouse(fr, lat, licks, velocity, mask, n_shuffle=200, seed=1)
    bp = res["beh"]["pval"]
    assert bp["value"][0] < ALPHA          # n0 encodes value
    assert bp["value"][1] >= ALPHA         # n1 (licks confound) NOT credited value
    assert bp["rpe"][1] >= ALPHA           # n1 NOT credited rpe
    assert bp["rpe"][2] < ALPHA            # n2 encodes RPE
    assert bp["precision"][3] < ALPHA      # n3 encodes precision
    assert bp["value"][4] >= ALPHA         # n4 (noise) NOT credited value


def test_spatial_demean_var_ratio_flags_stationary_latents():
    """A trial-varying latent keeps most variance after per-bin demeaning; a
    near-stationary (spatial-only) latent keeps almost none."""
    T, B = 120, 50
    bins = np.arange(B)
    spatial = np.exp(-0.5 * ((bins - 20) / 6.0) ** 2)
    mask = np.ones((T, B))

    stationary = np.tile(spatial, (T, 1))                 # no trial variation
    trial_varying = np.random.default_rng(0).standard_normal((T, B))

    assert spatial_demean_var_ratio(stationary, mask) < 0.05
    assert spatial_demean_var_ratio(trial_varying, mask) > 0.8
