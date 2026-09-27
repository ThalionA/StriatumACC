"""Remove running speed from a behavioural feature before comparing epochs.

The mouth and whisker ROIs carry running-related motion (they dip wherever the
mouse stops), and Expert trials are run faster, so an epoch difference in raw
ME is confounded with speed. SpeedModel is a nonparametric E[feature | speed]:
means in speed-quantile bins, interpolated linearly between bin centres, and
clamped at the ends. Fit it on trials OUTSIDE the contrast so that the nuisance
fit cannot absorb the effect being tested.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeedModel:
    centres: np.ndarray  # mean speed in each quantile bin
    means: np.ndarray    # mean feature in each quantile bin

    @classmethod
    def fit(cls, speed, feature, n_bins=20):
        speed, feature = np.asarray(speed, float), np.asarray(feature, float)
        ok = np.isfinite(speed) & np.isfinite(feature)
        s, f = speed[ok], feature[ok]
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, edges.size - 2)
        keep = np.bincount(idx, minlength=edges.size - 1) > 0
        centres = np.bincount(idx, s, edges.size - 1)[keep] / np.bincount(idx, minlength=edges.size - 1)[keep]
        means = np.bincount(idx, f, edges.size - 1)[keep] / np.bincount(idx, minlength=edges.size - 1)[keep]
        return cls(centres, means)

    def predict(self, speed):
        speed = np.asarray(speed, float)
        out = np.interp(speed, self.centres, self.means)
        out[~np.isfinite(speed)] = np.nan
        return out


def partial_corr(x, y, speed, n_bins=20):
    """Pearson r between x and y after removing E[. | speed] from each."""
    x, y, speed = (np.asarray(a, float) for a in (x, y, speed))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(speed)
    rx = x[ok] - SpeedModel.fit(speed[ok], x[ok], n_bins).predict(speed[ok])
    ry = y[ok] - SpeedModel.fit(speed[ok], y[ok], n_bins).predict(speed[ok])
    return float(np.corrcoef(rx, ry)[0, 1])
