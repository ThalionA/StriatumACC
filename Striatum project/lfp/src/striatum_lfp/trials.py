"""The one trial layer every driver uses: which trials exist, which are analysed,
and which fall in each learning epoch -- all as raw trial indices.

Three numberings meet here, and mixing them is how trials got paired with their
neighbours' behaviour before this module existed:

* **Raw trials** -- every trial the VR logged, in order. The band-power cubes,
  ``binned_spikes_trials``, ``npx_times_trials`` and ``trial_metrics`` are all
  indexed this way, and so is the disengagement point (``change_point_mean`` is
  computed before ``ProcessStriatumTask.m`` filters anything).
* **Good trials** -- raw trials whose corridor reward is non-empty, MATLAB's own
  filter (``ProcessStriatumTask.m:94``). ``zscored_lick_errors`` and therefore the
  learning point are numbered this way, and ``epoch_indices.m`` windows count
  good trials. The mask is read back from ``corridorData.trial_reward``, so it is
  MATLAB's rule, not a re-derivation.
* **Covered trials** -- raw trials the voltage export actually spans (407's stops
  26 min early; the cubes store at most 200 trials).

A trial is analysed if it is good, engaged (raw number <= DP) and covered. An
epoch window is defined on the good numbering up to DP, exactly as MATLAB passes
a DP-truncated ``n_trials`` to ``epoch_indices``, and it exists only if every
trial in it is covered: a window is dropped, never silently shortened.

A missing DP (NaN: the change-point detector found none) means no clip, which is
MATLAB's own behaviour -- ``min([change_point_mean, n_trials])`` ignores NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from . import analysis, config

EPOCHS: tuple[str, ...] = ("Naive", "Intermediate", "Expert")


@dataclass(frozen=True)
class SessionTrials:
    """One animal's trials: MATLAB's good mask, LP, DP and optional coverage."""

    mouse: int
    matlab_good: np.ndarray          # bool per raw trial
    lp: int | None                   # 1-based, on the good-trial numbering
    lp_source: str                   # "measured" | "cohort_average"
    dp: float                        # 1-based raw trial number; nan = no clip
    has_data: np.ndarray | None = None   # bool per raw trial; None = all covered

    @property
    def n_raw(self) -> int:
        return int(self.matlab_good.size)

    def with_data(self, has_data: np.ndarray) -> SessionTrials:
        """The same session restricted to the raw trials a recording covers."""
        return replace(self, has_data=np.asarray(has_data, bool))

    def _covered(self) -> np.ndarray:
        if self.has_data is None:
            return np.ones(self.n_raw, bool)
        out = np.zeros(self.n_raw, bool)
        n = min(self.n_raw, self.has_data.size)
        out[:n] = self.has_data[:n]
        return out

    def _engaged(self) -> np.ndarray:
        if not np.isfinite(self.dp):
            return np.ones(self.n_raw, bool)
        return np.arange(1, self.n_raw + 1) <= self.dp

    def usable(self) -> np.ndarray:
        """Raw 0-based indices of trials that are good, engaged and covered."""
        return np.flatnonzero(self.matlab_good & self._engaged() & self._covered())

    def epochs(self) -> dict[str, np.ndarray]:
        """``{epoch: raw 0-based indices}``; an empty array where the window does not exist."""
        good_raw = np.flatnonzero(self.matlab_good & self._engaged())
        covered = self._covered()
        out: dict[str, np.ndarray] = {}
        for name, window in zip(EPOCHS, analysis.epoch_indices(self.lp, good_raw.size)):
            raw = good_raw[np.asarray(window, int) - 1]
            out[name] = raw if raw.size and covered[raw].all() else np.array([], int)
        return out

    def epoch_of(self) -> np.ndarray:
        """Epoch label per raw trial (``""`` outside every window)."""
        labels = np.full(self.n_raw, "", dtype=object)
        for name, raw in self.epochs().items():
            labels[raw] = name
        return labels


def matlab_good_masks(cohort=None, preproc_mat=None) -> dict[int, np.ndarray]:
    """``{mouse: bool per raw trial}`` -- ``~cellfun(@isempty, trial_reward)``."""
    import h5py

    ch = cohort or config.TASK
    out: dict[int, np.ndarray] = {}
    with h5py.File(preproc_mat or ch.preproc_mat, "r") as handle:
        P = handle["preprocessed_data"]
        for i, mouse in enumerate(ch.mouse_ids):
            refs = np.asarray(handle[P["corridorData"][i, 0]]["trial_reward"]).ravel()
            out[mouse] = np.array([not _is_empty(handle[r]) for r in refs], bool)
    return out


def _is_empty(dataset) -> bool:
    """MATLAB stores ``[]`` as a 2-element dims array flagged ``MATLAB_empty``."""
    return bool(dataset.attrs.get("MATLAB_empty", 0)) or dataset.size == 0


def cohort_sessions(cohort=None, preproc_mat=None) -> dict[int, SessionTrials]:
    """``{mouse: SessionTrials}`` for a cohort, read once from the preprocessed struct."""
    ch = cohort or config.TASK
    lps = analysis.cohort_learning_points(ch, preproc_mat)
    sources = analysis.learning_point_sources(ch, preproc_mat)
    dps = analysis.disengagement_points(ch, preproc_mat)
    masks = matlab_good_masks(ch, preproc_mat)
    return {m: SessionTrials(mouse=m, matlab_good=masks[m], lp=lps.get(m),
                             lp_source=sources.get(m, "unknown"),
                             dp=float(dps.get(m, np.nan)))
            for m in ch.mouse_ids}


@lru_cache(maxsize=None)
def sessions_for(cohort_name: str) -> dict[int, SessionTrials]:
    """:func:`cohort_sessions` by cohort name, read once per process."""
    return cohort_sessions(config.get_cohort(cohort_name))
