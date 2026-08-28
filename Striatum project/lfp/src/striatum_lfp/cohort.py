"""Discovery and identity verification for the per-mouse LFP voltage exports.

The 2026-08 download names every file after its task animal
(``<mouse>_voltage_data_384ch.mat`` = probe 1, striatum/ACC;
``<mouse>_v1_voltage_data_384ch.mat`` = probe 2, V1/CA1/DG), replacing the
size-keyed ``lfp_mapping.txt`` guesswork of the June files.

That naming is a **claim**, not a verified fact: the 2026-08-11 audit found two
mice sharing one byte-identical file, and the mix-up was only caught by
physiology. :func:`coupling_score` reimplements the test that resolved it -- the
high-frequency LFP envelope of a probe correlates with the multi-unit activity
recorded on *that* probe and not with another animal's -- so every filename can
be checked against the spikes rather than trusted.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import config

# ``<mouse>[_v1]voltage_data_384ch.mat``; the underscore before "voltage" is
# optional because 727's file ships without it.
_LFP_RE = re.compile(
    r"^(?P<mouse>\d+)_?(?P<probe>v1_?)?voltage_data_384ch\.mat$", re.IGNORECASE
)

PROBES = ("striatum", "visual")


def parse_lfp_filename(name: str,
                       mouse_ids: tuple[int, ...] | None = None) -> tuple[int, str] | None:
    """``(mouse_id, probe)`` for an LFP export filename, else ``None``.

    ``probe`` is ``"striatum"`` (probe 1: DMS/DLS/ACC) or ``"visual"``
    (probe 2: V1/CA1/DG). Files with no mouse prefix -- the superseded June
    copies and any download still in flight -- return ``None`` rather than being
    guessed at, and so does a prefix outside ``mouse_ids``. That last guard is
    load-bearing in both cohorts: 507 (task) and 408 (control) each have an
    export on disk but are absent from their organiser's analysis list.
    """
    match = _LFP_RE.match(Path(name).name)
    if match is None:
        return None
    mouse = int(match.group("mouse"))
    if mouse not in (config.TASK_MOUSE_IDS if mouse_ids is None else mouse_ids):
        return None
    return mouse, ("visual" if match.group("probe") else "striatum")


def discover_lfp_files(
    directory: str | Path, mouse_ids: tuple[int, ...] | None = None, *,
    return_skipped: bool = False
):
    """Map ``(mouse_id, probe) -> Path`` for every named export in ``directory``.

    With ``return_skipped`` also returns the sorted names that look like a
    voltage export but carry no resolvable mouse -- these must be reported, not
    silently dropped, because an unnamed file is either a superseded copy or a
    download in progress.
    """
    directory = Path(directory)
    found: dict[tuple[int, str], Path] = {}
    skipped: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        key = parse_lfp_filename(path.name, mouse_ids)
        if key is not None:
            found[key] = path
        elif "voltage_data" in path.name.lower():
            skipped.append(path.name)
    if return_skipped:
        return found, sorted(skipped)
    return found


def bin_mean(x: np.ndarray, bin_samples: int) -> np.ndarray:
    """Mean of ``x`` over non-overlapping bins of ``bin_samples`` along axis 0.

    An incomplete trailing bin is dropped, so the result is exactly
    ``x.shape[0] // bin_samples`` rows. Used to bring the 1 ms envelope and the
    1 ms spike train onto a common coarse grid before correlating them.
    """
    if bin_samples <= 0:
        raise ValueError("bin_samples must be positive")
    x = np.asarray(x, dtype=float)
    n = (x.shape[0] // bin_samples) * bin_samples
    trimmed = x[:n]
    return trimmed.reshape(n // bin_samples, bin_samples, *x.shape[1:]).mean(axis=1)


def pearson_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson ``r`` between each column of ``X`` and the vector ``y``.

    A zero-variance column yields ``nan`` (undefined), never ``0`` -- a dead
    channel must not be scored as "uncorrelated evidence".
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        raise ValueError(f"length mismatch: X has {X.shape[0]} rows, y has {y.size}")
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(axis=0) / denom
    r[denom == 0] = np.nan
    return r


def coupling_score(env: np.ndarray, mua: np.ndarray) -> float:
    """Mean ``|r|`` between each LFP channel's envelope and the MUA rate.

    The identity statistic: high when ``env`` and ``mua`` come from the same
    probe in the same session, at chance otherwise. ``nan`` channels (dead or
    zero-variance) are excluded rather than counted as zero.
    """
    r = pearson_columns(env, mua)
    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return float("nan")
    return float(np.abs(finite).mean())


def column_normalise(matrix: np.ndarray) -> np.ndarray:
    """Divide each column of a file x candidate score matrix by its median.

    Raw coupling scores are not comparable across candidates. An animal whose
    multi-unit rate carries a session-wide artefact -- a synchronous burst, a
    movement transient -- correlates with *every* file's high-frequency envelope
    and can out-score the true match on a row it has nothing to do with.
    Dividing each candidate column by its median across files asks the right
    question instead: does this file couple to that animal more than the other
    files do? A universal correlator has a high median and is scaled back to 1.
    """
    matrix = np.asarray(matrix, dtype=float)
    med = np.nanmedian(matrix, axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = matrix / med
    out[:, np.nan_to_num(med.ravel()) == 0] = np.nan
    return out
