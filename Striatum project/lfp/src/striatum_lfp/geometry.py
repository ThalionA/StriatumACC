"""Channel -> depth -> brain-area mapping for the LFP probe.

The LFP ``.mat`` carries no depth information (``depth_to_save`` is a ``[0, 0]``
placeholder), so channel index is mapped to depth-from-tip via the Neuropixels
1.0 geometry and then to an area using the same micrometre boundaries the spike
pipeline applies to sorted units (``Neuropixels_Depth_Data.csv`` /
``Neuropixels_V1_Depth_Data.csv``, parsed in ``OrganiseStriatumDataIncV1.m``).
Boundaries are inclusive ``[start, end]``, matching the ``depth >= start &
depth <= end`` test there.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config

# Area columns per probe in the two depth CSVs.
_STRIATUM_AREAS = ("DMS", "DLS", "ACC")
_VISUAL_AREAS = ("V1", "CA1", "DG")


def channel_depths(
    n_channels: int = config.N_CHANNELS,
    pitch_um: float = config.PITCH_UM,
    ch_per_row: int = config.CH_PER_ROW,
) -> np.ndarray:
    """Depth from tip (um) of each channel index ``0..n-1``.

    Neuropixels 1.0: ``ch_per_row`` channels share each ``pitch_um`` row, so
    ``depth[c] = (c // ch_per_row) * pitch_um``. Channel 0 sits at the tip
    (deepest in tissue == smallest depth-from-tip). Returns float ``(n_channels,)``.
    """
    c = np.arange(n_channels)
    return (c // ch_per_row) * float(pitch_um)


def _read_boundary_csv(path: Path, areas: tuple[str, ...]) -> dict[int, dict[str, tuple[float, float]]]:
    """``{mouse_id: {area: (start, end)}}`` from a depth CSV, skipping blank cells."""
    out: dict[int, dict[str, tuple[float, float]]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mid = (row.get("Mouse ID") or "").strip()
            if not mid:
                continue
            mouse = int(float(mid))
            bounds: dict[str, tuple[float, float]] = {}
            for area in areas:
                s = (row.get(f"{area} Start") or "").strip()
                e = (row.get(f"{area} End") or "").strip()
                if s == "" or e == "":
                    continue                # blank cell (e.g. 731 DLS) -> area absent
                bounds[area] = (float(s), float(e))
            out[mouse] = bounds
    return out


def load_area_boundaries(
    mouse_id: int,
    probe: str = "striatum",
    depth_csv: Path | None = None,
    v1_csv: Path | None = None,
    cohort=None,
) -> dict[str, tuple[float, float]]:
    """``{area: (start_um, end_um)}`` for one mouse and one probe.

    ``probe="striatum"`` -> DMS/DLS/ACC; ``probe="visual"`` -> V1/CA1/DG. The task
    and control groups have separate CSVs with overlapping mouse-number ranges, so
    pass ``cohort`` (default: task) rather than relying on the id to disambiguate.
    Areas with a blank CSV cell are omitted -- e.g. control 407 has no ACC.
    """
    ch = cohort or config.TASK
    if probe == "striatum":
        path = Path(depth_csv or ch.depth_csv)
        table = _read_boundary_csv(path, _STRIATUM_AREAS)
    elif probe == "visual":
        path = Path(v1_csv or ch.v1_csv)
        table = _read_boundary_csv(path, _VISUAL_AREAS)
    else:
        raise ValueError(f"unknown probe {probe!r} (expected 'striatum' or 'visual')")
    if mouse_id not in table:
        raise KeyError(f"mouse {mouse_id} not in {path.name}")
    return table[mouse_id]


def channel_area_masks(
    depths: np.ndarray, boundaries: dict[str, tuple[float, float]],
    precedence: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """``{area: bool mask (n_channels,)}`` from per-channel depths + ``{area: (s, e)}``.

    Inclusive ``[start, end]``, mirroring ``OrganiseStriatumDataIncV1.m``'s depth
    test. The bands can touch: mouse 1206's probe-2 CSV has DG ending and CA1
    starting at the same 1160 um. MATLAB resolves that by assigning areas in
    column order and letting the LAST one overwrite (``OrganiseStriatumDataIncV1.m``
    :162-179, ``assign_areas_by_depth.m``), so the returned masks are made
    mutually exclusive the same way -- ``precedence`` defaults to the CSV column
    order, and a channel claimed by two bands goes to whichever comes later.
    """
    depths = np.asarray(depths, float)
    order = precedence or _default_precedence(boundaries)
    masks = {area: (depths >= s) & (depths <= e) for area, (s, e) in boundaries.items()}
    claimed = np.zeros(depths.shape, dtype=bool)
    for area in reversed([a for a in order if a in masks]):
        masks[area] = masks[area] & ~claimed
        claimed |= masks[area]
    return masks


def _default_precedence(boundaries: dict[str, tuple[float, float]]) -> tuple[str, ...]:
    """CSV column order for whichever probe these boundaries came from."""
    known = _STRIATUM_AREAS if set(boundaries) <= set(_STRIATUM_AREAS) else _VISUAL_AREAS
    return tuple(a for a in known if a in boundaries)
