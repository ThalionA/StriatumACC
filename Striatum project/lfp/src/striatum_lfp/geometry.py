"""Channel -> depth -> brain-area mapping for the LFP probe.

Channel index is mapped to depth-from-tip via the Neuropixels 1.0 geometry and
then to an area using the same micrometre boundaries the spike pipeline applies
to sorted units (``Neuropixels_Depth_Data.csv`` / ``Neuropixels_V1_Depth_Data.csv``,
parsed in ``OrganiseStriatumDataIncV1.m``). Boundaries are inclusive
``[start, end]``, matching the ``depth >= start & depth <= end`` test there.

Two depth conventions exist and differ by one 20 um row. The export ships
``depth_to_save`` = 0-3820 um (:func:`export_depths`); the unit depths the
boundaries were drawn against are Kilosort ``ycoords``, 20-3840 um over every
bundle (:func:`channel_depths`). Areas are assigned in the unit convention, so a
channel and a unit on the same site land in the same area.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config

# Area columns per probe in the two depth CSVs.
_STRIATUM_AREAS = ("DMS", "DLS", "ACC")
_VISUAL_AREAS = ("V1", "CA1", "DG")


def export_depths(
    n_channels: int = config.N_CHANNELS,
    pitch_um: float = config.PITCH_UM,
    ch_per_row: int = config.CH_PER_ROW,
) -> np.ndarray:
    """Depth (um) of each channel as the export's ``depth_to_save`` states it.

    Neuropixels 1.0: ``ch_per_row`` channels share each ``pitch_um`` row and
    channel 0 sits at the tip, so ``depth[c] = (c // ch_per_row) * pitch_um``.
    """
    c = np.arange(n_channels)
    return (c // ch_per_row) * float(pitch_um)


def channel_depths(
    n_channels: int = config.N_CHANNELS,
    pitch_um: float = config.PITCH_UM,
    ch_per_row: int = config.CH_PER_ROW,
) -> np.ndarray:
    """Depth (um) of each channel in the unit (Kilosort ``ycoords``) convention.

    One row above :func:`export_depths`: 20 um for the tip row, 3840 um for the
    top. This is the convention of ``goodcluster2(:, 2)`` and therefore of the
    area boundaries in the depth CSVs.
    """
    return export_depths(n_channels, pitch_um, ch_per_row) + float(pitch_um)


def vertical_pairs(channels: np.ndarray, ch_per_row: int = config.CH_PER_ROW) -> np.ndarray:
    """Non-overlapping bipolar pairs one row apart: ``(n_pairs, 2)`` as (deep, shallow).

    On Neuropixels 1.0 channel ``c`` and ``c + ch_per_row`` are the nearest sites
    one row up. Pairing within a row (``c``, ``c + 1``) differences two sites at
    the SAME depth, which cancels the local depth gradient along with the far
    field. Channels are taken in blocks of ``2 * ch_per_row`` -- ``(4k, 4k+2)`` and
    ``(4k+1, 4k+3)`` -- so no channel is used twice, both members must be in
    ``channels``, and neither may be a reference site. The orientation is fixed
    (deep, shallow) so a derivation's sign never depends on sort order.
    """
    present = set(np.asarray(channels, int).tolist()) - set(config.REFERENCE_CHANNELS)
    block = 2 * ch_per_row
    pairs = [(c, c + ch_per_row) for c in sorted(present)
             if (c % block) < ch_per_row and (c + ch_per_row) in present]
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


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

    For a full 384-channel depth vector the reference site(s) in
    ``config.REFERENCE_CHANNELS`` belong to no area.
    """
    depths = np.asarray(depths, float)
    order = precedence or _default_precedence(boundaries)
    tissue = np.ones(depths.shape, dtype=bool)
    if depths.size == config.N_CHANNELS:        # indexed by channel: drop the reference
        tissue[list(config.REFERENCE_CHANNELS)] = False
    masks = {area: (depths >= s) & (depths <= e) & tissue
             for area, (s, e) in boundaries.items()}
    claimed = np.zeros(depths.shape, dtype=bool)
    for area in reversed([a for a in order if a in masks]):
        masks[area] = masks[area] & ~claimed
        claimed |= masks[area]
    return masks


def _default_precedence(boundaries: dict[str, tuple[float, float]]) -> tuple[str, ...]:
    """CSV column order for whichever probe these boundaries came from."""
    known = _STRIATUM_AREAS if set(boundaries) <= set(_STRIATUM_AREAS) else _VISUAL_AREAS
    return tuple(a for a in known if a in boundaries)
