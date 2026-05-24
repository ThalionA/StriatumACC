"""Stage 3 driver: communication-subspace membership and reorientation.

For one prepared area pair, fits the full-data CCA per epoch, back-projects the
canonical coefficients to neuron space, scores per-neuron membership two ways
(structure coefficients and raw weights), measures weight sparsity (Gini), and
computes principal angles between the epoch subspaces against a within-epoch
split-half noise floor. Pure compute over a PreparedPair, so it parallelises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, core, membership, pipeline, subspace

EPOCH_TRANSITIONS = (
    ("naive", "expert"),
)
# Canonical dims defining the "communication subspace" (D9/D10).
# Set to 1 (the dominant canonical direction): with only 10 trials per epoch
# the higher canonical dimensions are not reliably estimable -- at d_sub=3 the
# within-epoch split-half principal angle was already near-orthogonal, i.e.
# pure noise. The dominant dimension is what H&H and the subspace paper
# emphasise too. See NOTES.md / UNDERSTANDING.md edit log.
D_SUB_MAX = 1


@dataclass
class EpochSubspace:
    """Membership and subspace geometry for one (animal, pair, epoch)."""

    epoch: str
    d_sub: int
    cc: np.ndarray                  # full-data canonical correlations (d_sub,)
    weights_x: np.ndarray           # (n_units_x, d_sub) neuron-space weights
    weights_y: np.ndarray
    struct_x: np.ndarray            # (n_units_x, d_sub) structure coefficients
    struct_y: np.ndarray
    contribution_x: np.ndarray      # (n_units_x,) L2 norm of structure coeffs
    contribution_y: np.ndarray
    member_x: np.ndarray            # (n_units_x,) bool, top-quartile members
    member_y: np.ndarray
    gini_x: float                   # sparsity of dominant-dim weights
    gini_y: float
    split_half_angle_x: np.ndarray  # (d_sub,) within-epoch noise floor
    split_half_angle_y: np.ndarray


@dataclass
class PairSubspace:
    """Stage-3 result for one (animal, area-pair)."""

    animal_id: int
    area_x: str
    area_y: str
    role: str
    lp: int
    k: int
    d_sub: int
    unit_index_x: np.ndarray
    unit_index_y: np.ndarray
    epochs: dict[str, EpochSubspace]
    angles_x: dict[str, np.ndarray]   # "naive->expert" etc -> principal angles
    angles_y: dict[str, np.ndarray]


def _flatten(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(-1, tensor.shape[-1])


def analyse_subspace(prepared: pipeline.PreparedPair, cfg=config.DEFAULT) -> PairSubspace:
    """Compute membership + principal angles for one prepared area pair."""
    d_sub = min(D_SUB_MAX, prepared.k)
    epochs: dict[str, EpochSubspace] = {}
    weights_x: dict[str, np.ndarray] = {}
    weights_y: dict[str, np.ndarray] = {}

    for epoch in config.EPOCH_NAMES:
        sx = prepared.scores_x[epoch]
        sy = prepared.scores_y[epoch]
        comp_x = prepared.pca_x[epoch].components
        comp_y = prepared.pca_y[epoch].components

        fit = core.cca_fit(_flatten(sx), _flatten(sy))
        dd = min(d_sub, fit.A.shape[1])

        wx = subspace.canonical_weights(comp_x, fit.A, dd)
        wy = subspace.canonical_weights(comp_y, fit.B, dd)
        sc_x = membership.structure_coefficients(sx, comp_x, fit.A, dd)
        sc_y = membership.structure_coefficients(sy, comp_y, fit.B, dd)
        contrib_x = membership.subspace_contribution(sc_x)
        contrib_y = membership.subspace_contribution(sc_y)
        sha_x, sha_y = subspace.split_half_angles(
            sx, sy, comp_x, comp_y, dd, seed=cfg.cv_seed
        )

        epochs[epoch] = EpochSubspace(
            epoch=epoch,
            d_sub=dd,
            cc=fit.r[:dd],
            weights_x=wx,
            weights_y=wy,
            struct_x=sc_x,
            struct_y=sc_y,
            contribution_x=contrib_x,
            contribution_y=contrib_y,
            member_x=membership.member_mask(contrib_x),
            member_y=membership.member_mask(contrib_y),
            gini_x=membership.gini(wx[:, 0]),
            gini_y=membership.gini(wy[:, 0]),
            split_half_angle_x=sha_x,
            split_half_angle_y=sha_y,
        )
        weights_x[epoch] = wx
        weights_y[epoch] = wy

    angles_x: dict[str, np.ndarray] = {}
    angles_y: dict[str, np.ndarray] = {}
    for e1, e2 in EPOCH_TRANSITIONS:
        name = f"{e1}->{e2}"
        dd = min(weights_x[e1].shape[1], weights_x[e2].shape[1])
        angles_x[name] = subspace.principal_angles(
            weights_x[e1][:, :dd], weights_x[e2][:, :dd]
        )
        angles_y[name] = subspace.principal_angles(
            weights_y[e1][:, :dd], weights_y[e2][:, :dd]
        )

    return PairSubspace(
        animal_id=prepared.animal_id,
        area_x=prepared.area_x,
        area_y=prepared.area_y,
        role=prepared.role,
        lp=prepared.lp,
        k=prepared.k,
        d_sub=d_sub,
        unit_index_x=prepared.unit_index_x,
        unit_index_y=prepared.unit_index_y,
        epochs=epochs,
        angles_x=angles_x,
        angles_y=angles_y,
    )
