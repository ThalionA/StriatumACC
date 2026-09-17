"""Discrete information-theoretic estimators, in bits.

Built to mirror Lemke, Celotto, Maffulli, Ganguly & Panzeri (2024), *Information
flow between motor cortex and striatum reverses during skill learning*, Curr Biol
34:1831-1843, whose analysis this project wants to repeat on the corridor task.
Their chain is: mutual information between neural activity and a behavioural
feature, a partial information decomposition of what two areas jointly carry, and
a directed measure (transfer entropy, plus a feature-specific transfer built from
PID atoms on top of it).

Three decisions are worth stating because they are where such analyses go wrong.

**Discretisation.** Everything here takes integer codes and never guesses. The
paper used 5 equipopulated bins for LFP magnitude and 3 for each movement
feature; this repo's existing MATLAB arm
(``MutualInformationStriatum_v2.m:discretize_zero_aware``) instead gives exact
zeros their own bin and equipopulates the rest, because spike counts and lick
rates are sparse and an equipopulated split scatters the zeros. Both schemes are
provided; the caller picks, and the choice is recorded in the output.

**Bias.** The plug-in estimate of mutual information is biased UPWARD, and on the
sample sizes here that bias is not small: 5 x 3 bins with 200 trials gives a
visibly positive value on independent data (pinned by test). Two corrections are
offered — the paper's shuffle subtraction, and the Miller-Madow analytic term the
existing MATLAB arm already uses — so the two can be compared on the same data
rather than argued about. Neither is applied by default: ``mutual_information``
returns the plug-in value unless asked, so no caller gets a correction it did not
choose.

**Redundancy.** The PID here uses ``I_min`` (Williams & Beer 2010), which is what
the paper itself uses for the three-source atoms inside FIT. For the two-source
shared information the paper uses BROJA instead, which needs a constrained
optimisation and is NOT implemented here; ``pid_imin`` says so in its docstring
rather than quietly substituting a different measure. I_min is known to
over-attribute redundancy for correlated sources, so a redundancy number from
this module is an upper bound.

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np

LOG2 = np.log(2.0)


# --------------------------------------------------------------------------
# discretisation
# --------------------------------------------------------------------------

def equipopulated_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Integer codes ``0..n_bins-1`` holding (as near as possible) equal counts.

    The paper's scheme. Ties are broken by rank rather than by jittering the
    values, so the result is deterministic.
    """
    x = np.asarray(x, dtype=float).ravel()
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if n_bins == 1:
        return np.zeros(x.size, dtype=int)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return np.minimum((order * n_bins) // x.size, n_bins - 1).astype(int)


def zero_aware_bins(x: np.ndarray, n_bins: int, *, tol: float = 1e-8) -> np.ndarray:
    """Exact zeros in bin 0; the rest equipopulated across the remaining bins.

    The convention of ``MutualInformationStriatum_v2.m``, kept so a Python result
    can be compared with the existing MATLAB one without a binning difference
    explaining any discrepancy.
    """
    x = np.asarray(x, dtype=float).ravel()
    out = np.zeros(x.size, dtype=int)
    nz = np.abs(x) >= tol
    if not nz.any() or n_bins < 2:
        return out
    out[nz] = 1 + equipopulated_bins(x[nz], n_bins - 1)
    return out


def pair_code(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A single integer code for the joint state of two discrete variables."""
    x = np.asarray(x, dtype=int).ravel()
    y = np.asarray(y, dtype=int).ravel()
    return x * (int(y.max()) + 1) + y


# --------------------------------------------------------------------------
# entropies and mutual information
# --------------------------------------------------------------------------

def _joint_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=int).ravel()
    y = np.asarray(y, dtype=int).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must match; got {x.size} and {y.size}")
    nx, ny = int(x.max()) + 1, int(y.max()) + 1
    return np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)


def mutual_information(x: np.ndarray, y: np.ndarray, *,
                       correction: str | None = None) -> float:
    """``I(X;Y)`` in bits from integer codes.

    ``correction="miller_madow"`` subtracts the asymptotic
    ``(R_xy - R_x - R_y + 1) / (2 N ln2)`` term the repo's MATLAB arm uses, where
    each ``R`` counts OCCUPIED states rather than possible ones. It is not
    clamped at zero here, unlike the MATLAB version: clamping turns a symmetric
    estimator error into a one-sided one and quietly guarantees a positive mean.
    """
    counts = _joint_counts(x, y)
    n = counts.sum()
    if n == 0:
        return np.nan
    pxy = counts / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])) / LOG2)
    if correction is None:
        return mi
    if correction != "miller_madow":
        raise ValueError(f"unknown correction {correction!r}")
    r_xy = int((counts > 0).sum())
    r_x = int((counts.sum(axis=1) > 0).sum())
    r_y = int((counts.sum(axis=0) > 0).sum())
    return mi - (r_xy - r_x - r_y + 1) / (2.0 * n * LOG2)


def conditional_mutual_information(x: np.ndarray, y: np.ndarray,
                                   z: np.ndarray) -> float:
    """``I(X;Y|Z)`` in bits, summed over the states of ``Z``."""
    x = np.asarray(x, dtype=int).ravel()
    y = np.asarray(y, dtype=int).ravel()
    z = np.asarray(z, dtype=int).ravel()
    n = x.size
    total = 0.0
    for zv in np.unique(z):
        m = z == zv
        w = m.sum() / n
        if m.sum() < 2:
            continue
        total += w * mutual_information(x[m], y[m])
    return float(total)


def shuffle_subtracted_mi(x: np.ndarray, y: np.ndarray, *, n_shuffles: int = 100,
                          seed: int = 0) -> dict:
    """The paper's bias handling: subtract the mean of a shuffled distribution.

    ``y`` is permuted, which destroys the dependence while leaving both marginals
    and the bin counts exactly as they were -- so the shuffled mean estimates the
    bias that the same bin count produces on this sample size, and subtracting it
    removes it. Also returns the permutation p-value, since the same shuffles
    give it for free.
    """
    observed = mutual_information(x, y)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int).ravel()
    null = np.array([mutual_information(x, rng.permutation(y))
                     for _ in range(n_shuffles)])
    return {
        "mi_plugin": float(observed),
        "mi_shuffled_mean": float(null.mean()),
        "mi_corrected": float(observed - null.mean()),
        "p": float((1.0 + int((null >= observed).sum())) / (n_shuffles + 1.0)),
        "n_shuffles": int(n_shuffles),
    }


# --------------------------------------------------------------------------
# partial information decomposition
# --------------------------------------------------------------------------

def _specific_information(source: np.ndarray, target: np.ndarray,
                          target_value: int) -> float:
    """``I(S=s; X)``: how much knowing X reduces surprise about this one target state."""
    counts = _joint_counts(target, source)
    n = counts.sum()
    p_s = counts[target_value].sum() / n
    if p_s <= 0:
        return 0.0
    p_x = counts.sum(axis=0) / n
    p_x_given_s = counts[target_value] / counts[target_value].sum()
    nz = (p_x_given_s > 0) & (p_x > 0)
    p_s_given_x = counts[target_value][nz] / counts.sum(axis=0)[nz]
    return float(np.sum(p_x_given_s[nz] * (np.log(1.0 / p_s) - np.log(1.0 / p_s_given_x)) / LOG2))


def pid_imin(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> dict:
    """Williams & Beer partial information decomposition of ``I(X,Y;S)``.

    Redundancy is ``I_min``: for each target state, the smaller of what each
    source tells you about that state, averaged over states. Unique terms are
    each source's own information minus the redundancy, and synergy is whatever
    the pair carries beyond the three.

    **This is not BROJA.** The paper uses BROJA for two-source shared information
    (and I_min only for the three-source atoms inside FIT). BROJA needs a
    constrained optimisation over distributions with fixed pairwise marginals and
    is not implemented here. I_min is known to over-attribute redundancy when the
    sources are correlated, so treat the redundancy returned here as an upper
    bound and the synergy as a lower one.
    """
    x = np.asarray(x, dtype=int).ravel()
    y = np.asarray(y, dtype=int).ravel()
    s = np.asarray(target, dtype=int).ravel()
    n = s.size
    redundancy = 0.0
    for sv in np.unique(s):
        p_s = (s == sv).sum() / n
        redundancy += p_s * min(_specific_information(x, s, int(sv)),
                                _specific_information(y, s, int(sv)))
    i_x = mutual_information(x, s)
    i_y = mutual_information(y, s)
    i_joint = mutual_information(pair_code(x, y), s)
    return {
        "redundancy": float(redundancy),
        "unique_x": float(i_x - redundancy),
        "unique_y": float(i_y - redundancy),
        "synergy": float(i_joint - i_x - i_y + redundancy),
        "i_x": float(i_x),
        "i_y": float(i_y),
        "i_joint": float(i_joint),
    }


# --------------------------------------------------------------------------
# directed measures
# --------------------------------------------------------------------------

def transfer_entropy(source: np.ndarray, target: np.ndarray, *, lag: int = 1) -> float:
    """``TE(X -> Y) = I(X_{t-d}; Y_t | Y_{t-d})`` in bits.

    The paper's single-time-point form, in which the sender's past and the
    receiver's past share the SAME lag. Conditioning on the receiver's own past
    is what stops a signal that merely echoes ``Y`` from looking like transfer
    into it — pinned by test.
    """
    x = np.asarray(source, dtype=int).ravel()
    y = np.asarray(target, dtype=int).ravel()
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if x.size != y.size:
        raise ValueError("source and target must be the same length")
    if x.size <= lag + 1:
        return np.nan
    return conditional_mutual_information(x[:-lag], y[lag:], y[:-lag])
