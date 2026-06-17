"""Linear mixed-effects test of the epoch (learning) effect, per area pair.

The landmark arm produces one cell stat per (animal, landmark, epoch). Pooling
the 6 landmarks per animal into an independent-sample Wilcoxon pseudoreplicates
-- the landmarks are repeated measures, not independent draws -- which inflates
significance (see ``STATE.md`` §3). The correct test models the within-animal
correlation explicitly:

    value ~ C(epoch) [+ C(landmark)]   with a per-animal random intercept.

``lmm_epoch_contrasts`` fits that model for one pair's records and returns, for
each epoch contrast, the estimated mean shift and its Wald p-value, computed
over ``n_animals`` -- the honest experimental unit. The spatial arm and Arm A
have no landmark dimension; pass ``landmark=None`` to drop that term.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DEFAULT_CONTRASTS = (("expert", "naive"), ("expert", "intermediate"),
                     ("intermediate", "naive"))
MIN_ANIMALS = 3


def _na(reason: str) -> dict:
    return {"estimate": float("nan"), "p": float("nan"),
            "n_animals": 0, "n_obs": 0, "ok": False, "reason": reason}


def collapsed_epoch_contrasts(records, value: str = "value",
                              group: str = "animal_id", epoch: str = "epoch",
                              contrasts=DEFAULT_CONTRASTS) -> dict:
    """Per-animal summary-statistics test (the robust small-N companion).

    Collapse the repeated measures (landmarks) to one value per (animal, epoch)
    by averaging, then run a paired Wilcoxon across animals on the per-animal
    delta. This is the textbook repeated-measures treatment and is numerically
    bulletproof where the random-slope LMM is not -- but it is underpowered
    (n = #animals; the two-sided signed-rank p-floor at n=4 is 0.125).
    """
    from . import paired_stats
    df = pd.DataFrame(records)
    out = {f"{a}-{b}": _na("not_fit") for a, b in contrasts}
    if df.empty or value not in df:
        return out
    df = df.dropna(subset=[value, group, epoch])
    means = df.groupby([group, epoch])[value].mean()
    for a, b in contrasts:
        key = f"{a}-{b}"
        deltas = []
        for animal in df[group].unique():
            if (animal, a) in means.index and (animal, b) in means.index:
                deltas.append(float(means[(animal, a)] - means[(animal, b)]))
        n, med, w, p = paired_stats.wilcoxon_signed(deltas)
        out[key] = {"estimate": med, "p": p, "n_animals": n,
                    "n_obs": int(len(df)), "ok": n >= 3 and np.isfinite(p),
                    "reason": "collapsed"}
    return out


def lmm_slope(records, value: str = "value", group: str = "animal_id",
              axis: str = "axis") -> dict:
    """Population slope of ``value`` on a continuous ``axis`` via a random-slope
    LMM (``value ~ axis`` with ``~axis`` random effects per animal).

    Pools ALL windows across animals (more powerful than a per-animal-slope sign
    test) while accounting for within-animal correlation. Returns the fixed-effect
    slope, its Wald p-value, ``n_animals`` and ``n_obs``. ``ok=False`` (NaN p) if
    fewer than MIN_ANIMALS animals or the fit fails.
    """
    df = pd.DataFrame(records)
    out = _na("not_fit")
    if df.empty or value not in df or axis not in df:
        return out
    df = df.dropna(subset=[value, group, axis])
    n_animals = int(df[group].nunique())
    if n_animals < MIN_ANIMALS or df[axis].nunique() < 3:
        return {**_na("too_few"), "n_animals": n_animals}
    # A random slope on a continuous axis gives each animal a random
    # intercept AND a random slope -> a 2x2 RE covariance (3 free params). It is
    # unidentifiable unless there are more animals than covariance parameters;
    # below that the fit degenerates and the Wald p collapses to an
    # anticonservative value (same hazard guarded in lmm_epoch_contrasts).
    if n_animals <= 3:
        return {**_na("too_few_animals_for_random_slope"), "n_animals": n_animals}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(f"{value} ~ {axis}", df, groups=df[group],
                                re_formula=f"~{axis}")
            fit = model.fit(reml=False, method="lbfgs")
    except Exception as exc:                                # noqa: BLE001
        return {**_na(f"fit_failed:{type(exc).__name__}"), "n_animals": n_animals}
    if axis not in fit.fe_params.index:
        return {**_na("no_slope_term"), "n_animals": n_animals}
    est = float(fit.fe_params[axis])
    var = float(fit.cov_params().loc[axis, axis])
    if not np.isfinite(var) or var <= 0:
        return {**_na("bad_variance"), "n_animals": n_animals}
    from scipy import stats as _st
    p = float(2 * _st.norm.sf(abs(est / np.sqrt(var))))
    return {"estimate": est, "p": p, "n_animals": n_animals,
            "n_obs": int(len(df)), "ok": True, "reason": ""}


def lmm_epoch_contrasts(records, value: str = "value",
                        group: str = "animal_id", epoch: str = "epoch",
                        landmark: str | None = "landmark",
                        contrasts=DEFAULT_CONTRASTS) -> dict:
    """Fit one per-pair LMM and return per-contrast estimates + Wald p-values.

    ``records`` is a list of dicts (or a DataFrame) with at least ``value``,
    ``group`` (animal), ``epoch`` columns and -- unless ``landmark is None`` --
    a ``landmark`` column. Returns ``{f"{a}-{b}": {estimate, p, n_animals,
    n_obs, ok, reason}}`` for every contrast ``(a, b)``. Contrasts that cannot
    be estimated come back with ``ok=False`` and a NaN p-value rather than
    raising, so a sweep over many pairs never aborts on one bad fit.
    """
    df = pd.DataFrame(records)
    out: dict = {f"{a}-{b}": _na("not_fit") for a, b in contrasts}
    if df.empty or value not in df:
        return out
    df = df.dropna(subset=[value, group, epoch])
    epochs_present = set(df[epoch].unique())
    n_animals = int(df[group].nunique())
    if n_animals < MIN_ANIMALS:
        return {k: _na("too_few_animals") for k in out}

    # Reference level "naive" must be present for the Treatment coding to map
    # cleanly onto the contrasts; fall back to the lexicographically first.
    # A random slope on a k-level epoch has k random effects per animal, hence
    # a k*(k+1)/2-parameter covariance. That is unidentifiable unless there are
    # more animals than covariance parameters; below that the fit degenerates
    # and the Wald p collapses to the pseudoreplicated value. Reject up front.
    k = len(epochs_present)
    re_cov_params = k * (k + 1) // 2
    if n_animals <= re_cov_params:
        return {kk: {**_na("too_few_animals_for_random_slope"),
                     "n_animals": n_animals} for kk in out}

    ref = "naive" if "naive" in epochs_present else sorted(epochs_present)[0]
    epoch_term = f"C({epoch}, Treatment('{ref}'))"
    terms = [epoch_term]
    if landmark is not None and landmark in df and df[landmark].nunique() > 1:
        terms.append(f"C({landmark})")
    formula = f"{value} ~ " + " + ".join(terms)
    # Random SLOPE on epoch per animal (not just a random intercept): the
    # within-animal epoch contrast is independent of the animal intercept, so a
    # random intercept alone would still test the effect against pooled
    # per-observation residuals (re-pseudoreplicating the landmarks). The random
    # slope makes the effect's SE reflect BETWEEN-animal variation (n = animals).
    re_formula = f"~ {epoch_term}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # convergence chatter
            model = smf.mixedlm(formula, df, groups=df[group],
                                re_formula=re_formula)
            fit = model.fit(reml=False, method="lbfgs")
    except Exception as exc:                            # noqa: BLE001
        return {k: _na(f"fit_failed:{type(exc).__name__}") for k in out}

    params = fit.fe_params
    names = list(params.index)
    # cov_params() spans fixed effects AND the random-effect variance terms;
    # restrict to the fixed-effect block so the contrast variance is well-formed.
    cov = fit.cov_params().loc[names, names]

    def coef_name(level: str) -> str | None:
        want = f"C({epoch}, Treatment('{ref}'))[T.{level}]"
        return want if want in names else None

    def contrast_vector(a: str, b: str) -> np.ndarray | None:
        # estimate(a) - estimate(b) where each level's estimate is its coef vs
        # ref (the ref level itself has coef 0).
        vec = np.zeros(len(names))
        for level, sign in ((a, +1.0), (b, -1.0)):
            if level == ref:
                continue
            cn = coef_name(level)
            if cn is None:
                return None
            vec[names.index(cn)] = sign
        return vec

    n_obs = int(len(df))
    for a, b in contrasts:
        key = f"{a}-{b}"
        if not ({a, b} <= epochs_present):
            out[key] = _na("epoch_missing")
            out[key]["n_animals"] = n_animals
            continue
        vec = contrast_vector(a, b)
        if vec is None:
            out[key] = _na("contrast_unestimable")
            out[key]["n_animals"] = n_animals
            continue
        est = float(vec @ params.values)
        var = float(vec @ cov.values @ vec)
        if not np.isfinite(var) or var <= 0:
            out[key] = _na("bad_variance")
            out[key]["n_animals"] = n_animals
            continue
        z = est / np.sqrt(var)
        from scipy import stats as _st
        p = float(2 * _st.norm.sf(abs(z)))
        out[key] = {"estimate": est, "p": p, "n_animals": n_animals,
                    "n_obs": n_obs, "ok": True, "reason": ""}
    return out
