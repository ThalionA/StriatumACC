"""Per-neuron encoding analysis across the 16 task mice — do area neurons
encode the RL latents (value, RPE, precision)?

A synthetic ground-truth check runs first (the TDD gate); then each invocation
encodes a batch of mice (incremental — results saved per mouse) and the final
invocation aggregates per area and writes the figure.

    python -m scripts.run_neural_encoding        # repeat until it prints DONE
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.config import TaskConfig                                  # noqa
from rl_model.io_real import load_real_cohort                           # noqa
from rl_model.neural_encoding import load_neural, encode_mouse, LATENTS  # noqa

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")
ENCDIR = os.path.join(RESDIR, "encoding_v6")   # v6: + slow-drift nuisance + bin-shuffle null
MAT = os.path.join(HERE, "..", "processed_data", "preprocessed_data5cm.mat")
LATENTS_NPZ = os.path.join(RESDIR, "rl_latents.npz")
os.makedirs(ENCDIR, exist_ok=True)

MODELS = ("beh", "beh_spatial")
PLOT_AREAS = ("acc", "dms", "dls", "v1", "ca1", "dg")
TIME_BUDGET = 30.0
ALPHA = 0.05


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
# Synthetic ground-truth check (TDD gate)
# --------------------------------------------------------------------------
def synthetic_check():
    """Neurons with known encoding must be recovered; a neuron driven purely by
    a behavioural nuisance must NOT score as encoding a latent."""
    rng = np.random.default_rng(0)
    T, B = 140, 50
    bins = np.arange(B)
    bump = np.exp(-0.5 * ((bins - 28) / 4.0) ** 2)
    ramp = 1.0 / (1.0 + np.exp(-(np.arange(T) - 30) / 8.0))
    value = ramp[:, None] * bump[None, :] + 0.15 * rng.standard_normal((T, B))
    rpe = (np.exp(-np.arange(T) / 25.0)[:, None]
           * (-np.exp(-0.5 * ((bins - 26) / 3.0) ** 2))[None, :]
           + 0.15 * rng.standard_normal((T, B)))
    precision = ((0.5 + 0.5 * np.exp(-0.5 * ((bins - 20) / 6.0) ** 2))[None, :]
                 * np.ones((T, 1)) + 0.15 * rng.standard_normal((T, B)))
    licks = 0.7 * value + 0.6 * rng.standard_normal((T, B))      # confound
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

    res = encode_mouse(fr, lat, licks, velocity, mask, n_shuffle=200, seed=1)
    bp = res["beh"]["pval"]
    checks = [
        ("n0 encodes value",        bp["value"][0] < ALPHA),
        ("n1 (licks) NOT value",    bp["value"][1] >= ALPHA),
        ("n1 (licks) NOT rpe",      bp["rpe"][1] >= ALPHA),
        ("n2 encodes RPE",          bp["rpe"][2] < ALPHA),
        ("n3 encodes precision",    bp["precision"][3] < ALPHA),
        ("n4 (noise) NOT value",    bp["value"][4] >= ALPHA),
    ]
    for name, passed in checks:
        log(f"   [{'PASS' if passed else 'FAIL'}] {name}")
    return all(p for _, p in checks)


# --------------------------------------------------------------------------
# Real-data encoding
# --------------------------------------------------------------------------
def fit_batch():
    cfg = TaskConfig()
    cohort = load_real_cohort(MAT, cfg)
    neural = load_neural(MAT)
    lat_npz = np.load(LATENTS_NPZ, allow_pickle=True)

    done = {f.split(".")[0] for f in os.listdir(ENCDIR) if f.endswith(".npz")}
    todo = [(i, m) for i, m in enumerate(cohort) if m["mouse"] not in done]
    if not todo:
        return True
    t0 = time.time()
    for i, m in todo:
        if time.time() - t0 > TIME_BUDGET and len(done) > len(todo) * 0:
            if done:                       # at least one done this run
                break
        mid = m["mouse"]
        nt = m["n_trials"]
        fr = neural[i]["fr"][:nt]
        lat = {k: np.asarray(lat_npz[k][i], dtype=float) for k in LATENTS}
        vel = np.exp(m["logv"])
        res = encode_mouse(fr, lat, m["licks"], vel, m["mask"])
        out = dict(area=neural[i]["area"].astype(str))
        for mdl in MODELS:
            out[f"r2_{mdl}"] = res[mdl]["r2_full"]
            for k in LATENTS:
                out[f"dR2_{mdl}_{k}"] = res[mdl]["dR2"][k]
                out[f"pval_{mdl}_{k}"] = res[mdl]["pval"][k]
                out[f"pvalbin_{mdl}_{k}"] = res[mdl]["pval_bin"][k]
        np.savez(os.path.join(ENCDIR, f"{mid}.npz"), **out)
        done.add(mid)
        log(f"  {mid}: {neural[i]['n_cells']} cells encoded "
            f"({time.time() - t0:.1f}s)")
    return len(done) >= len(cohort)


def finalise():
    files = sorted(f for f in os.listdir(ENCDIR) if f.endswith(".npz"))
    rec = [np.load(os.path.join(ENCDIR, f)) for f in files]
    area = np.concatenate([r["area"] for r in rec])
    data = {key: np.concatenate([r[key] for r in rec])
            for key in rec[0].files if key != "area"}

    log("=" * 74)
    log("ENCODING — fraction of neurons with significant unique dR2 "
        f"(p<{ALPHA}); mean dR2")
    log(f"{'area':5s} {'n':>5s} | " + " | ".join(
        f"{k:^22s}" for k in LATENTS))
    rows = {}
    for a in PLOT_AREAS:
        sel = area == a
        n = int(sel.sum())
        if n == 0:
            continue
        rows[a] = (n, {})
        cells = []
        for k in LATENTS:
            txt = []
            for mdl in MODELS:
                lab = {"beh": "beh", "beh_spatial": "bsp"}[mdl]
                sig = np.mean(data[f"pval_{mdl}_{k}"][sel] < ALPHA)
                md = np.mean(data[f"dR2_{mdl}_{k}"][sel])
                rows[a][1][(mdl, k)] = (sig, md)
                txt.append(f"{lab}:{sig*100:4.0f}% {md:+.3f}")
            cells.append(" ".join(txt))
        log(f"{a:5s} {n:5d} | " + " | ".join(cells))

    _figure(rows)
    log("DONE")


def _figure(rows):
    areas = list(rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    x = np.arange(len(LATENTS))
    for ax, a in zip(axes.flat, areas):
        n, vals = rows[a]
        beh = [vals[("beh", k)][0] * 100 for k in LATENTS]
        bes = [vals[("beh_spatial", k)][0] * 100 for k in LATENTS]
        ax.bar(x - 0.2, beh, 0.4, color="#3457a6", label="behaviour-controlled")
        ax.bar(x + 0.2, bes, 0.4, color="#b03030", hatch="//",
               label="+ spatial-controlled")
        ax.axhline(ALPHA * 100, c="0.5", ls="--", lw=1, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(LATENTS)
        ax.set_title(f"{a.upper()}  (n = {n})", fontsize=11)
        ax.set_ylabel("% neurons encoding", fontsize=9)
        ax.set_ylim(0, max(100 * 0.05, ax.get_ylim()[1]))
    for j in range(len(areas), axes.size):
        axes.flat[j].axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Per-neuron encoding of RL latents by area — fraction with "
                 "significant unique dR2", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, "fig_neural_encoding.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"wrote {out}")


def main():
    log("synthetic ground-truth check:")
    if not synthetic_check():
        log("SYNTHETIC CHECK FAILED — not touching real data")
        return
    log("synthetic check passed")
    if fit_batch():
        finalise()
    else:
        n = len(os.listdir(ENCDIR))
        log(f"{n}/16 mice encoded — run again")


if __name__ == "__main__":
    main()
