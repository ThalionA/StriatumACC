"""Does shared movement drive the cross-area CCA? Held-out CC1 per epoch for
plain CCA vs CCA with movement regressed out of each area (VR only; VR + video)
vs a shifted-movement control, in the spatial (5 cm) and temporal (20 ms) arms.

The control for each confound is a set of CONTROL_FRACTIONS circular shifts of
that same confound (same dimensionality and statistics, misaligned): one epoch's
held-out CC1 moves by ~+-0.05 when anything is regressed out (a single-shift
control in the first run raised spatial CC1 by a median 0.027), so the effect
is judged against each animal-pair-epoch's own control distribution.

Uses the committed cca pipeline (config.DEFAULT; plain, not area-partial),
the cca cohort entries, and striatum_cca.pipeline.prepare_pair_confounded.

Usage: python scripts/run_cca_movement.py 1105 1106 1201 1206
Writes results/cca_movement.npz (one row per arm x animal x pair x epoch x
variant) and prints a summary.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "cca" / "src"), str(ROOT.parent / "lfp" / "src")]
from striatum_cca import config, dataio, pipeline
from striatum_lfp import config as lfp_config

from striatum_video.confounds import shift_confound, spatial_confound, temporal_confound
from striatum_video.signals import frame_signals

VR = ("vr_speed", "lick_frac")
VIDEO = ("me_wheel", "me_whiskers", "me_mouth", "me_spout", *(f"svd_{k}" for k in range(1, 11)))
TEMPORAL_LAGS_MS = (-200, -100, 0, 100, 200)
CONTROL_FRACTIONS = tuple(np.linspace(0.1, 0.9, 10))  # circular shifts of the confound, as fractions of the session


def confounds(session, animal, cfg):
    """{'vr': ..., 'vr_video': ...} in the arm's layout, over the cca usable trials."""
    if cfg.bin_mode == "spatial":
        b = np.load(ROOT / "results" / f"{session}_binned.npz")
        n_use = dataio.n_usable_trials(animal)
        return {"vr": spatial_confound(b, VR)[:n_use], "vr_video": spatial_confound(b, VR + VIDEO)[:n_use]}
    vr, t_ms, sig = frame_signals(session)
    lengths = [t.shape[0] for t in dataio.area_tensor(animal, "DMS", cfg)[0]]
    make = lambda names: temporal_confound(vr, t_ms, sig, names, lengths, cfg.temporal_bin_ms, TEMPORAL_LAGS_MS)
    return {"vr": make(VR), "vr_video": make(VR + VIDEO)}


def held_out_cc1(prepared, cfg):
    """Held-out CC1 per epoch; confounded pairs are partialled inside the folds
    (pipeline.held_out_cca) -- partialling first inflated held-out CC."""
    if isinstance(prepared, pipeline.SkippedPair):
        return {e: np.nan for e in config.EPOCH_NAMES}, np.nan
    return ({e: float(pipeline.held_out_cca(prepared, e, cfg).held_out_r[0]) for e in config.EPOCH_NAMES},
            prepared.k)


def main(sessions):
    animals = dataio.load_animals()
    ids = list(lfp_config.TASK.mouse_ids)
    entries, _ = dataio.classify_cohort(animals, config.DEFAULT)
    rows = []
    for arm, cfg in (("spatial", config.DEFAULT), ("temporal", replace(config.DEFAULT, bin_mode="temporal"))):
        for s in sessions:
            animal = animals[ids.index(int(s))]
            entry = entries[animal.animal_id]
            conf = confounds(s, animal, cfg)
            for base in ("vr", "vr_video"):
                for j, frac in enumerate(CONTROL_FRACTIONS):
                    conf[f"ctrl_{base}_{j}"] = shift_confound(conf[base], frac)
            for ax, ay in config.PAIRS:
                if min(len(dataio.select_units(animal, a, cfg)) for a in (ax, ay)) < cfg.min_units:
                    continue
                t0 = time.time()
                variants = {"plain": pipeline.prepare_pair(animal, ax, ay, entry, cfg)}
                for name, c in conf.items():
                    variants[name] = pipeline.prepare_pair_confounded(animal, ax, ay, entry, c, cfg)
                for name, prep in variants.items():
                    cc, k = held_out_cc1(prep, cfg)
                    for e, v in cc.items():
                        rows.append((arm, int(s), entry.role, f"{ax}-{ay}", e, name, v, k))
                print(f"{arm} {s} {ax}-{ay}: {len(variants)} variants in {time.time() - t0:.0f} s", flush=True)
    cols = list(zip(*rows))
    table = dict(zip(("arm", "session", "role", "pair", "epoch", "variant", "cc1", "k"), map(np.array, cols)))
    np.savez(ROOT / "results" / "cca_movement.npz", **table)
    summarise(table)


def effects(table, arm, base):
    """Per animal-pair-epoch: real CC1 with `base` removed vs its own control
    distribution (the same confound circularly shifted). Returns arrays of
    (plain, real, control median, fraction of controls <= real)."""
    m = table["arm"] == arm
    keys = sorted(set(zip(table["session"][m], table["pair"][m], table["epoch"][m])))
    out = []
    for s, p, e in keys:
        sel = m & (table["session"] == s) & (table["pair"] == p) & (table["epoch"] == e)
        v = dict(zip(table["variant"][sel], table["cc1"][sel]))
        ctrl = np.array([v[f"ctrl_{base}_{j}"] for j in range(len(CONTROL_FRACTIONS))])
        out.append((v["plain"], v[base], np.nanmedian(ctrl), np.mean(ctrl <= v[base])))
    return np.array(out)


def summarise(table):
    for arm in ("spatial", "temporal"):
        for base in ("vr", "vr_video"):
            eff = effects(table, arm, base)
            ok = np.isfinite(eff).all(1)
            eff = eff[ok]
            drop = eff[:, 2] - eff[:, 0 + 1]
            below_all = np.mean(eff[:, 3] == 0)
            print(f"== {arm} / {base}: {eff.shape[0]} animal-pair-epochs | median plain CC1 {np.median(eff[:, 0]):.3f}, "
                  f"median control CC1 {np.median(eff[:, 2]):.3f}")
            print(f"   control-median - real: median {np.median(drop):+.4f}; >= 0.05 in {np.mean(drop >= 0.05):.2f}; "
                  f"real below ALL {len(CONTROL_FRACTIONS)} controls in {below_all:.2f} "
                  f"(chance {1 / (len(CONTROL_FRACTIONS) + 1):.2f})")


if __name__ == "__main__":
    main(sys.argv[1:])
