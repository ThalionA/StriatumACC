"""Partial CCA -- committed config, all 10 pairs, conditioned on the rest.

For every animal, prepares all five areas; for each pair X-Y it compares the
plain held-out CC1 with the partial held-out CC1 after regressing *every other
area the animal recorded* (concatenated) out of both X and Y. A large
plain->partial drop means the pair's apparent coupling is largely carried by
the other areas; little drop means the coupling is direct.

The conditioning set is whatever other areas that animal has -- its size
varies by animal and is recorded per cell (`n_control`). Cells need at least
one other area to condition on. Plain CC1 uses the per-area `prepare_area`
scores (so it can differ slightly from the Stage-2 CC, which picks a symmetric
k per pair); plain vs partial is internally consistent here. The confound
regression is fitted on all samples (not cross-validated) -- as in partial.py.

Resumable by animal; --max-seconds chunks the run. Saves
results/partial_committed.pkl.

Run:  python scripts/run_partial.py [--max-seconds N]
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_cca import config, core, dataio, partial, pipeline  # noqa: E402

EPOCHS = config.EPOCH_NAMES
OUT = config.RESULTS_DIR / "partial_committed.pkl"


def _save(rows, done_animals):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"rows": rows, "done": sorted(done_animals),
                     "cfg": config.DEFAULT}, fh)
    os.replace(tmp, OUT)


def animal_rows(animal, entry, cfg):
    """Plain vs partial held-out CC1 for every pair x epoch of one animal."""
    prepped = {a: pipeline.prepare_area(animal, a, entry, cfg)
               for a in config.AREAS}
    rows = []
    for area_x, area_y in config.PAIRS:
        if prepped[area_x] is None or prepped[area_y] is None:
            continue
        others = [a for a in config.AREAS
                  if a not in (area_x, area_y) and prepped[a] is not None]
        if not others:
            continue                       # nothing to condition on
        for epoch in EPOCHS:
            sx, sy = prepped[area_x][epoch], prepped[area_y][epoch]
            sz = np.concatenate([prepped[a][epoch] for a in others], axis=-1)
            plain = core.cca_cv(sx, sy, cfg).held_out_r[0]
            part = partial.partial_cca_cv(sx, sy, sz, cfg).held_out_r[0]
            rows.append(dict(animal=animal.animal_id, role=entry.role,
                             pair=f"{area_x}-{area_y}", epoch=epoch,
                             plain_cc1=float(plain), partial_cc1=float(part),
                             n_control=len(others),
                             controls="+".join(others)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()
    deadline = time.time() + args.max_seconds if args.max_seconds else None
    cfg = config.DEFAULT

    animals = dataio.load_animals()
    entries, _ = dataio.classify_cohort(animals, cfg)

    rows, done = [], set()
    if OUT.exists() and not args.fresh:
        with open(OUT, "rb") as fh:
            blob = pickle.load(fh)
        rows, done = blob["rows"], set(blob.get("done", []))
    todo = [a for a in animals if a.animal_id not in done]
    print(f"partial CCA (committed config); {len(done)}/{len(animals)} "
          f"animals done, {len(todo)} to do.")

    for animal in todo:
        rows.extend(animal_rows(animal, entries[animal.animal_id], cfg))
        done.add(animal.animal_id)
        _save(rows, done)
        print(f"  animal {animal.animal_id} done "
              f"({len(done)}/{len(animals)}); {len(rows)} cells total")
        if deadline is not None and time.time() > deadline:
            print("  stopped at deadline -- re-run to resume")
            return

    print(f"COMPLETE: {OUT.name} -- {len(rows)} (pair x epoch x animal) cells "
          f"from {len({r['animal'] for r in rows})} animals")


if __name__ == "__main__":
    main()
