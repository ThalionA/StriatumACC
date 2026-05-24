"""Stage 3 driver for the round-8 sweep.

Communication-subspace membership and principal-angle reorientation for every
config in a named sweep. No surrogate loop, so it runs fast (a few seconds per
config); resumable -- configs whose results/stage3_<tag>.pkl already exists are
skipped unless ``--fresh``.

Run:  python scripts/run_stage3.py --sweep spatial [--fresh]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_cca import config, dataio, pipeline, stage3, sweep  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", choices=("spatial", "temporal", "all"),
                   default="spatial")
    p.add_argument("--fresh", action="store_true",
                   help="recompute configs even if a result file exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid = sweep.build_sweep(args.sweep)
    print(f"Sweep '{args.sweep}': Stage 3 for {len(grid)} configs.\n")
    animals_by_path: dict = {}
    for tag, data_path, cfg in grid:
        out = config.RESULTS_DIR / f"stage3_{tag}.pkl"
        if out.exists() and not args.fresh:
            print(f"  skip '{tag}' (done)")
            continue
        if data_path not in animals_by_path:
            animals_by_path.clear()                    # 1-slot -- bound memory
            animals_by_path[data_path] = dataio.load_animals(data_path)
        run_variant(animals_by_path[data_path], cfg, tag)
    print("\nStage 3 sweep done.")


def run_variant(animals, cfg, tag: str) -> None:
    entries, _ = dataio.classify_cohort(animals, cfg)
    results: list[stage3.PairSubspace] = []
    t0 = time.time()
    for animal in animals:
        entry = entries[animal.animal_id]
        for area_x, area_y in config.PAIRS:
            prepared = pipeline.prepare_pair(animal, area_x, area_y, entry, cfg)
            if isinstance(prepared, pipeline.PreparedPair):
                results.append(stage3.analyse_subspace(prepared, cfg))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"stage3_{tag}.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"results": results, "cfg": cfg}, fh)
    print(f"  [{tag}] ({pipeline.config_label(cfg)}) {len(results)} pairs "
          f"in {time.time() - t0:.1f}s -> {out.name}")


if __name__ == "__main__":
    main()
