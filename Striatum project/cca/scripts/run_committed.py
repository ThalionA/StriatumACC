"""Run the committed config (config.DEFAULT) for Stage 2 or Stage 3.

Stage 2 (``--stage 2``) runs the surrogate / lagged / IFI analysis and can
compare nulls -- ``--null-type circshift`` (committed default, per-trial bin
circular shift) vs ``--null-type trials`` (trial-permutation, H&H) -- and vary
the epoch width (``--trials-per-epoch``). Saves
results/stage2_committed_<tag>.pkl where <tag> encodes the overrides.

Stage 3 (``--stage 3``) runs the subspace driver (membership, Gini, principal
angles). Stage 3 is null-independent, so a single run serves every Stage 3
figure regardless of the surrogate. Saves results/stage3_committed*.pkl.

Partial CCA -- every other recorded area regressed out of the pair -- is the
committed default (round-14 lock-in); pass ``--plain`` for plain CCA.
``--include-fs`` keeps fast-spiking units. Output filenames carry ``_partial``
/ ``_fsincl`` tags accordingly.

Resumable; --max-seconds chunks the run.

Run:  python scripts/run_committed.py --stage 2          # committed: partial
      python scripts/run_committed.py --stage 2 --plain  # plain comparison
      python scripts/run_committed.py --stage 3
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import functools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_cca import analysis, config, dataio, pipeline, stage3  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=(2, 3), default=2)
    p.add_argument("--null-type", choices=("trials", "circshift"),
                   default="circshift")
    p.add_argument("--trials-per-epoch", type=int, default=0,
                   help="0 = use the config default")
    p.add_argument("--shuffles", type=int, default=250)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--plain", action="store_true",
                   help="plain CCA; the committed default is partial "
                        "(every other recorded area regressed out)")
    p.add_argument("--include-fs", action="store_true",
                   help="include fast-spiking units (committed default excludes)")
    args = p.parse_args()
    args.partial = not args.plain          # partial CCA is the committed default
    return args


def _key(o):
    return (o.animal_id, o.area_x, o.area_y)


def _save(path, done, cfg):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"results": list(done.values()), "cfg": cfg}, fh)
    os.replace(tmp, path)


def _resolve(args):
    """Config, output path and per-pair worker for the requested stage."""
    suffix = "_partial" if args.partial else ""
    fs = "_fsincl" if args.include_fs else ""
    if args.stage == 3:
        # Stage 3 is null-independent: one run serves every figure.
        repl = {}
        if args.trials_per_epoch:
            repl["trials_per_epoch"] = args.trials_per_epoch
        if args.include_fs:
            repl["exclude_fast_spiking"] = False
        cfg = dataclasses.replace(config.DEFAULT, **repl) if repl else config.DEFAULT
        out = config.RESULTS_DIR / f"stage3_committed{fs}{suffix}.pkl"
        return cfg, out, stage3.analyse_subspace

    over = {"null_type": args.null_type, "n_shuffles": args.shuffles}
    tag = args.null_type
    if args.trials_per_epoch:
        over["trials_per_epoch"] = args.trials_per_epoch
        tag += f"_tpe{args.trials_per_epoch}"
    if args.include_fs:
        over["exclude_fast_spiking"] = False
        tag += "_fsincl"
    cfg = dataclasses.replace(config.DEFAULT, **over)
    out = config.RESULTS_DIR / f"stage2_committed_{tag}{suffix}.pkl"
    return cfg, out, analysis.analyse_pair


def main():
    args = parse_args()
    deadline = time.time() + args.max_seconds if args.max_seconds else None
    cfg, out, worker_fn = _resolve(args)
    prep_fn = (pipeline.prepare_pair_partial if args.partial
               else pipeline.prepare_pair)
    print(f"stage {args.stage}; committed config ({pipeline.config_label(cfg)})"
          + ("; PARTIAL (all other recorded areas removed)"
             if args.partial else "")
          + (f"; null={cfg.null_type}; {cfg.n_shuffles} surrogates"
             if args.stage == 2 else "; null-independent"))

    animals = dataio.load_animals()
    entries, _ = dataio.classify_cohort(animals, cfg)
    prepared, skipped = [], []
    for a in animals:
        entry = entries[a.animal_id]
        for ax, ay in config.PAIRS:
            r = prep_fn(a, ax, ay, entry, cfg)
            (prepared if isinstance(r, pipeline.PreparedPair)
             else skipped).append(r)

    done = {}
    if out.exists() and not args.fresh:
        with open(out, "rb") as fh:
            done = {_key(r): r for r in pickle.load(fh)["results"]}
    todo = [p for p in prepared if _key(p) not in done]
    print(f"  {len(prepared)} pairs prepared, {len(done)} done, "
          f"{len(todo)} to do.")

    if todo:
        worker = functools.partial(worker_fn, cfg=cfg)
        t0 = time.time()
        with mp.Pool(args.jobs) as pool:
            for i, res in enumerate(pool.imap_unordered(worker, todo), 1):
                done[_key(res)] = res
                if i % 6 == 0:
                    _save(out, done, cfg)
                    print(f"    {len(done)}/{len(prepared)} "
                          f"({time.time() - t0:.0f}s)")
                if deadline is not None and time.time() > deadline:
                    break
        _save(out, done, cfg)

    complete = len(done) == len(prepared)
    print(f"  {'COMPLETE' if complete else 'INCOMPLETE -- re-run to resume'}"
          f": {out.name}")


if __name__ == "__main__":
    main()
