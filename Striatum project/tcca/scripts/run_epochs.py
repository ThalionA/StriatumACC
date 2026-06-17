"""Temporal CCA on the naive / intermediate / expert learning epochs.

For each learner and area pair, fit the full ``subspace_window`` readout on the
running, in-corridor bins of each epoch (``dataio.epoch_windows`` from the
animal's learning point), partialling out every other recorded area. Writes
per (animal, pair, epoch) metrics, per-dim rows (dims-as-n), per-neuron weight
contributions (Gini CDF), and cross-epoch rotation/Jaccard. Learners only.

Run:  python3 scripts/run_epochs.py [--group task] [--bin-ms 10] [--smooth-ms 2.5]
                                     [--max-lag 5] [--include-fs] [--pairs A-B,C-D]
                                     [--limit N] [--tag _bin10]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from striatum_tcca import config, dataio, runner  # noqa: E402

K = 20                       # PCA components per area (report Section 2.5: epochs)
SAT = 0.99                   # drop saturated (degenerate) cells
MIN_CELL_BINS = K * 30       # >= 600 running bins per (pair, epoch) cell
TRANSITIONS = [("naive", "intermediate"), ("intermediate", "expert"),
               ("naive", "expert")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="task", choices=["task", "control"])
    p.add_argument("--bin-ms", type=int, default=25,
                   help="25 = magnitude reference (report/Tom default); "
                        "10 = fine/directionality view (noisier per-cell cc1)")
    p.add_argument("--smooth-ms", type=float, default=2.5)
    p.add_argument("--max-lag", type=int, default=0,
                   help="IFI lag half-width in bins; 0 = auto (+/-50 ms headline)")
    p.add_argument("--include-fs", action="store_true")
    p.add_argument("--pairs", default="", help="comma list A-B,C-D to restrict pairs")
    p.add_argument("--limit", type=int, default=0, help="cap number of learners (smoke)")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dataclasses.replace(config.DEFAULT, temporal_bin_ms=args.bin_ms,
                              gaussian_sd_ms=args.smooth_ms,
                              exclude_fast_spiking=not args.include_fs)
    suffix = ("_fsincl" if args.include_fs else "") + args.tag
    pairs = ([tuple(p.split("-")) for p in args.pairs.split(",")]
             if args.pairs else list(config.PAIRS))
    # IFI integration half-width: report headline is +/-50 ms, so scale to bins.
    max_lag = args.max_lag or max(2, round(50 / args.bin_ms))

    animals = dataio.load_animals(group=args.group)
    entries, _ = dataio.classify_cohort(animals, cfg)
    learners = [a for a in animals if entries[a.animal_id].role == "learner"]
    if args.limit:
        learners = learners[: args.limit]
    print(f"{args.group} | bin={args.bin_ms}ms sigma={args.smooth_ms} K={K} "
          f"max_lag={max_lag} shuffles={config.SURROGATE_SHUFFLES} "
          f"FS={'incl' if args.include_fs else 'excl'} | {len(learners)} learners\n")

    metrics, dims, weights, cross = [], [], [], []
    for a in learners:
        e = entries[a.animal_id]
        streams = dataio.load_temporal_streams(a, cfg, period="engaged")
        run = dataio.running_mask(streams, cfg)
        trial_ids = streams.trial_idx[run]
        uniq = np.unique(trial_ids)
        ew = dataio.epoch_windows(int(e.lp), uniq.size, cfg)
        if ew is None:
            print(f"  animal {a.animal_id}: lp={e.lp}, {uniq.size} run-trials -> "
                  f"epochs don't fit (skip)")
            continue
        present = runner.build_present(streams, run, a, cfg)
        n_rows = 0
        for ax, ay in pairs:
            if ax not in present or ay not in present:
                continue
            ws_by_epoch = {}
            for epoch in config.EPOCH_NAMES:
                pos = ew[epoch]
                pos = pos[(pos >= 0) & (pos < uniq.size)]
                ws = runner.fit_window(
                    present, trial_ids, ax, ay, uniq[pos], cfg,
                    k=K, max_lag=max_lag, n_shuffles=config.SURROGATE_SHUFFLES,
                    n_folds=cfg.n_folds, min_cell_bins=MIN_CELL_BINS)
                if ws is None:
                    continue
                cc1 = float(ws.cc[0]) if ws.cc.size else float("nan")
                if not np.isfinite(cc1) or cc1 >= SAT:
                    continue
                ws_by_epoch[epoch] = (ws, uniq[pos])
                n_bins = int(np.isin(trial_ids, uniq[pos]).sum())
                metrics.append({
                    "animal": a.animal_id, "role": e.role, "lp": e.lp,
                    "pair": f"{ax}-{ay}", "epoch": epoch, "n_bins": n_bins,
                    "cc1": round(cc1, 4), "n_sig": ws.n_sig,
                    "mi_sig": round(ws.mi_sig, 4), "ifi": round(ws.ifi, 4),
                    "optimal_lag": ws.optimal_lag,
                    "gini_x": round(ws.gini_x, 4), "gini_y": round(ws.gini_y, 4),
                    "sh_x_cc1": round(ws.split_half_x_cc1, 2),
                    "sh_y_cc1": round(ws.split_half_y_cc1, 2)})
                n_rows += 1
                for d in range(int(ws.cc.size)):
                    dims.append({
                        "animal": a.animal_id, "pair": f"{ax}-{ay}", "epoch": epoch,
                        "dim": d + 1, "peak_cc": round(float(ws.cc[d]), 4),
                        "sig": int(bool(ws.sig_mask[d])) if d < ws.sig_mask.size else 0,
                        "ifi": round(float(ws.ifi_per_dim[d]), 4) if d < ws.ifi_per_dim.size else "",
                        "lag": int(ws.lag_per_dim[d]) if d < ws.lag_per_dim.size else ""})
                for area, W in ((ax, ws.weights_x), (ay, ws.weights_y)):
                    contrib = np.linalg.norm(np.atleast_2d(W), axis=1)
                    for u, c in enumerate(contrib):
                        weights.append({"animal": a.animal_id, "pair": f"{ax}-{ay}",
                                        "epoch": epoch, "area": area, "unit": u,
                                        "contrib": round(float(c), 6)})
            for ea, eb in TRANSITIONS:
                if ea in ws_by_epoch and eb in ws_by_epoch:
                    cw = runner.cross_window(ws_by_epoch[ea][0], ws_by_epoch[eb][0])
                    cross.append({"animal": a.animal_id, "pair": f"{ax}-{ay}",
                                  "transition": f"{ea}->{eb}", **dataclasses.asdict(cw)})
        print(f"  animal {a.animal_id}: lp={e.lp} -> {n_rows} cells")

    _write(config.RESULTS_DIR / f"epoch_metrics{suffix}.csv", metrics)
    _write(config.RESULTS_DIR / f"epoch_dims{suffix}.csv", dims)
    _write(config.RESULTS_DIR / f"epoch_weights{suffix}.csv", weights)
    _write(config.RESULTS_DIR / f"epoch_cross{suffix}.csv",
           [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
            for r in cross])
    print(f"\nwrote epoch_metrics{suffix}.csv ({len(metrics)} cells), "
          f"epoch_dims ({len(dims)}), epoch_weights ({len(weights)}), "
          f"epoch_cross ({len(cross)})")


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
