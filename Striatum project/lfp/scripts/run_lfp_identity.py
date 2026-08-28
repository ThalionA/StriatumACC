"""Verify every LFP file's mouse identity against spiking, instead of trusting the filename.

The 2026-08 download is named by animal, which removes the size-keyed guesswork
of ``lfp_mapping.txt`` -- but a name is an assertion. On 2026-08-11 two animals
turned out to share one byte-identical export, and only physiology caught it.

The test: the 30-90 Hz LFP envelope on a probe tracks the multi-unit rate on
that same probe. Correlate each file's envelope against EVERY animal's MUA over
the same samples and check that the file's own animal wins. A file whose best
match is another animal, or whose own match is at chance, is not what its name says.

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/run_lfp_identity.py [--jobs N]

Writes ``results/lfp_identity_matrix.csv`` and ``results/lfp_identity.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import cohort, config, inventory  # noqa: E402

FS = config.FS
BIN_MS = 100
# Three 10 min windows, all inside behaviour for every animal (the latest VR
# start in the cohort is ~670 s) and clear of the terminal padding. Three rather
# than one because some margins are thin and at least one window (t = 80 min)
# contains a synchronous artefact in several sessions: a verdict is a majority
# across independent windows, not a single reading.
WINDOW_STARTS = (1_200_000, 3_000_000, 4_800_000)
WINDOW_SAMPLES = 600_000
CHANNEL_STEP = 4
# Offsets probed when a file's sample count disagrees with its binned_spikes,
# to tell "truncated export" from "wrong session".
OFFSET_SCAN = (0, 600_000, 1_500_000, 3_000_000)


def mua_rate(mouse_id: int, probe: str, start: int, n: int, ch, *, block: int = 100_000):
    """Multi-unit rate: spikes summed over units, binned to ``BIN_MS``.

    Read in blocks and summed on the fly -- a 555-unit slice is gigabytes if
    materialised, and only the across-unit sum is ever needed.
    """
    path = config.raw_mat(mouse_id, probe, ch)
    if not path.exists():
        return None
    out = np.empty(n, dtype=np.float64)
    with h5py.File(path, "r") as handle:
        dset = handle["binned_spikes"]
        total = int(dset.shape[0])
        if start + n > total:
            return None
        for off in range(0, n, block):
            stop = min(off + block, n)
            out[off:stop] = np.asarray(
                dset[start + off:start + stop, :], dtype=np.float32
            ).sum(axis=1)
    return cohort.bin_mean(out[:, None], int(FS * BIN_MS / 1000)).ravel()


def _mua_job(args):
    mouse_id, probe, start, n, cohort_name = args
    t0 = time.time()
    rate = mua_rate(mouse_id, probe, start, n, config.get_cohort(cohort_name))
    print(f"[mua] {mouse_id}/{probe:9s} "
          f"{'ok' if rate is not None else 'ABSENT':6s} {time.time() - t0:5.1f}s", flush=True)
    return (mouse_id, probe, start), (None if rate is None else rate.astype(np.float32))


def _env_job(args):
    (mouse_id, probe), path, start = args
    t0 = time.time()
    env = inventory.coupling_envelope(Path(path), start, WINDOW_SAMPLES,
                                      channel_step=CHANNEL_STEP, fs=FS, bin_ms=BIN_MS)
    print(f"[env] {mouse_id}/{probe:9s} t={start // FS}s {env.shape} "
          f"{time.time() - t0:5.1f}s", flush=True)
    return (mouse_id, probe, start), env.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--cohort", type=str, default="task",
                        choices=sorted(config.COHORTS))
    args = parser.parse_args()
    ch = config.get_cohort(args.cohort)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    found = cohort.discover_lfp_files(ch.lfp_dir, ch.mouse_ids)
    probes_present = sorted({p for _, p in found})

    # Candidate MUA sources: every task animal that has a raw bundle for a probe
    # that appears in the LFP set -- the controls must include animals whose LFP
    # has not been downloaded, or the test only asks "which of the files I have".
    # Candidates are this cohort's animals: a control file must beat the other
    # controls, which is the mix-up that could actually have happened at download.
    mua_requests = [
        (m, p, w, WINDOW_SAMPLES, args.cohort)
        for p in probes_present for m in ch.mouse_ids for w in WINDOW_STARTS
        if config.raw_mat(m, p, ch).exists()
    ]
    # Extra offsets for animals whose raw session is longer than 8.4 M bins.
    for m in ch.mouse_ids:
        for p in probes_present:
            path = config.raw_mat(m, p, ch)
            if not path.exists():
                continue
            with h5py.File(path, "r") as h:
                total = int(h["binned_spikes"].shape[0])
            if total > 8_400_000:
                for off in OFFSET_SCAN[1:]:
                    mua_requests.append(
                        (m, p, WINDOW_STARTS[0] + off, WINDOW_SAMPLES, args.cohort))

    env_requests = [(k, str(v), w) for k, v in sorted(found.items()) for w in WINDOW_STARTS]

    t0 = time.time()
    with mp.Pool(min(args.jobs, len(env_requests))) as pool:
        envs = dict(pool.map(_env_job, env_requests))
    with mp.Pool(min(args.jobs, len(mua_requests))) as pool:
        muas = dict(pool.map(_mua_job, mua_requests))
    print(f"[identity] features in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Score every (file, candidate) pair per window, then column-normalise so a
    # candidate whose MUA correlates with everything cannot win someone else's row.
    files = sorted(found)
    cands = list(ch.mouse_ids)
    rows, verdicts = [], []
    per_window_norm: dict[int, np.ndarray] = {}
    per_window_raw: dict[int, np.ndarray] = {}

    for w in WINDOW_STARTS:
        mat = np.full((len(files), len(cands)), np.nan)
        for i, (mouse_id, probe) in enumerate(files):
            env = envs.get((mouse_id, probe, w))
            if env is None:
                continue
            for j, cand in enumerate(cands):
                rate = muas.get((cand, probe, w))
                if rate is None:
                    continue
                n = min(env.shape[0], rate.size)
                mat[i, j] = cohort.coupling_score(env[:n], rate[:n])
        per_window_raw[w] = mat
        per_window_norm[w] = cohort.column_normalise(mat)

    for i, (mouse_id, probe) in enumerate(files):
        if mouse_id not in cands:
            continue
        j_own = cands.index(mouse_id)
        wins, window_rows = [], []
        for w in WINDOW_STARTS:
            raw, norm = per_window_raw[w][i], per_window_norm[w][i]
            if not np.isfinite(norm).any():
                continue
            best_j = int(np.nanargmax(norm))
            others = np.delete(norm, j_own)
            wins.append(best_j == j_own)
            window_rows.append({
                "window_start_s": w // FS,
                "own_raw": float(raw[j_own]), "own_norm": float(norm[j_own]),
                "best_match": cands[best_j], "best_norm": float(norm[best_j]),
                "control_p95_norm": float(np.nanpercentile(others, 95)),
                "wins": bool(best_j == j_own),
            })
            for j, cand in enumerate(cands):
                if not np.isfinite(raw[j]):
                    continue
                rows.append({"cohort": args.cohort, "mouse_id": mouse_id, "probe": probe,
                             "window_start_s": w // FS, "candidate": cand,
                             "mean_abs_r": raw[j], "normalised": norm[j],
                             "is_claimed": cand == mouse_id})
        if not wins:
            continue
        n_win = sum(wins)
        verdict = ("CONFIRMED" if n_win == len(wins) else
                   f"MAJORITY({n_win}/{len(wins)})" if n_win > len(wins) / 2 else "FAILED")
        record = {"cohort": args.cohort, "mouse_id": mouse_id,
                  "probe": probe, "verdict": verdict,
                  "windows_won": f"{n_win}/{len(wins)}", "windows": window_rows}
        print(f"[identity] {mouse_id}/{probe:9s} {verdict:14s} " + "  ".join(
            f"t={r['window_start_s']}s own={r['own_norm']:5.2f}x "
            f"(best {r['best_match']} {r['best_norm']:5.2f}x)" for r in window_rows),
            flush=True)

        env0 = envs.get((mouse_id, probe, WINDOW_STARTS[0]))
        offset_scores = {}
        for off in OFFSET_SCAN:
            rate = muas.get((mouse_id, probe, WINDOW_STARTS[0] + off))
            if rate is None or env0 is None:
                continue
            n = min(env0.shape[0], rate.size)
            offset_scores[off] = cohort.coupling_score(env0[:n], rate[:n])
        if len(offset_scores) > 1:
            record["offset_scan_s"] = {o // FS: v for o, v in offset_scores.items()}
            print("           offset scan (raw |r| vs own MUA): " + ", ".join(
                f"{o / FS:+.0f}s={v:.4f}" for o, v in offset_scores.items()), flush=True)
        verdicts.append(record)

    out = config.RESULTS_DIR / f"lfp_identity_matrix_{args.cohort}.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (config.RESULTS_DIR / f"lfp_identity_{args.cohort}.json").write_text(
        json.dumps(verdicts, indent=2, default=str)
    )

    bad = [v for v in verdicts if v["verdict"] != "CONFIRMED"]
    print(f"[identity] {len(verdicts) - len(bad)}/{len(verdicts)} filenames confirmed "
          f"in every window; not fully confirmed: "
          f"{[(v['mouse_id'], v['probe'], v['verdict']) for v in bad]}")


if __name__ == "__main__":
    main()
