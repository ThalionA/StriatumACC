"""Where in the image does licking move pixels? Mean |frame difference| images
on lick frames vs lick-free frames, matched within running-speed strata.

Lick frame: VR lick sensor on. Lick-free: no lick within +-LICKFREE_HALF frames.
Strata: quintiles of |VR velocity| over the lick frames. Map per stratum =
mean(lick) - mean(lick-free); the summary map is their lick-count-weighted mean,
so speed-related motion cancels and lick-related motion remains.

Usage: python scripts/run_lick_maps.py 1206 1201
Writes results/<s>_lick_maps.npz (sums, counts, strata edges).
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from striatum_video.motion import diff_maps_by_label, iter_gray_frames, probe
from striatum_video.sessions import SESSIONS, frame_times, load_vr, video_parts

N_STRATA = 5
LICKFREE_HALF = 5


def frame_labels(speed, lick):
    """1..2*N_STRATA: odd = lick frame, even = lick-free, by speed stratum; 0 = skip."""
    lick = lick > 0
    near_lick = np.convolve(lick.astype(float), np.ones(2 * LICKFREE_HALF + 1), mode="same") > 0
    edges = np.quantile(speed[lick], np.linspace(0, 1, N_STRATA + 1))
    stratum = np.clip(np.searchsorted(edges, speed, side="right") - 1, 0, N_STRATA - 1)
    labels = np.zeros(speed.size, int)
    labels[lick] = 2 * stratum[lick] + 1
    free = ~near_lick & (speed >= edges[0]) & (speed <= edges[-1])
    labels[free] = 2 * stratum[free] + 2
    return labels, edges


def main(session):
    vr = load_vr(SESSIONS[session]["vr"])
    parts = video_parts(SESSIONS[session]["video"])
    counts_per_part = [probe(p)[2] for p in parts]
    frame_times(vr, sum(counts_per_part))
    labels, edges = frame_labels(np.abs(vr["velocity"]), vr["lick"])
    sums, counts, start = 0, 0, 0
    for p, n in zip(parts, counts_per_part):
        t0 = time.time()
        s, c = diff_maps_by_label(iter_gray_frames(p), labels[start:start + n], 2 * N_STRATA)
        sums, counts, start = sums + s, counts + c, start + n
        print(f"{session} {p.name}: {n} frames in {time.time() - t0:.0f} s", flush=True)
    np.savez(ROOT / "results" / f"{session}_lick_maps.npz", sums=sums, counts=counts, speed_edges=edges)
    print(f"{session}: lick frames per stratum {counts[0::2].tolist()}, lick-free {counts[1::2].tolist()}", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
