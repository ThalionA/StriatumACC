"""Face motion SVD per session (motion_svd.py; Stringer et al. 2019 method).

Usage: python scripts/run_motion_svd.py 1105 1106 1201 1206
Writes results/<s>_motion_svd.npz: pcs (n_frames, N_COMPONENTS) float32,
components (n_blocks, N_COMPONENTS) spatial masks, singular_values, mean,
mask_shape. Fails unless frames = VR rows.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from striatum_video.motion import iter_gray_frames, probe
from striatum_video.motion_svd import binned_motion, fit_motion_svd, project
from striatum_video.rois import FACE_SVD_BIN, FACE_SVD_ROI
from striatum_video.sessions import SESSIONS, frame_times, load_vr, video_parts

N_COMPONENTS = 50
N_SAMPLE = 30000
SEED = 0


def main(session):
    vr = load_vr(SESSIONS[session]["vr"])
    parts = video_parts(SESSIONS[session]["video"])
    counts = [probe(p)[2] for p in parts]
    frame_times(vr, sum(counts))
    chunks = []
    for p, n in zip(parts, counts):
        t0 = time.time()
        m = binned_motion(iter_gray_frames(p), FACE_SVD_ROI, FACE_SVD_BIN)
        if m.shape[0] != n:
            raise RuntimeError(f"{p.name}: {m.shape[0]} frames decoded, {n} expected")
        m[0] = 0  # the first frame of each part has no predecessor
        chunks.append(m)
        print(f"{session} {p.name}: {n} frames in {time.time() - t0:.0f} s", flush=True)
    motion = np.concatenate(chunks)
    del chunks
    svd = fit_motion_svd(motion, N_COMPONENTS, N_SAMPLE, np.random.default_rng(SEED))
    pcs = project(motion, svd).astype(np.float32)
    starts = np.cumsum([0, *counts[:-1]])
    pcs[starts] = np.nan  # part starts: no motion defined
    var = svd["singular_values"] ** 2
    shape = (FACE_SVD_ROI.h // FACE_SVD_BIN, FACE_SVD_ROI.w // FACE_SVD_BIN)
    np.savez(ROOT / "results" / f"{session}_motion_svd.npz", pcs=pcs, components=svd["components"],
             singular_values=svd["singular_values"], mean=svd["mean"], mask_shape=shape)
    print(f"{session}: top-5 components carry {100 * var[:5].sum() / var.sum():.0f}% of the top-{N_COMPONENTS} "
          f"variance -> {session}_motion_svd.npz", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
