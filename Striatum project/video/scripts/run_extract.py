"""One decode pass per video part -> per-frame motion energy (wheel, mouth,
whiskers ROIs) and signed wheel-texture shift, stamped with VR time.

Usage: python scripts/run_extract.py 1105 [1106 ...]
Writes results/<session>_video.npz: t_vr (s, VR clock), me_<roi>
(mean |frame diff|, grey levels), wheel_dy / wheel_dx (px/frame; forward
running moves the texture up, so dy < 0), wheel_response (phase-correlation
peak, 0-1). Fails if the frame count differs from the VR row count.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from striatum_video.motion import frame_features, iter_gray_frames, probe
from striatum_video.rois import ROIS
from striatum_video.sessions import SESSIONS, frame_times, load_vr, video_parts

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(session):
    vr = load_vr(SESSIONS[session]["vr"])
    parts = video_parts(SESSIONS[session]["video"])
    out = {f"me_{r}": [] for r in ROIS} | {"wheel_dy": [], "wheel_dx": [], "wheel_response": []}
    counts = []
    for p in parts:
        t0 = time.time()
        n_expected = probe(p)[2]
        me, dy, dx, resp = frame_features(iter_gray_frames(p), ROIS, ROIS["wheel"])
        if dy.size != n_expected:
            raise RuntimeError(f"{p.name}: decoded {dy.size} frames, ffprobe counted {n_expected}")
        counts.append(dy.size)
        for r in ROIS:
            out[f"me_{r}"].append(me[r])
        out["wheel_dy"].append(dy)
        out["wheel_dx"].append(dx)
        out["wheel_response"].append(resp)
        print(f"{session} {p.name}: {dy.size} frames in {time.time() - t0:.0f} s", flush=True)
    t_vr = frame_times(vr, sum(counts))
    np.savez(RESULTS / f"{session}_video.npz", t_vr=t_vr, part_counts=counts,
             parts=[p.name for p in parts], **{k: np.concatenate(v) for k, v in out.items()},
             **{f"roi_{r}": [roi.x, roi.y, roi.w, roi.h] for r, roi in ROIS.items()})
    print(f"{session}: {t_vr.size} frames = VR rows -> {session}_video.npz", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
