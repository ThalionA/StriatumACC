"""Per-frame signals of one session (VR rows = video frames), named as in the
binned product, plus VR position and each frame's Neuropixels-clock time."""

from pathlib import Path

import h5py
import numpy as np

from .sessions import SESSIONS, load_vr

RESULTS = Path(__file__).resolve().parents[2] / "results"
RAWDATA = Path(__file__).resolve().parents[3] / "RawData"
ME_ROIS = ("wheel", "whiskers", "mouth", "spout")


def frame_signals(session, n_svd=10):
    """(vr, t_ms, sig): t_ms = VR_times_synched in ms from the first row;
    sig = vr_speed, lick_frac, x, me_<roi>, svd_1..n_svd (per frame)."""
    vr = load_vr(SESSIONS[session]["vr"])
    v = np.load(RESULTS / f"{session}_video.npz")
    pcs = np.load(RESULTS / f"{session}_motion_svd.npz")["pcs"]
    with h5py.File(RAWDATA / f"{session}_raw.mat", "r") as f:
        ts = np.asarray(f["VR_times_synched"]).ravel()
    sig = {"vr_speed": np.abs(vr["velocity"]), "lick_frac": vr["lick"], "x": vr["x"]}
    sig |= {f"me_{r}": v[f"me_{r}"] for r in ME_ROIS}
    sig |= {f"svd_{k + 1}": pcs[:, k] for k in range(n_svd)}
    return vr, (ts - ts[0]) * 1000, sig
