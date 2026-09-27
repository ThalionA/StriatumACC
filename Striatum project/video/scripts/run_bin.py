"""Video features -> raw trial x 5 cm bin, on the same grid as MATLAB's
spatial_binned_data (checked 2026-09-25: durations reproduce MATLAB's to 2e-15 s,
with an identical NaN pattern, in 1105/1106/1201/1206).

Usage: python scripts/run_bin.py 1105 1106 1201 1206
Reads results/<s>_video.npz; writes results/<s>_binned.npz with
  durations (s), me_wheel / me_mouth / me_whiskers (mean |frame diff| per bin,
  grey levels), vr_speed (mean |VR velocity|, a.u./s), lick_frac (fraction of
  VR frames with the lick sensor on) -- all (n_raw_trials, 50);
  still_me_<roi> (n_raw_trials,): median ME over still, lick-free frames (the
  per-trial noise floor, for drift checks);
  usable (raw 0-based trials: good, engaged <= DP), epoch_<name> (raw 0-based),
  lp, dp -- from striatum_lfp.trials, the project's one trial layer.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "lfp" / "src")]
from striatum_lfp.trials import sessions_for

from striatum_video.binning import bin_session, per_trial_still_median, still_frames
from striatum_video.sessions import SESSIONS, frame_times, load_vr

RAWDATA = ROOT.parent / "RawData"
FEATURES = ("me_wheel", "me_mouth", "me_whiskers", "me_spout")
N_SVD = 10  # face motion SVD components carried into the binned product
STILL_HALF_WINDOW = 15  # frames (~0.5 s) with no VR motion and no lick either side


def vr_times_synched_ms(session):
    """Neuropixels-clock time of every VR row (= every video frame), in ms from
    the first row, exactly as OrganiseStriatumDataIncV1.m builds corrected_vr_time."""
    with h5py.File(RAWDATA / f"{session}_raw.mat", "r") as f:
        ts = np.asarray(f["VR_times_synched"]).ravel()
    return (ts - ts[0]) * 1000


def main(sessions):
    trials = sessions_for("task")
    for s in sessions:
        v = np.load(ROOT / "results" / f"{s}_video.npz")
        vr = load_vr(SESSIONS[s]["vr"])
        frame_times(vr, v["t_vr"].size)  # re-asserts the 1:1 frame lock
        t_ms = vr_times_synched_ms(s)
        if t_ms.size != v["t_vr"].size:
            raise ValueError(f"{s}: VR_times_synched has {t_ms.size} rows, video {v['t_vr'].size} frames")
        feats = {k: v[k] for k in FEATURES} | {"vr_speed": np.abs(vr["velocity"]), "lick_frac": vr["lick"]}
        svd_path = ROOT / "results" / f"{s}_motion_svd.npz"
        if svd_path.exists():  # face motion SVD components (run_motion_svd.py), per frame
            pcs = np.load(svd_path)["pcs"]
            if pcs.shape[0] != v["t_vr"].size:
                raise ValueError(f"{s}: motion SVD has {pcs.shape[0]} frames, video {v['t_vr'].size}")
            feats |= {f"svd_{k + 1}": pcs[:, k] for k in range(N_SVD)}
        binned = bin_session(vr, t_ms, feats)
        still = still_frames(vr["velocity"], vr["lick"], STILL_HALF_WINDOW)
        floors = {f"still_{k}": per_trial_still_median(v[k], still, vr["trial"]) for k in FEATURES}
        st = trials[int(s)]
        if st.n_raw != binned["durations"].shape[0]:
            raise ValueError(f"{s}: {binned['durations'].shape[0]} raw trials here, {st.n_raw} in SessionTrials")
        epochs = {f"epoch_{k}": idx for k, idx in st.epochs().items()}
        np.savez(ROOT / "results" / f"{s}_binned.npz", **binned, **floors, usable=st.usable(),
                 lp=np.nan if st.lp is None else st.lp, dp=st.dp, **epochs)
        n_ep = {k: len(i) for k, i in epochs.items()}
        print(f"{s}: {binned['durations'].shape[0]} raw trials, {st.usable().size} usable, "
              f"LP {st.lp} DP {st.dp}, epochs {n_ep}")


if __name__ == "__main__":
    main(sys.argv[1:])
