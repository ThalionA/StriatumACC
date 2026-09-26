"""Gate for each session: is video frame i still VR row i at the end of the
session? Correlates video wheel displacement with VR displacement
(velocity * dt) in 12 blocks, over frames the wheel tracker is confident about.

Usage: python scripts/check_frame_lock.py 1105 1106 1201 1206
Writes results/frame_lock.json and figures/frame_lock.{svg,png}.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from striatum_video.lock import block_lock
from striatum_video.rois import ROIS
from striatum_video.sessions import SESSIONS, load_vr

ROOT = Path(__file__).resolve().parents[1]
MIN_RESPONSE = 0.5   # phase-correlation peak below this = blur/occlusion, shift unreliable
PASS_R = 0.8         # r at zero shift required in every block
PASS_SHIFT = 1       # |best shift| allowed (frames); VR velocity may lag the image by one frame


def session_lock(session):
    v = np.load(ROOT / "results" / f"{session}_video.npz")
    vr = load_vr(SESSIONS[session]["vr"])
    dy = v["wheel_dy"]
    valid = (np.isfinite(dy) & (v["wheel_response"] > MIN_RESPONSE)
             & (np.abs(dy) < 0.45 * ROIS["wheel"].h))
    vr_disp = vr["velocity"] * np.diff(vr["time"], prepend=vr["time"][0])
    out = block_lock(np.nan_to_num(-dy), vr_disp, valid)
    out["frac_valid"] = float(valid.mean())
    out["passes"] = bool(np.all(out["r_zero"] >= PASS_R) and np.all(np.abs(out["best_shift"]) <= PASS_SHIFT))
    return out


def main(sessions):
    res = {s: session_lock(s) for s in sessions}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for s, r in res.items():
        blocks = np.arange(1, r["r_zero"].size + 1)
        ax[0].plot(blocks, r["r_zero"], "o-", label=f"{s} ({'pass' if r['passes'] else 'FAIL'})")
        ax[1].plot(blocks, r["best_shift"], "o-", label=s)
    ax[0].axhline(PASS_R, color="k", lw=0.8, ls="--")
    ax[0].set(xlabel="session block (1/12 of frames)", ylabel="Pearson r at zero frame shift",
              title="Video wheel shift vs VR displacement", ylim=(0, 1))
    ax[1].axhspan(-PASS_SHIFT, PASS_SHIFT, color="0.9")
    ax[1].set(xlabel="session block (1/12 of frames)", ylabel="best frame shift (frames)",
              title="Best video→VR frame shift (0 = frame i is VR row i)", ylim=(-8.5, 8.5))
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle(f"Frame lock: confident wheel frames only (phase-corr response > {MIN_RESPONSE})")
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(ROOT / "figures" / f"frame_lock.{ext}", dpi=130)
    summary = {s: {"passes": r["passes"], "frac_valid": r["frac_valid"],
                   "r_zero_min": float(np.min(r["r_zero"])),
                   "best_shift": r["best_shift"].tolist()} for s, r in res.items()}
    (ROOT / "results" / "frame_lock.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
