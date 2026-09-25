"""Which clock are the visual-probe (probe 2) data on?

In the control cohort the probe-2 bundle's ``VR_times_synched`` differs from the
probe-1 bundle's by 6-48 ms (513) and 116-159 ms (817), drifting over the
session; in the task cohort the two are identical. Python bins probe-2 LFP with
the probe-2 bundle's times, MATLAB crops probe-2 units with probe 1's
(``OrganiseStriatumDataControlIncV1.m:176-191``). Two measurements decide it:

1. **Onset latency.** Probe-2 multi-unit activity locked to corridor onset,
   aligned once with each bundle's VR times. Task 1105 (bundles identical) is the
   reference latency; the right clock reproduces it, the wrong one is shifted by
   the offset.
2. **LFP vs spikes.** Lag of the probe-2 LFP's 100-400 Hz envelope against the
   probe-2 MUA in windows across the session. A constant lag near 0 means the
   LFP export shares probe 2's spike clock; a lag that drifts with the bundle
   offset means it is on probe 1's.

Reads short slices only. Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/audit_probe2_clock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import analysis, bandpower, cohort, config, filtering, results_io  # noqa: E402
from striatum_lfp.figstyle import save_pair  # noqa: E402
from striatum_lfp.reader import DATASET  # noqa: E402

PRE_MS, POST_MS = 300, 500
MAX_ONSETS = 150
N_LAG_WINDOWS = 6
LAG_WINDOW_MS = 60_000
MAX_LAG_MS = 300
ANIMALS = (("task", 1105), ("control", 513), ("control", 515), ("control", 817))


def corridor_onsets_ms(beh: dict) -> np.ndarray:
    """Absolute ms (on this bundle's clock) of the first corridor frame per trial."""
    starts, ends = bandpower.trial_boundaries(beh["trial"])
    out = []
    for s, e in zip(starts, ends):
        hits = np.flatnonzero(beh["world"][s:e + 1] > 6)
        if hits.size:
            out.append(beh["vr_times_s"][s + hits[0]] * 1000.0)
    return np.asarray(out)


def mua_psth(spikes, onsets_ms: np.ndarray) -> np.ndarray:
    """Mean summed-unit count per ms around each onset (1 ms spike bins)."""
    n = spikes.shape[0]
    rows = []
    for t in np.round(onsets_ms).astype(int):
        a, b = t - PRE_MS, t + POST_MS
        if a >= 0 and b <= n:
            rows.append(np.asarray(spikes[a:b]).sum(axis=1))
    return np.mean(rows, axis=0)


def latency_ms(psth: np.ndarray) -> tuple[float, float]:
    """(half-rise latency, peak latency) of the smoothed, baseline-removed PSTH."""
    smooth = uniform_filter1d(psth, 10)
    base = smooth[: PRE_MS - 50].mean()
    post = smooth[PRE_MS: PRE_MS + 300] - base
    peak = int(np.argmax(post))
    half = np.flatnonzero(post[: peak + 1] >= 0.5 * post[peak])
    return float(half[0]) if half.size else np.nan, float(peak)


def lfp_mua_lag(lfp_path: Path, spikes, starts_ms: np.ndarray) -> list[tuple[float, int]]:
    """``[(window start s, lag ms)]``: LFP index + lag = spike bin, per window."""
    sos = filtering.design_band_sos((100.0, 400.0), fs=config.FS)
    out = []
    with h5py.File(lfp_path, "r") as fh:
        volt = fh[DATASET]
        for a in np.round(starts_ms).astype(int):
            lfp = np.asarray(volt[a:a + LAG_WINDOW_MS], dtype=np.float64)
            env = filtering.band_envelope(lfp - np.median(lfp, axis=1, keepdims=True), sos)
            env = uniform_filter1d(env.mean(axis=1), 10)
            lo, hi = a - MAX_LAG_MS, a + LAG_WINDOW_MS + MAX_LAG_MS
            if lo < 0 or hi > spikes.shape[0]:
                continue
            mua = uniform_filter1d(np.asarray(spikes[lo:hi]).sum(axis=1), 10)
            e = (env - env.mean()) / env.std()
            lags = np.arange(-MAX_LAG_MS, MAX_LAG_MS + 1)
            r = [np.dot(e, (m := mua[MAX_LAG_MS + L: MAX_LAG_MS + L + e.size]) - m.mean())
                 / (m.std() * e.size) for L in lags]
            out.append((a / 1000.0, int(lags[int(np.argmax(r))])))
    return out


def main() -> None:
    rows, psths = [], {}
    for cohort_name, mouse in ANIMALS:
        ch = config.get_cohort(cohort_name)
        beh = {p: analysis.read_behaviour(mouse, p, ch) for p in ("striatum", "visual")}
        onsets = {p: corridor_onsets_ms(b) for p, b in beh.items()}
        pick = np.linspace(0, onsets["visual"].size - 1,
                           min(MAX_ONSETS, onsets["visual"].size)).astype(int)
        offset = onsets["visual"] - onsets["striatum"]
        with h5py.File(config.raw_mat(mouse, "visual", ch), "r") as fh:
            spikes = fh["binned_spikes"]
            for clock in ("visual", "striatum"):
                psth = mua_psth(spikes, onsets[clock][pick])
                psths[(mouse, clock)] = psth
                half, peak = latency_ms(psth)
                rows.append({"cohort": cohort_name, "mouse_id": mouse, "clock": clock,
                             "offset_ms_median": float(np.median(offset)),
                             "latency_half_rise_ms": half, "latency_peak_ms": peak,
                             "n_onsets": int(pick.size)})
                print(f"[clock] {mouse} {clock:8s}-bundle times: half-rise {half:5.0f} ms, "
                      f"peak {peak:5.0f} ms  (V1 - probe-1 offset median "
                      f"{np.median(offset):6.1f} ms)", flush=True)
            lfp = cohort.discover_lfp_files(ch.lfp_dir, ch.mouse_ids).get((mouse, "visual"))
            if lfp is not None:
                v = onsets["visual"]
                starts = np.linspace(v[0] + 5_000, v[-1] - LAG_WINDOW_MS - 5_000, N_LAG_WINDOWS)
                lags = lfp_mua_lag(lfp, spikes, starts)
                off_at = np.interp([s * 1000 for s, _ in lags], onsets["visual"], offset)
                for (t_s, lag), o in zip(lags, off_at):
                    rows.append({"cohort": cohort_name, "mouse_id": mouse, "clock": "lfp_vs_mua",
                                 "window_start_s": t_s, "lfp_to_mua_lag_ms": lag,
                                 "offset_ms_here": float(o)})
                print(f"[clock] {mouse} LFP->MUA lag per window (ms): "
                      f"{[lag for _, lag in lags]}  bundle offset there: "
                      f"{[round(float(o)) for o in off_at]}", flush=True)

    out = config.RESULTS_DIR / "lfp_probe2_clock_audit.csv"
    results_io.write_rows(rows, out, tag="clock")

    t = np.arange(-PRE_MS, POST_MS)
    fig, axes = plt.subplots(1, len(ANIMALS), figsize=(3.2 * len(ANIMALS), 3.0), sharex=True)
    for ax, (cohort_name, mouse) in zip(axes, ANIMALS):
        for clock, colour in (("visual", "#1f4e79"), ("striatum", "#e69f00")):
            p = uniform_filter1d(psths[(mouse, clock)], 10)
            ax.plot(t, p, color=colour, lw=1.2, label=f"{clock}-bundle VR times")
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        ax.set_title(f"{mouse} ({cohort_name})", fontsize=9)
        ax.set_xlabel("time from corridor onset (ms)", fontsize=8)
    axes[0].set_ylabel("probe-2 MUA (spikes / ms, summed)", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Probe-2 onset response under each bundle's VR clock", fontsize=10)
    fig.tight_layout()
    save_pair(fig, "lfp_probe2_clock_audit")


if __name__ == "__main__":
    main()
