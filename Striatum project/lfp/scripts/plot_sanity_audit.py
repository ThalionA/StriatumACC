"""Publication-ready visualisation of the cached full-session LFP audit."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config  # noqa: E402
from striatum_lfp.reader import DATASET  # noqa: E402
from striatum_lfp.sanity import (  # noqa: E402
    boolean_runs,
    distance_to_nearest,
    median_welch_psd,
    robust_synchronous_peak_sample,
    signed_distance_to_nearest,
)

FS = config.FS
CHANNELS = np.array([10, 90, 160, 225, 300, 360])
COLOURS = {
    "ordinary": "#4C72B0",
    "event": "#DD8452",
    "zero": "#C44E52",
    "dark": "#7F7F7F",
    "outside": "#55A868",
}


def load_record(mouse_id):
    cache = np.load(config.RESULTS_DIR / f"sanity_audit_{mouse_id}.npz")
    with h5py.File(config.RAW_MAT[mouse_id], "r") as handle:
        frame_s = np.asarray(handle["VR_times_synched"]).ravel()
        vr = np.asarray(handle["VR_data"])
    transitions = np.flatnonzero(np.diff(vr[:, 8]) != 0) + 1
    return {
        "mouse": mouse_id,
        **{key: cache[key] for key in cache.files},
        "frame_s": frame_s,
        "sync_transition_s": frame_s[transitions],
    }


def event_mask(record):
    return np.isfinite(record["artefact_z"]) & (record["artefact_z"] > 5)


def choose_indices(mask, count):
    indices = np.flatnonzero(mask)
    if indices.size <= count:
        return indices
    return indices[np.linspace(0, indices.size - 1, count).astype(int)]


def read_centred_windows(mouse_id, one_second_indices, count, seconds=4):
    selected = choose_indices(one_second_indices, count)
    path = config.LFP_DIR / config.FILE_BY_MOUSE[mouse_id]
    windows = []
    with h5py.File(path, "r") as handle:
        dset = handle[DATASET]
        duration = seconds * FS
        for index in selected:
            centre = int(index * FS + FS / 2)
            start = max(0, min(centre - duration // 2, dset.shape[0] - duration))
            windows.append(np.asarray(dset[start : start + duration, CHANNELS], dtype=float))
    return np.stack(windows)


def normalised_psd(windows):
    frequencies, psd = median_welch_psd(windows, fs_hz=FS)
    band = (frequencies >= 1) & (frequencies <= 200)
    area = np.trapz(psd[band], frequencies[band])
    return frequencies, psd / area


def representative_window(record, mode):
    event = event_mask(record)
    zero = record["window_exact_zero_fraction"] >= 0.99
    corridor = record["world"] == 6
    if mode == "ordinary":
        candidates = corridor & ~zero & np.isfinite(record["artefact_z"]) & (np.abs(record["artefact_z"]) < 1)
        target = np.nanmedian(record["window_median_channel_rms"][candidates])
        idx = np.flatnonzero(candidates)[np.argmin(
            np.abs(record["window_median_channel_rms"][candidates] - target)
        )]
    else:
        candidates = event
        target = np.nanmedian(record["window_median_channel_rms"][candidates])
        idx = np.flatnonzero(candidates)[np.argmin(
            np.abs(record["window_median_channel_rms"][candidates] - target)
        )]
    path = config.LFP_DIR / config.FILE_BY_MOUSE[record["mouse"]]
    with h5py.File(path, "r") as handle:
        dset = handle[DATASET]
        centre = idx * FS + FS // 2
        start = max(0, min(centre - 2 * FS, dset.shape[0] - 4 * FS))
        voltage = np.asarray(dset[start : start + 4 * FS, CHANNELS], dtype=float)
    return start, voltage


def build_spectral_cache(records):
    for record in records:
        event = event_mask(record)
        zero = record["window_exact_zero_fraction"] >= 0.99
        corridor = record["world"] == 6
        ordinary = corridor & ~zero & np.isfinite(record["artefact_z"]) & (np.abs(record["artefact_z"]) < 1)
        ordinary_windows = read_centred_windows(record["mouse"], ordinary, 40)
        event_windows = read_centred_windows(record["mouse"], event, 40)
        record["ordinary_psd"] = normalised_psd(ordinary_windows)
        record["event_psd"] = normalised_psd(event_windows)
        record["ordinary_example"] = representative_window(record, "ordinary")
        record["event_example"] = representative_window(record, "event")


def save_figure_source_data(records):
    """Persist the exact targeted raw examples and PSD curves used in figures."""
    output = {
        "selected_channels_zero_based": CHANNELS,
        "sampling_rate_hz": np.asarray(FS),
    }
    for record in records:
        mouse = record["mouse"]
        ordinary_f, ordinary_psd = record["ordinary_psd"]
        event_f, event_psd = record["event_psd"]
        ordinary_start, ordinary_voltage = record["ordinary_example"]
        event_start, event_voltage = record["event_example"]
        output[f"{mouse}_ordinary_frequency_hz"] = ordinary_f
        output[f"{mouse}_ordinary_normalised_psd"] = ordinary_psd
        output[f"{mouse}_event_frequency_hz"] = event_f
        output[f"{mouse}_event_normalised_psd"] = event_psd
        output[f"{mouse}_ordinary_example_start_sample"] = np.asarray(
            ordinary_start
        )
        output[f"{mouse}_ordinary_example_voltage"] = ordinary_voltage
        output[f"{mouse}_event_example_start_sample"] = np.asarray(event_start)
        output[f"{mouse}_event_example_voltage"] = event_voltage
    np.savez_compressed(
        config.RESULTS_DIR / "sanity_figure_examples_and_psd.npz", **output
    )


def plot_overview(records):
    fig, axes = plt.subplots(4, 4, figsize=(15, 11), constrained_layout=True)
    for row, record in enumerate(records):
        mouse = record["mouse"]
        seconds = record["window_start_sample"] / FS
        minutes = seconds / 60
        rms = record["window_median_channel_rms"]
        log_rms = np.log10(np.maximum(rms, np.finfo(float).tiny))
        zero = record["window_exact_zero_fraction"] >= 0.99
        event = event_mask(record)
        ordinary = ~zero & ~event
        corridor = record["world"] == 6
        dark = np.isfinite(record["world"]) & ~corridor
        outside = ~np.isfinite(record["world"])

        ax = axes[row, 0]
        ax.scatter(minutes[ordinary], log_rms[ordinary], s=1.2, color=COLOURS["ordinary"], rasterized=True)
        ax.scatter(minutes[event], log_rms[event], s=3, color=COLOURS["event"], label="periodic high-amplitude mode")
        floor = np.nanpercentile(log_rms[ordinary], 0.5) - 0.25
        ax.scatter(minutes[zero], np.full(zero.sum(), floor), s=3, color=COLOURS["zero"], label="≥99% exact-zero window")
        ax.set_ylabel(f"{mouse}\nlog10 RMS\n(stored units)")
        if row == 0:
            ax.set_title("A. Whole-session amplitude")
            ax.legend(fontsize=6, loc="best")

        ax = axes[row, 1]
        # Exclude the separated high-amplitude mode: this panel describes the
        # ordinary voltage distribution, not state occupancy of the events.
        valid_corridor = corridor & ~zero & ~event
        valid_dark = dark & ~zero & ~event
        lo, hi = np.nanpercentile(log_rms[~zero], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        ax.hist(log_rms[valid_dark], bins=bins, density=True, histtype="step",
                color=COLOURS["dark"], label="dark/ITI")
        ax.hist(log_rms[valid_corridor], bins=bins, density=True, histtype="step",
                color=COLOURS["ordinary"], label="corridor")
        if row == 0:
            ax.set_title("B. Ordinary amplitude by state")
            ax.legend(fontsize=6)
        ax.set_ylabel("density")

        ax = axes[row, 2]
        groups = [corridor, dark, outside]
        values = [100 * zero[group].mean() if group.any() else np.nan for group in groups]
        ax.bar(["corridor", "dark/ITI", "outside\nbehaviour"], values,
               color=[COLOURS["ordinary"], COLOURS["dark"], COLOURS["outside"]])
        ax.set_ylim(0, max(5, np.nanmax(values) * 1.15))
        ax.set_ylabel("≥99% exact-zero windows (%)")
        if row == 0:
            ax.set_title("C. Zero gaps by session state")

        ax = axes[row, 3]
        f, ordinary_psd = record["ordinary_psd"]
        _, high_psd = record["event_psd"]
        band = (f >= 1) & (f <= 200)
        ax.loglog(f[band], ordinary_psd[band], color=COLOURS["ordinary"], lw=1,
                  label="40 ordinary corridor windows")
        ax.loglog(f[band], high_psd[band], color=COLOURS["event"], lw=1,
                  label="40 high-amplitude events")
        for low, high, colour in [(4, 8, "#55A868"), (15, 30, "#4C72B0"), (30, 80, "#8172B3")]:
            ax.axvspan(low, high, color=colour, alpha=0.06)
        ax.set_ylabel("normalised PSD (shape only)")
        if row == 0:
            ax.set_title("D. Robust spectral shape")
            ax.legend(fontsize=6)

        for col in range(4):
            axes[row, col].spines[["top", "right"]].set_visible(False)
            if row == 3:
                axes[row, col].set_xlabel(
                    ["Time (min)", "log10 RMS (stored units)", "Session state", "Frequency (Hz)"][col]
                )
    fig.suptitle("LFP integrity audit: continuous task voltage plus periodic high-amplitude events", fontsize=14)
    save_pair(fig, "sanity_audit_overview_v2")


def plot_raw_examples(records):
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), constrained_layout=True)
    for row, record in enumerate(records):
        for col, (key, title) in enumerate((("ordinary_example", "Ordinary corridor"),
                                            ("event_example", "Periodic high-amplitude event"))):
            _, voltage = record[key]
            centred = voltage - np.median(voltage, axis=0, keepdims=True)
            scale = 1.4826 * np.median(np.abs(centred), axis=0)
            scale[scale == 0] = 1.0
            standardised = centred / scale
            for channel_i, channel in enumerate(CHANNELS):
                axes[row, col].plot(np.arange(len(voltage)) / FS,
                                    standardised[:, channel_i] + channel_i * 8,
                                    lw=0.35)
            axes[row, col].set_ylabel(f"{record['mouse']}\nrobust-SD offset")
            axes[row, col].set_ylim(-8, len(CHANNELS) * 8)
            axes[row, col].spines[["top", "right"]].set_visible(False)
            if row == 0:
                axes[row, col].set_title(f"{title} (4 s; shape standardised)")
            if row == 3:
                axes[row, col].set_xlabel("Time (s)")
    fig.suptitle("Raw voltage morphology across six depths", fontsize=14)
    save_pair(fig, "sanity_audit_raw_examples_v2")


def timing_summary_and_plot(records):
    rows = []
    fig, axes = plt.subplots(4, 1, figsize=(15, 9), constrained_layout=True)
    for ax, record in zip(axes, records):
        event = event_mask(record)
        runs = boolean_runs(event)
        window_starts_s = record["window_start_sample"] / FS
        starts_s = window_starts_s[runs[:, 0]]
        intervals = np.diff(starts_s)
        sync_s = record["sync_transition_s"]
        start_distances = distance_to_nearest(starts_s, sync_s)
        start_signed_offsets = signed_distance_to_nearest(starts_s, sync_s)

        # Resolve event timing on the nominal 1 ms export grid. The peak score
        # is a robust median across 24 depth-spanning channels, so one electrode
        # cannot set the event time. This is timing evidence, not an alignment
        # proof: the export's absolute offset remains undocumented.
        path = config.LFP_DIR / config.FILE_BY_MOUSE[record["mouse"]]
        peak_times_s = []
        timing_channels = np.arange(0, config.N_CHANNELS, 16)
        with h5py.File(path, "r") as handle:
            dataset = handle[DATASET]
            for run_start, run_stop in runs:
                sample_start = int(record["window_start_sample"][run_start])
                sample_stop = int(
                    record["window_start_sample"][run_stop] + FS
                )
                voltage = np.asarray(
                    dataset[sample_start:sample_stop, timing_channels], dtype=float
                )
                peak_times_s.append(
                    (sample_start + robust_synchronous_peak_sample(voltage)) / FS
                )
        peak_times_s = np.asarray(peak_times_s)
        peak_signed_offsets = signed_distance_to_nearest(peak_times_s, sync_s)
        corridor = record["world"] == 6
        dark = np.isfinite(record["world"]) & ~corridor
        outside = ~np.isfinite(record["world"])
        rows.append({
            "mouse_id": record["mouse"],
            "high_amplitude_one_second_bin_count": int(event.sum()),
            "event_run_count": len(runs),
            "median_event_interval_s": float(np.median(intervals)),
            "min_event_interval_s": float(np.min(intervals)),
            "max_event_interval_s": float(np.max(intervals)),
            "median_abs_event_start_to_vr_sync_s": float(np.median(start_distances)),
            "median_signed_event_start_to_vr_sync_s": float(np.median(start_signed_offsets)),
            "median_signed_voltage_peak_to_vr_sync_s": float(np.median(peak_signed_offsets)),
            "q25_signed_voltage_peak_to_vr_sync_s": float(np.quantile(peak_signed_offsets, 0.25)),
            "q75_signed_voltage_peak_to_vr_sync_s": float(np.quantile(peak_signed_offsets, 0.75)),
            "fraction_event_starts_within_2s_of_vr_sync": float(np.mean(start_distances <= 2)),
            "corridor_window_high_amplitude_fraction": float(event[corridor].mean()),
            "dark_window_high_amplitude_fraction": float(event[dark].mean()),
            "outside_behaviour_window_high_amplitude_fraction": float(event[outside].mean()),
        })

        # Six-minute illustrative window beginning at first behavioural frame.
        start = record["frame_s"].min()
        stop = start + 360
        window_s = record["window_start_sample"] / FS
        keep = (window_s >= start) & (window_s <= stop)
        rms = np.log10(np.maximum(record["window_median_channel_rms"], np.finfo(float).tiny))
        ax.scatter(window_s[keep] - start, rms[keep], s=5, color=COLOURS["ordinary"])
        highlight = keep & event
        ax.scatter(window_s[highlight] - start, rms[highlight], s=18, color=COLOURS["event"],
                   label="high-amplitude mode")
        for transition in sync_s[(sync_s >= start) & (sync_s <= stop)]:
            ax.axvline(transition - start, color="black", lw=0.45, alpha=0.35)
        ax.set_ylabel(f"{record['mouse']}\nlog10 RMS")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.99, 0.78,
                f"interval {np.median(intervals):.0f} s; peak-sync "
                f"{np.median(peak_signed_offsets):+.2f} s "
                f"[IQR {np.quantile(peak_signed_offsets, 0.25):+.2f}, "
                f"{np.quantile(peak_signed_offsets, 0.75):+.2f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                      "pad": 1.5})
    axes[0].set_title(
        "High-amplitude one-second bins (orange); black lines are VR sync transitions"
    )
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("Seconds from first behavioural frame (six-minute excerpt)")
    save_pair(fig, "sanity_audit_event_timing")

    path = config.RESULTS_DIR / "sanity_timing_summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_pair(fig, stem):
    for extension in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"{stem}.{extension}", dpi=100)
    plt.close(fig)


def main():
    records = [load_record(mouse) for mouse in config.LFP_MICE]
    build_spectral_cache(records)
    save_figure_source_data(records)
    plot_overview(records)
    plot_raw_examples(records)
    timing_summary_and_plot(records)
    print("Saved v2 overview, raw examples, event timing, and timing summary.")


if __name__ == "__main__":
    main()
