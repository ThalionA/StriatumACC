"""Deep sanity check: is ordinary exported voltage LF-like or broadband noise?"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config, geometry  # noqa: E402
from striatum_lfp.reader import DATASET  # noqa: E402
from striatum_lfp.sanity import (  # noqa: E402
    common_median_reference,
    median_welch_psd,
    temporal_identity_metrics,
)

FS = config.FS
BANDS = {"theta": (4, 8), "beta": (15, 30), "low_gamma_confounded": (30, 80)}
COLOURS = {"DMS": "#4C72B0", "DLS": "#55A868", "ACC": "#DD8452"}


def median_pairwise_correlation(data):
    matrix = np.corrcoef(data, rowvar=False)
    return float(np.nanmedian(matrix[np.triu_indices(matrix.shape[0], 1)]))


def main():
    rows = []
    figure_records = []
    figure_data = {
        "selected_channels_zero_based": np.arange(0, config.N_CHANNELS, 16),
        "nominal_areas": np.asarray(tuple(COLOURS)),
        "correlation_bands": np.asarray(tuple(BANDS)),
    }
    filters = {
        name: butter(4, limits, btype="bandpass", fs=FS, output="sos")
        for name, limits in BANDS.items()
    }
    for mouse in config.LFP_MICE:
        cached = np.load(config.RESULTS_DIR / f"sanity_audit_{mouse}.npz")
        ordinary = (
            (cached["world"] == 6)
            & (cached["window_exact_zero_fraction"] < 0.99)
            & np.isfinite(cached["artefact_z"])
            & (np.abs(cached["artefact_z"]) < 1)
        )
        candidates = np.flatnonzero(ordinary)
        selected = candidates[np.linspace(0, len(candidates) - 1, min(40, len(candidates))).astype(int)]
        channels = np.arange(0, config.N_CHANNELS, 16)
        windows = []
        referenced_windows = []
        with h5py.File(config.LFP_DIR / config.FILE_BY_MOUSE[mouse], "r") as handle:
            dataset = handle[DATASET]
            for second in selected:
                start = max(0, min(second * FS - 1_500, dataset.shape[0] - 4_000))
                full_window = np.asarray(
                    dataset[start : start + 4_000, :], dtype=float
                )
                windows.append(full_window[:, channels])
                referenced_windows.append(
                    common_median_reference(full_window)[:, channels]
                )
        windows = np.stack(windows)
        referenced_windows = np.stack(referenced_windows)
        frequencies, psd = median_welch_psd(windows, fs_hz=FS)
        _, referenced_psd = median_welch_psd(referenced_windows, fs_hz=FS)
        identity = temporal_identity_metrics(windows, fs_hz=FS)

        peak_band = (frequencies >= 40) & (frequencies <= 180)
        peaks, properties = find_peaks(np.log10(psd[peak_band]), prominence=0.15)
        peak_freqs = frequencies[peak_band][peaks]
        prominences = properties["prominences"]
        order = np.argsort(prominences)[::-1]
        strongest = peak_freqs[order[:2]] if order.size else np.array([])
        peak_reference_change_db = []
        for peak_hz in strongest:
            index = int(np.argmin(np.abs(frequencies - peak_hz)))
            peak_reference_change_db.append(
                float(10 * np.log10(referenced_psd[index] / psd[index]))
            )

        # 1212's voltage probe identity is unverified (it had two probes), so
        # striatal area labels would be an unsupported assertion. For the three
        # one-probe mice these remain nominal depth-band diagnostics rather than
        # physiological connectivity estimates.
        area_mapping_status = (
            "withheld_probe_identity_unverified"
            if mouse == 1212
            else "nominal_striatal_depth_bands"
        )
        if mouse == 1212:
            masks = {}
        else:
            bounds = geometry.load_area_boundaries(mouse, "striatum")
            masks = geometry.channel_area_masks(geometry.channel_depths(), bounds)
        correlations = {
            area: {"raw": []} | {band: [] for band in BANDS} for area in masks
        }
        # Correlation is evaluated on the already loaded 24 depth-spanning channels,
        # restricted to whichever of those fall in each area.
        for window in windows:
            filtered = {
                band: sosfiltfilt(sos, window, axis=0) for band, sos in filters.items()
            }
            for area, mask in masks.items():
                local = np.flatnonzero(mask[channels])
                if local.size < 2:
                    continue
                correlations[area]["raw"].append(median_pairwise_correlation(window[:, local]))
                for band in BANDS:
                    correlations[area][band].append(
                        median_pairwise_correlation(filtered[band][:, local])
                    )

        rows.append({
            "mouse_id": mouse,
            "n_corridor_windows": len(windows),
            "median_lag1_correlation": identity.median_lag1_correlation,
            "median_difference_to_signal_sd": identity.median_difference_to_signal_sd,
            "power_fraction_1_100hz": identity.power_fraction_1_100hz,
            "loglog_slope_2_40hz": identity.loglog_slope_2_40hz,
            "strongest_peak_40_180hz": strongest[0] if strongest.size else np.nan,
            "second_peak_40_180hz": strongest[1] if strongest.size > 1 else np.nan,
            "strongest_peak_common_median_change_db": (
                peak_reference_change_db[0] if peak_reference_change_db else np.nan
            ),
            "second_peak_common_median_change_db": (
                peak_reference_change_db[1]
                if len(peak_reference_change_db) > 1
                else np.nan
            ),
            "area_mapping_status": area_mapping_status,
            **{
                f"{area.lower()}_{band}_median_r": (
                    float(np.median(correlations[area][band]))
                    if area in correlations and correlations[area][band]
                    else np.nan
                )
                for area in COLOURS
                for band in ("raw", *BANDS)
            },
        })
        figure_records.append(
            (mouse, frequencies, psd, referenced_psd, identity, correlations)
        )
        figure_data[f"{mouse}_frequency_hz"] = frequencies
        figure_data[f"{mouse}_stored_psd"] = psd
        figure_data[f"{mouse}_common_median_referenced_psd"] = referenced_psd
        figure_data[f"{mouse}_identity_metrics"] = np.asarray(
            [
                identity.median_lag1_correlation,
                identity.power_fraction_1_100hz,
                identity.median_difference_to_signal_sd,
            ]
        )
        figure_data[f"{mouse}_nominal_depth_band_correlations"] = np.asarray(
            [
                [
                    np.median(correlations[area][band])
                    if area in correlations and correlations[area][band]
                    else np.nan
                    for band in BANDS
                ]
                for area in COLOURS
            ]
        )

    output = config.RESULTS_DIR / "signal_identity_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        config.RESULTS_DIR / "signal_identity_figure_data.npz", **figure_data
    )
    plot(figure_records)
    print(f"Saved {output} and signal_identity.png/.svg")


def plot(records):
    fig, axes = plt.subplots(4, 3, figsize=(15, 10), constrained_layout=True)
    for row, (
        mouse,
        frequencies,
        psd,
        referenced_psd,
        identity,
        correlations,
    ) in enumerate(records):
        ax = axes[row, 0]
        band = (frequencies >= 1) & (frequencies <= 300)
        ax.loglog(
            frequencies[band], psd[band], color="#4C72B0", lw=1,
            label="stored voltage",
        )
        ax.loglog(
            frequencies[band], referenced_psd[band], color="#DD8452", lw=0.8,
            ls="--", label="common-median referenced",
        )
        for low, high, colour in [(4, 8, "#55A868"), (15, 30, "#4C72B0"), (30, 80, "#8172B3")]:
            ax.axvspan(low, high, color=colour, alpha=0.07)
        ax.set_ylabel(f"{mouse}\nPSD\n(stored units²/Hz)")
        if row == 0:
            ax.set_title("A. Median spectrum: 40 corridor windows")
            ax.legend(fontsize=6, loc="best")

        ax = axes[row, 1]
        metrics = [identity.median_lag1_correlation,
                   identity.power_fraction_1_100hz,
                   identity.median_difference_to_signal_sd]
        labels = ["lag-1 r", "power\n≤100 Hz", "ΔSD /\nsignal SD"]
        ax.bar(labels, metrics,
               color=["#4C72B0", "#55A868", "#DD8452"])
        white_noise_reference = [0.0, 99 / 498, np.sqrt(2)]
        ax.scatter(range(3), white_noise_reference, marker="_", s=170,
                   color="black", linewidths=1.2,
                   label="white-noise expectation" if row == 0 else None)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-0.2, 1.62)
        if row == 0:
            ax.set_title("B. LF-identity diagnostics")
            ax.legend(fontsize=6, loc="upper left")

        ax = axes[row, 2]
        if not correlations:
            ax.text(0.5, 0.5, "Area labels withheld:\n1212 probe identity unverified",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            areas = list(correlations)
            x = np.arange(len(areas))
            width = 0.22
            for index, band_name in enumerate(BANDS):
                values = [np.median(correlations[area][band_name])
                          if correlations[area][band_name] else np.nan for area in areas]
                bars = ax.bar(
                    x + (index - 1) * width,
                    values,
                    width,
                    label=band_name.replace("_confounded", " (confounded)").replace("_", " "),
                )
                if band_name == "low_gamma_confounded":
                    for bar in bars:
                        bar.set_hatch("//")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(areas)
            ax.set_ylim(-0.6, 0.8)
            if row == 2:
                ax.set_ylabel("median within-depth-band channel r")
        if row == 0:
            ax.set_title("C. Nominal depth-band correlations")
        if row == 1 and correlations:
            ax.legend(fontsize=7)

        for col in range(3):
            axes[row, col].spines[["top", "right"]].set_visible(False)
            if row == 3:
                axes[row, col].set_xlabel(
                    ["Frequency (Hz)", "dimensionless metric", "Area"][col]
                )
    fig.suptitle("Signal identity: broadband voltage with weak/mixed low-frequency structure", fontsize=14)
    for extension in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"signal_identity.{extension}", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
