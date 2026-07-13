"""Full-file, scale-aware LFP sanity audit for all four sessions.

Reads every voltage value once and saves one-second summaries as CSV/NPZ. No
absolute signal-present threshold is used. Plotting is deliberately separate;
run ``scripts/plot_sanity_audit.py`` after this cache has been regenerated.

Run from ``Striatum project/lfp``::

    python3 scripts/run_sanity_audit.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import audit, config  # noqa: E402
from striatum_lfp.sanity import frame_values_at_samples, robust_zscore  # noqa: E402

FS = config.FS


def behaviour_at_windows(mouse_id: int, centres: np.ndarray):
    with h5py.File(config.RAW_MAT[mouse_id], "r") as handle:
        frame_ms = np.rint(np.asarray(handle["VR_times_synched"]).ravel() * 1_000).astype(int)
        vr = np.asarray(handle["VR_data"])
    world = frame_values_at_samples(centres, frame_ms, vr[:, 4])
    position_au = frame_values_at_samples(centres, frame_ms, vr[:, 1])
    valid = (centres >= frame_ms.min()) & (centres <= frame_ms.max())
    world[~valid] = np.nan
    position_au[~valid] = np.nan
    return world, position_au


def classify_windows(result: audit.VoltageAudit):
    rms = result.window_median_channel_rms
    zero_window = result.window_exact_zero_fraction >= 0.99
    log_rms = np.full(rms.shape, np.nan)
    positive = (~zero_window) & (rms > 0)
    log_rms[positive] = np.log10(rms[positive])
    artefact_z = np.full(rms.shape, np.nan)
    artefact_z[positive] = robust_zscore(log_rms[positive])
    return log_rms, artefact_z, zero_window


def save_window_table(mouse_id, result, world, position_au, artefact_z, zero_window):
    out = config.RESULTS_DIR / f"sanity_windows_{mouse_id}.csv"
    fields = [
        "mouse_id", "start_s", "world", "position_au", "corridor",
        "exact_zero_fraction", "median_channel_rms_stored_units",
        "median_abs_stored_units", "max_abs_stored_units",
        "log_rms_robust_z", "zero_window_99pct",
    ]
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, start in enumerate(result.window_start_sample):
            writer.writerow({
                "mouse_id": mouse_id,
                "start_s": start / FS,
                "world": world[i],
                "position_au": position_au[i],
                "corridor": bool(world[i] == 6),
                "exact_zero_fraction": result.window_exact_zero_fraction[i],
                "median_channel_rms_stored_units": result.window_median_channel_rms[i],
                "median_abs_stored_units": result.window_median_channel_abs[i],
                "max_abs_stored_units": result.window_max_abs[i],
                "log_rms_robust_z": artefact_z[i],
                "zero_window_99pct": bool(zero_window[i]),
            })
    np.savez_compressed(
        config.RESULTS_DIR / f"sanity_audit_{mouse_id}.npz",
        window_start_sample=result.window_start_sample,
        window_exact_zero_fraction=result.window_exact_zero_fraction,
        window_median_channel_rms=result.window_median_channel_rms,
        window_median_channel_abs=result.window_median_channel_abs,
        window_max_abs=result.window_max_abs,
        channel_exact_zero_fraction=result.channel_exact_zero_fraction,
        world=world,
        position_au=position_au,
        artefact_z=artefact_z,
    )


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for mouse_id in config.LFP_MICE:
        path = config.LFP_DIR / config.FILE_BY_MOUSE[mouse_id]
        print(f"[audit] mouse {mouse_id}: full scan of {path.name}", flush=True)
        result = audit.audit_voltage_file(path, fs_hz=FS)
        centres = result.window_start_sample + FS // 2
        world, position_au = behaviour_at_windows(mouse_id, centres)
        corridor = world == 6
        dark = np.isfinite(world) & ~corridor
        _, artefact_z, zero_window = classify_windows(result)
        save_window_table(
            mouse_id, result, world, position_au, artefact_z, zero_window
        )

        zero_global = result.exact_zero_count / result.total_value_count
        artefact_fractions = {
            threshold: float(np.nanmean(artefact_z > threshold))
            for threshold in (3, 5, 8, 10)
        }
        summary_rows.append({
            "mouse_id": mouse_id,
            "n_samples": result.n_samples,
            "duration_min_at_1khz": result.n_samples / FS / 60,
            "exact_zero_fraction_all_values": zero_global,
            "nonfinite_fraction": 1 - result.finite_count / result.total_value_count,
            "zero_window_fraction_99pct": zero_window.mean(),
            "corridor_window_count": int(corridor.sum()),
            "dark_window_count": int(dark.sum()),
            "corridor_zero_window_fraction": float(zero_window[corridor].mean()),
            "dark_zero_window_fraction": float(zero_window[dark].mean()),
            "median_rms_stored_units": float(np.nanmedian(result.window_median_channel_rms[~zero_window])),
            "max_abs_stored_units": float(np.max(result.window_max_abs)),
            **{f"artefact_fraction_z_gt_{k}": v for k, v in artefact_fractions.items()},
        })
        print(
            f"        exact zeros={100*zero_global:.3f}% | "
            f"zero windows={100*zero_window.mean():.2f}% | "
            f"artefact z>5={100*artefact_fractions[5]:.2f}%",
            flush=True,
        )

    summary_path = config.RESULTS_DIR / "sanity_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"[done] summaries: {summary_path}; now run scripts/plot_sanity_audit.py"
    )


if __name__ == "__main__":
    main()
