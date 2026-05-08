"""
cebra_analysis.py
=================

Unified CEBRA pipeline for the StriatumACC project. Replaces the three
disjoint Legacy scripts (cebra_test.py, cebra_multianimal.py,
cebra_single_multi_comparison.py) with a single configurable workflow.

Improvements over the legacy versions
-------------------------------------
1. Multi-dimensional contrastive labels: position, lick_rate, lick_errors
   (and optionally velocity) are stacked into the CEBRA-Behavior label
   vector. The legacy script only used lick_errors.
2. Per-fold StandardScaler. Legacy code fit the scaler on full cleaned
   data before the train/test split — test data leaked into the
   normalisation.
3. Held-out linear-decoder benchmark. After training, a ridge regression
   is fit from embedding -> position on a held-out trial set and the
   R^2 is reported. This is the canonical CEBRA evaluation and lets you
   compare against the existing TCA / PCA / ridge baselines.
4. Per-area subset training. Optionally re-fits CEBRA using only DMS,
   DLS, ACC, or V1 neurons and reports cross-area decoding parity.
5. Multi-session model + Procrustes-aligned single-session models in one
   pass, with consistency-score export.
6. Reproducible: every random source is seeded from a single config seed.
7. Outputs (embeddings, decoder scores, consistency scores) are written
   back to ./cebra_results/ as both .npz (for Python) and .mat (for the
   MATLAB plotting code).

Usage
-----
1. From MATLAB, run save_for_cebra.m to populate ./cebra_data/.
2. From this directory:
       python cebra_analysis.py
   Tweak the CFG dict at the top of __main__ to change which mice/areas
   are included or which contrastive labels to use.

Data layout
-----------
Each cebra_data/cebra_mouse{N}_data.mat (v7.3) is loaded with h5py and
contains:
    neural_data   (N, B, T)       firing rates
    lick_rate     (T, B)
    lick_errors   (1, T)
    position      (1, B)          bin index
    velocity      (T, B)
    learning_point  scalar (NaN if non-learner)
    area_labels   {N} object array of byte strings
    neuron_types  (N, 1)
    mouse_id      scalar
    bin_size_cm   scalar
    group_id      scalar (1=task)

This script flattens neural_data to (B*T, N) along the first two axes
so that each timepoint = (bin, trial) is one CEBRA training sample.

Created 2026-05-07 as part of the StriatumACC audit.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

import cebra
from cebra import CEBRA, plot_embedding, plot_loss
from cebra.data.helper import OrthogonalProcrustesAlignment

# `consistency_score` lives at different paths across cebra versions.
# Try the known locations in order; fall back to None and skip the
# multi-session consistency block at runtime.
try:
    from cebra.integrations.sklearn.metrics import consistency_score as _cebra_consistency_score
except ImportError:
    try:
        from cebra.sklearn.metrics import consistency_score as _cebra_consistency_score
    except ImportError:
        try:
            from cebra.metrics import consistency_score as _cebra_consistency_score
        except ImportError:
            _cebra_consistency_score = None

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.io import savemat

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #


@dataclass
class CebraConfig:
    """Pipeline configuration. Edit at the bottom of __main__ or load from JSON."""

    data_dir: Path = Path("./cebra_data")
    out_dir: Path = Path("./cebra_results")
    n_animals: int = 16                          # number of mice to scan; missing files skipped
    seed: int = 42

    # Which behavioural labels to stack as contrastive signal.
    # Order matters only for naming.
    label_keys: Sequence[str] = field(
        default_factory=lambda: ("position", "lick_rate", "lick_errors")
    )

    # Per-area refits. Set to ("all",) to skip per-area decomposition.
    # Note: only mice with a V1 probe contribute to the V1 fit; other mice
    # are skipped via the min-units gate inside fit_single_session.
    area_subsets: Sequence[str] = field(
        default_factory=lambda: ("all", "DMS", "DLS", "ACC", "V1")
    )

    # CEBRA hyperparameters
    output_dimension: int = 3
    max_iterations_single: int = 15000
    max_iterations_multi: int = 15000
    batch_size: int = 512
    learning_rate: float = 3e-4
    temperature_mode: str = "auto"
    time_offsets: int = 10
    distance: str = "cosine"
    conditional: str = "time_delta"
    architecture: str = "offset10-model"
    hybrid: bool = False
    device: str = "cuda_if_available"
    verbose: bool = True

    # Decoder benchmark
    decode_target: str = "position"  # which label to decode from the embedding
    decoder_test_frac: float = 0.25
    ridge_alpha: float = 1.0

    def make_cebra_kwargs(self) -> dict:
        return dict(
            model_architecture=self.architecture,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            output_dimension=self.output_dimension,
            time_offsets=self.time_offsets,
            temperature_mode=self.temperature_mode,
            conditional=self.conditional,
            distance=self.distance,
            hybrid=self.hybrid,
            device=self.device,
            verbose=self.verbose,
        )


# ------------------------------------------------------------------ #
# Reproducibility
# ------------------------------------------------------------------ #


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #


def _decode_matlab_string(arr: np.ndarray) -> str:
    """Convert an int16/uint16 char-code array (MATLAB v7.3 string) to str."""
    flat = np.asarray(arr).ravel()
    return "".join(chr(int(c)) for c in flat if int(c) != 0)


def _read_mat_v73(path: Path) -> dict[str, Any]:
    """Read a MATLAB v7.3 .mat file into a flat dict of numpy arrays.

    Numeric arrays come back transposed to MATLAB-native axis order (h5py
    presents axes reversed). Cell arrays of strings (saved by MATLAB as
    HDF5 reference arrays) are resolved into a numpy object array of str.
    """
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            ds = f[key]
            if not isinstance(ds, h5py.Dataset):
                # Skip groups; everything we save is at the top level
                continue

            # Detect cell-array-of-strings (stored as references)
            if h5py.check_dtype(ref=ds.dtype) is not None:
                refs = np.array(ds).ravel()
                resolved = []
                for ref in refs:
                    try:
                        sub = f[ref]
                        resolved.append(_decode_matlab_string(np.array(sub)))
                    except Exception:
                        resolved.append("Unknown")
                out[key] = np.array(resolved, dtype=object)
                continue

            arr = np.array(ds)
            # h5py reverses axes vs MATLAB
            if arr.ndim >= 2:
                arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
            out[key] = arr
    return out


@dataclass
class MouseData:
    mouse_id: int
    neural: np.ndarray            # (T*B, N) flattened
    labels: np.ndarray            # (T*B, K)
    label_keys: tuple[str, ...]
    trial_id: np.ndarray          # (T*B,) which trial each timepoint came from
    bin_id: np.ndarray            # (T*B,) which spatial bin
    learning_point: float
    area_labels: np.ndarray       # (N,)
    n_neurons: int
    n_bins: int
    n_trials: int


def load_mouse(path: Path, label_keys: Sequence[str]) -> MouseData:
    """Load one cebra_mouse{N}_data.mat and flatten to (T*B, N) layout."""
    raw = _read_mat_v73(path)

    # _read_mat_v73 transposes axes so arrays come back in MATLAB-native
    # shape. So neural_data here is (n_neurons, n_bins, n_trials).
    nd = raw["neural_data"]
    if nd.ndim != 3:
        raise ValueError(f"{path}: neural_data must be 3D, got {nd.shape}")
    n_neurons, n_bins, n_trials = nd.shape

    # Reorder to (n_trials, n_bins, n_neurons) for time-major flattening.
    nd_tbN = np.transpose(nd, (2, 1, 0))

    # Flatten: each row = one (trial, bin) timepoint = one CEBRA training sample.
    # Final neural shape: (n_trials * n_bins, n_neurons).
    neural = nd_tbN.reshape(n_trials * n_bins, n_neurons)

    # Build labels matrix. bin_id ranges 0..n_bins-1 within each trial.
    bin_id = np.tile(np.arange(n_bins), n_trials)
    trial_id = np.repeat(np.arange(n_trials), n_bins)

    def _shape_TB(arr: np.ndarray, name: str) -> np.ndarray:
        """Coerce a 2D label array to (n_trials, n_bins)."""
        if arr.shape == (n_trials, n_bins):
            return arr
        if arr.shape == (n_bins, n_trials):
            return arr.T
        raise ValueError(
            f"{name} has shape {arr.shape}; expected ({n_trials}, {n_bins}) or transpose."
        )

    label_cols = []
    for key in label_keys:
        if key == "position":
            label_cols.append(bin_id.astype(float))
        elif key == "lick_rate":
            lr = _shape_TB(raw["lick_rate"], "lick_rate")
            label_cols.append(lr.reshape(-1))
        elif key == "lick_errors":
            le = np.asarray(raw["lick_errors"]).ravel()
            if le.size != n_trials:
                le = le[:n_trials]
            label_cols.append(np.repeat(le, n_bins))
        elif key == "velocity":
            vel = _shape_TB(raw["velocity"], "velocity")
            label_cols.append(vel.reshape(-1))
        else:
            raise KeyError(f"Unknown label key: {key}")

    labels = np.column_stack(label_cols)

    # Areas — _read_mat_v73 already resolved the cell-of-strings refs.
    if "area_labels" in raw and np.asarray(raw["area_labels"]).size == n_neurons:
        areas = np.asarray(raw["area_labels"], dtype=object).ravel()
    else:
        areas = np.array(["Unknown"] * n_neurons, dtype=object)

    lp = float(np.asarray(raw.get("learning_point", np.nan)).ravel()[0])
    mouse_id = int(np.asarray(raw.get("mouse_id", -1)).ravel()[0])

    return MouseData(
        mouse_id=mouse_id,
        neural=neural,
        labels=labels,
        label_keys=tuple(label_keys),
        trial_id=trial_id,
        bin_id=bin_id,
        learning_point=lp,
        area_labels=areas,
        n_neurons=n_neurons,
        n_bins=n_bins,
        n_trials=n_trials,
    )


def drop_nan_rows(neural: np.ndarray, labels: np.ndarray, *aux: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Drop rows where neural OR any label column has a NaN."""
    mask = ~(np.isnan(neural).any(axis=1) | np.isnan(labels).any(axis=1))
    return neural[mask], labels[mask], [a[mask] for a in aux]


# ------------------------------------------------------------------ #
# Train/test split that respects trial structure
# ------------------------------------------------------------------ #


def trialwise_split(trial_id: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split timepoints into train/test by holding out whole trials."""
    rng = np.random.default_rng(seed)
    unique_trials = np.unique(trial_id)
    n_test = max(1, int(round(test_frac * unique_trials.size)))
    test_trials = rng.choice(unique_trials, size=n_test, replace=False)
    test_mask = np.isin(trial_id, test_trials)
    return ~test_mask, test_mask


# ------------------------------------------------------------------ #
# Single-session pipeline
# ------------------------------------------------------------------ #


def fit_single_session(
    md: MouseData,
    cfg: CebraConfig,
    area_filter: str | None = None,
) -> dict[str, Any]:
    """Fit one CEBRA model on one mouse, with held-out decoder benchmark."""

    # Optional area subset (column mask on neurons)
    if area_filter is None or area_filter == "all":
        col_mask = np.ones(md.n_neurons, dtype=bool)
        area_tag = "all"
    else:
        col_mask = (md.area_labels == area_filter)
        area_tag = area_filter
        if col_mask.sum() < 5:
            return {
                "skipped": True,
                "reason": f"too few {area_filter} units ({col_mask.sum()})",
            }

    X = md.neural[:, col_mask]
    Y = md.labels
    trial_id = md.trial_id

    # Drop rows with NaNs in either X or Y (CEBRA can't handle them)
    X, Y, [trial_id] = drop_nan_rows(X, Y, trial_id)

    # Trial-wise train/test split — fit scaler on train only.
    train_mask, test_mask = trialwise_split(trial_id, cfg.decoder_test_frac, cfg.seed + md.mouse_id)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    Y_train = Y[train_mask]
    Y_test = Y[test_mask]

    # Fit CEBRA on the training half only.
    model = CEBRA(max_iterations=cfg.max_iterations_single, **cfg.make_cebra_kwargs())
    model.fit(X_train, Y_train)

    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)

    # Decode the chosen target from the held-out embedding
    if cfg.decode_target not in md.label_keys:
        raise ValueError(
            f"decode_target={cfg.decode_target!r} not in label_keys={md.label_keys}"
        )
    target_idx = md.label_keys.index(cfg.decode_target)
    y_train_target = Y_train[:, target_idx]
    y_test_target = Y_test[:, target_idx]

    decoder = Ridge(alpha=cfg.ridge_alpha)
    decoder.fit(Z_train, y_train_target)
    r2 = decoder.score(Z_test, y_test_target)
    rmse = float(np.sqrt(np.mean((decoder.predict(Z_test) - y_test_target) ** 2)))

    return {
        "skipped": False,
        "model": model,
        "embedding_train": Z_train,
        "embedding_test": Z_test,
        "labels_train": Y_train,
        "labels_test": Y_test,
        "decoder_r2": float(r2),
        "decoder_rmse": rmse,
        "area": area_tag,
        "n_train_samples": int(train_mask.sum()),
        "n_test_samples": int(test_mask.sum()),
        "n_units_used": int(col_mask.sum()),
    }


# ------------------------------------------------------------------ #
# Multi-session pipeline
# ------------------------------------------------------------------ #


def fit_multisession(
    mice: list[MouseData],
    cfg: CebraConfig,
) -> dict[str, Any]:
    """Fit one multi-session CEBRA across all mice and report consistency."""

    # Normalise per-mouse (each mouse trains on its own scaler — multi-session
    # alignment is the point of the model itself, not StandardScaler).
    Xs, Ys = [], []
    for md in mice:
        X, Y, _ = drop_nan_rows(md.neural, md.labels)
        Xs.append(StandardScaler().fit_transform(X))
        Ys.append(Y)

    model = CEBRA(max_iterations=cfg.max_iterations_multi, **cfg.make_cebra_kwargs())
    model.fit(Xs, Ys)

    embeddings = [model.transform(X, session_id=i) for i, X in enumerate(Xs)]

    # Consistency on the chosen decode target (typically position).
    target_idx = cfg.label_keys.index(cfg.decode_target) if cfg.decode_target in cfg.label_keys else 0
    cons_labels = [Y[:, target_idx] for Y in Ys]
    dataset_ids = [f"Mouse{md.mouse_id}" for md in mice]

    if _cebra_consistency_score is not None:
        try:
            scores, pairs, subjects = _cebra_consistency_score(
                embeddings=embeddings,
                labels=cons_labels,
                dataset_ids=dataset_ids,
                between="datasets",
            )
        except Exception as e:
            logging.warning("consistency_score failed: %s. Skipping.", e)
            scores, pairs, subjects = np.array([]), [], dataset_ids
    else:
        logging.warning("cebra consistency_score not importable in this CEBRA version. Skipping.")
        scores, pairs, subjects = np.array([]), [], dataset_ids

    return {
        "model": model,
        "embeddings": embeddings,
        "consistency_scores": scores,
        "consistency_pairs": pairs,
        "consistency_subjects": subjects,
        "dataset_ids": dataset_ids,
    }


# ------------------------------------------------------------------ #
# Visualisation
# ------------------------------------------------------------------ #


def plot_summary(single_results: dict, multi_results: dict, cfg: CebraConfig) -> None:
    """Top-level QC figure. Skipped silently if matplotlib is unavailable."""
    try:
        n_mice = len(multi_results["embeddings"])
        fig = plt.figure(figsize=(4 * n_mice, 4))
        for i, emb in enumerate(multi_results["embeddings"]):
            ax = fig.add_subplot(1, n_mice, i + 1, projection="3d")
            colour = single_results["mice_loaded"][i].labels[:, 0]  # position
            try:
                cebra.plot_embedding(emb, embedding_labels=colour, markersize=3,
                                     title=multi_results["dataset_ids"][i], ax=ax)
            except Exception:
                # Fallback if cebra.plot_embedding is unhappy with the axis
                ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colour, s=2)
                ax.set_title(multi_results["dataset_ids"][i])
            ax.axis("off")
        plt.tight_layout()
        out = cfg.out_dir / "cebra_multisession_embeddings.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logging.info("Wrote %s", out)
    except Exception as e:
        logging.warning("plot_summary failed: %s", e)


# ------------------------------------------------------------------ #
# Output writers
# ------------------------------------------------------------------ #


def write_results(single_results: dict, multi_results: dict, cfg: CebraConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # NPZ for Python downstream
    npz_path = cfg.out_dir / "cebra_results.npz"
    np.savez(
        npz_path,
        single_decoder_r2=single_results["decoder_r2_table"],
        single_decoder_rmse=single_results["decoder_rmse_table"],
        consistency_scores=multi_results["consistency_scores"],
        consistency_pairs=np.asarray(multi_results["consistency_pairs"]),
        dataset_ids=np.asarray(multi_results["dataset_ids"], dtype=object),
        label_keys=np.asarray(cfg.label_keys, dtype=object),
        area_subsets=np.asarray(cfg.area_subsets, dtype=object),
    )
    logging.info("Wrote %s", npz_path)

    # MAT for the MATLAB plotting code
    mat_path = cfg.out_dir / "cebra_results.mat"
    mat_payload = {
        "single_decoder_r2": single_results["decoder_r2_table"],
        "single_decoder_rmse": single_results["decoder_rmse_table"],
        "consistency_scores": multi_results["consistency_scores"],
        "dataset_ids": np.array(multi_results["dataset_ids"], dtype=object),
        "label_keys": np.array(cfg.label_keys, dtype=object),
        "area_subsets": np.array(cfg.area_subsets, dtype=object),
    }
    # Save per-mouse embeddings — each as a separate variable.
    for i, emb in enumerate(multi_results["embeddings"]):
        mat_payload[f"embedding_mouse{i+1}"] = emb
    savemat(mat_path, mat_payload)
    logging.info("Wrote %s", mat_path)

    # Save the config alongside for reproducibility
    cfg_path = cfg.out_dir / "cebra_config.json"
    cfg_dict = asdict(cfg)
    cfg_dict["data_dir"] = str(cfg_dict["data_dir"])
    cfg_dict["out_dir"] = str(cfg_dict["out_dir"])
    cfg_dict["label_keys"] = list(cfg_dict["label_keys"])
    cfg_dict["area_subsets"] = list(cfg_dict["area_subsets"])
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    logging.info("Wrote %s", cfg_path)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def run(cfg: CebraConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    seed_everything(cfg.seed)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load all mice --
    mice: list[MouseData] = []
    for ianimal in range(1, cfg.n_animals + 1):
        path = cfg.data_dir / f"cebra_mouse{ianimal}_data.mat"
        if not path.exists():
            logging.warning("Missing %s, skipping.", path)
            continue
        md = load_mouse(path, cfg.label_keys)
        logging.info("Loaded mouse %d: N=%d units, B=%d bins, T=%d trials, LP=%s",
                     md.mouse_id, md.n_neurons, md.n_bins, md.n_trials, md.learning_point)
        mice.append(md)

    if len(mice) < 2:
        raise RuntimeError(f"Only {len(mice)} mouse files loaded — need at least 2.")

    # -- Per-mouse, per-area single-session fits --
    decoder_r2 = np.full((len(mice), len(cfg.area_subsets)), np.nan)
    decoder_rmse = np.full((len(mice), len(cfg.area_subsets)), np.nan)

    for i_mouse, md in enumerate(mice):
        for j_area, area in enumerate(cfg.area_subsets):
            tag = f"mouse{md.mouse_id}_{area}"
            logging.info("Single-session fit: %s", tag)
            res = fit_single_session(md, cfg, area_filter=None if area == "all" else area)
            if res["skipped"]:
                logging.info("  skipped: %s", res["reason"])
                continue
            decoder_r2[i_mouse, j_area] = res["decoder_r2"]
            decoder_rmse[i_mouse, j_area] = res["decoder_rmse"]
            logging.info("  decode %s: R^2=%.3f, RMSE=%.3f (n_test=%d, n_units=%d)",
                         cfg.decode_target, res["decoder_r2"], res["decoder_rmse"],
                         res["n_test_samples"], res["n_units_used"])

    single_results = dict(
        decoder_r2_table=decoder_r2,
        decoder_rmse_table=decoder_rmse,
        mice_loaded=mice,
    )

    # -- Multi-session fit --
    logging.info("Multi-session fit across %d mice...", len(mice))
    multi_results = fit_multisession(mice, cfg)

    plot_summary(single_results, multi_results, cfg)
    write_results(single_results, multi_results, cfg)

    # Pretty-print a small results table
    print("\n========== CEBRA decoder R^2 by mouse x area ==========")
    print("Decode target:", cfg.decode_target, "| Labels:", cfg.label_keys)
    header = f"{'Mouse':>8} | " + " | ".join(f"{a:>6}" for a in cfg.area_subsets)
    print(header)
    print("-" * len(header))
    for i_mouse, md in enumerate(mice):
        row_vals = " | ".join(f"{decoder_r2[i_mouse, j]:>6.3f}" for j in range(len(cfg.area_subsets)))
        print(f"{md.mouse_id:>8} | {row_vals}")
    print("-" * len(header))
    mean_row = "    mean | " + " | ".join(
        f"{np.nanmean(decoder_r2[:, j]):>6.3f}" for j in range(len(cfg.area_subsets))
    )
    print(mean_row)
    print()


if __name__ == "__main__":
    cfg = CebraConfig(
        data_dir=Path("./cebra_data"),
        out_dir=Path("./cebra_results"),
        n_animals=16,                         # all task mice; missing files skipped
        seed=42,
        label_keys=("position", "lick_rate", "lick_errors"),
        area_subsets=("all", "DMS", "DLS", "ACC", "V1"),
    )
    run(cfg)
