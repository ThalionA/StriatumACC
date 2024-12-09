# %% [markdown]
"""
# CEBRA Multi-Session Analysis Script (Using Lick Errors Only, with Single vs Multi-Session Comparison)

This script:
- Loads and preprocesses neural and lick error data from multiple animals.
- Trains single-session CEBRA models for each animal and obtains embeddings.
- Aligns single-session embeddings to a reference animal using Orthogonal Procrustes alignment.
- Trains a multi-session CEBRA model across all animals.
- Compares single-session (aligned) and multi-session embeddings visually.
- Computes consistency maps for single vs multi-session embeddings.

This approach is inspired by the hippocampus multi-session demo.
"""

# %% Import Necessary Libraries
import numpy as np
import h5py
import cebra
from cebra import CEBRA
from cebra import plot_embedding, plot_loss, plot_consistency
from cebra.data.helper import OrthogonalProcrustesAlignment
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import torch
import random
import logging
import os

# %% Setup Logging and Reproducibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# %% Configuration
ANIMAL_COUNT = 8
animal_names = [f"Mouse {i}" for i in range(1, ANIMAL_COUNT + 1)]
DATA_DIR = Path('.')  # Update if needed

MODEL_ARCHITECTURE = "offset10-model"
MODEL_OUTPUT_DIM = 3
MODEL_ITERATIONS = 1000
MODEL_BATCH_SIZE = 256
MODEL_TIME_OFFSETS = 10
CONDITIONAL_MODE = 'time_delta'

# %% Data Loading and Preparation
logger.info("Step 1: Loading and preparing your data...")

animal_neural_data_list = []
animal_continuous_label_list = []

for ianimal in range(1, ANIMAL_COUNT + 1):
    filename = f'cebra_mouse{ianimal}data.mat'
    with h5py.File(filename, 'r') as mat_file:
        # List variables in the file
        logger.info(f"Variables in {filename}: {list(mat_file.keys())}")
        neural_data = np.array(mat_file['neural_data']).squeeze()
        lick_errors = np.array(mat_file['lick_errors']).squeeze()

    # neural_data: (trials, bins, neurons)
    # Transpose to (bins, trials, neurons)
    neural_data = np.transpose(neural_data, (1, 0, 2))
    spatial_bins, trials, neurons = neural_data.shape

    # Flatten to (time_steps, neurons)
    neural_data = neural_data.reshape(-1, neurons)
    # Repeat lick errors for each bin
    lick_errors_per_time = np.repeat(lick_errors, spatial_bins).reshape(-1, 1)

    # Remove samples with NaNs
    combined = np.hstack((neural_data, lick_errors_per_time))
    non_nan_indices = ~np.isnan(combined).any(axis=1)
    neural_data_clean = neural_data[non_nan_indices]
    continuous_label_clean = lick_errors_per_time[non_nan_indices]

    logger.info(f"Original neural_data shape for Mouse {ianimal}: (time_steps={spatial_bins*trials}, neurons={neurons})")
    logger.info(f"Cleaned neural_data shape for Mouse {ianimal}: {neural_data_clean.shape}")
    logger.info(f"Number of samples removed due to NaNs for Mouse {ianimal}: {(spatial_bins*trials) - neural_data_clean.shape[0]}")

    # Standardize the data
    scaler = StandardScaler()
    neural_data_clean = scaler.fit_transform(neural_data_clean)

    # Append to lists
    animal_neural_data_list.append(neural_data_clean)
    animal_continuous_label_list.append(continuous_label_clean)

# %% Single-Session Training
logger.info("Step 2: Training single-session models for each animal...")

max_iterations = 15000  # For better embeddings, as in the demo
single_embeddings = {}
single_models = []

for name, X, y in zip(animal_names, animal_neural_data_list, animal_continuous_label_list):
    logger.info(f"Fitting single-session CEBRA for {name}")
    cebra_model = CEBRA(
        model_architecture=MODEL_ARCHITECTURE,
        batch_size=512,
        learning_rate=3e-4,
        temperature=1,
        output_dimension=3,
        max_iterations=max_iterations,
        distance='cosine',
        conditional=CONDITIONAL_MODE,
        device='cuda_if_available',
        verbose=True,
        time_offsets=MODEL_TIME_OFFSETS,
        hybrid=True
    )

    cebra_model.fit(X, y)
    single_embeddings[name] = cebra_model.transform(X)
    single_models.append(cebra_model)

# %% Align Single-Session Embeddings
logger.info("Step 3: Aligning single-session embeddings to the first animal...")

alignment = OrthogonalProcrustesAlignment()
reference_name = animal_names[0]
reference_embedding = single_embeddings[reference_name]
reference_label = animal_continuous_label_list[0]

aligned_embeddings = {reference_name: reference_embedding}
for i, name in enumerate(animal_names[1:]):
    aligned_embeddings[name] = alignment.fit_transform(
        reference_embedding, 
        single_embeddings[name],
        reference_label, 
        animal_continuous_label_list[i+1]
    )

# %% Multi-Session Training
logger.info("Step 4: Training a multi-session model on all animals...")

multi_cebra_model = CEBRA(
    model_architecture=MODEL_ARCHITECTURE,
    batch_size=512,
    learning_rate=3e-4,
    temperature=1,
    output_dimension=3,
    max_iterations=max_iterations,
    distance='cosine',
    conditional=CONDITIONAL_MODE,
    device='cuda_if_available',
    verbose=True,
    time_offsets=MODEL_TIME_OFFSETS,
    hybrid=True
)

multi_cebra_model.fit(animal_neural_data_list, animal_continuous_label_list)

multi_embeddings = {}
for i, (name, X) in enumerate(zip(animal_names, animal_neural_data_list)):
    multi_embeddings[name] = multi_cebra_model.transform(X, session_id=i)

# %% Visualization of Single vs Multi-Session Embeddings
logger.info("Step 5: Visualizing single (aligned) vs multi-session embeddings...")

fig = plt.figure(figsize=(4 * ANIMAL_COUNT, 8))
for idx, (name, label) in enumerate(zip(animal_names, animal_continuous_label_list)):
    # Single-session aligned embeddings
    ax_single = fig.add_subplot(2, ANIMAL_COUNT, idx + 1, projection='3d')
    cebra.plot_embedding(emb_single := aligned_embeddings[name], embedding_labels=label[:, 0],
                         markersize=1, title=f"Single-{name}", ax=ax_single)
    ax_single.axis('off')
    # Retrieve the scatter object
    sc_single = ax_single.collections[-1]
    fig.colorbar(sc_single, ax=ax_single, shrink=0.5, aspect=10)

    # Multi-session embeddings
    ax_multi = fig.add_subplot(2, ANIMAL_COUNT, ANIMAL_COUNT + idx + 1, projection='3d')
    cebra.plot_embedding(emb_multi := multi_embeddings[name], embedding_labels=label[:, 0],
                         markersize=1, title=f"Multi-{name}", ax=ax_multi)
    ax_multi.axis('off')
    # Retrieve the scatter object
    sc_multi = ax_multi.collections[-1]
    fig.colorbar(sc_multi, ax=ax_multi, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

# %% Visualization of Single vs Multi-Session Embeddings with Consistent Color Code
logger.info("Step 5: Visualizing single (aligned) vs multi-session embeddings with consistent color code...")

# Gather all label values (from all animals) to determine global min and max
all_labels = np.concatenate([lbl[:, 0] for lbl in animal_continuous_label_list])
vmin, vmax = np.percentile(all_labels, 1), np.percentile(all_labels, 99)

# Define a normalization and a colormap
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis  # choose any colormap you like

fig = plt.figure(figsize=(4 * ANIMAL_COUNT, 8))
scatters = []  # to store scatter plots for colorbar reference

for idx, (name, label) in enumerate(zip(animal_names, animal_continuous_label_list)):
    emb_single = aligned_embeddings[name]
    emb_multi = multi_embeddings[name]

    # Single-session aligned embeddings
    ax_single = fig.add_subplot(2, ANIMAL_COUNT, idx + 1, projection='3d')
    sc_single = ax_single.scatter(
        emb_single[:, 0], emb_single[:, 1], emb_single[:, 2],
        c=label[:, 0], cmap=cmap, norm=norm, s=1
    )
    ax_single.set_title(f"Single-{name}")
    ax_single.axis('off')
    scatters.append(sc_single)
    ax_single.view_init(elev=10, azim=-15)


    # Multi-session embeddings
    ax_multi = fig.add_subplot(2, ANIMAL_COUNT, ANIMAL_COUNT + idx + 1, projection='3d')
    sc_multi = ax_multi.scatter(
        emb_multi[:, 0], emb_multi[:, 1], emb_multi[:, 2],
        c=label[:, 0], cmap=cmap, norm=norm, s=1
    )
    ax_multi.set_title(f"Multi-{name}")
    ax_multi.axis('off')
    scatters.append(sc_multi)
    ax_multi.view_init(elev=10, azim=-15)

plt.tight_layout()

# Add one colorbar that applies to all plots
# We can pass a list of axes or just use the last scatter since they share the same colormap/norm
fig.colorbar(scatters[-1], ax=fig.get_axes(), shrink=0.5, aspect=10, label="Lick Error Value")

plt.show()

# %% Training Loss Visualization (Optional)
logger.info("Step 6: (Optional) Visualizing training loss of multi-session model...")

plt.figure(figsize=(8, 6))
cebra.plot_loss(multi_cebra_model, color='blue', label='CEBRA Multi-Session')
plt.xlabel('Iterations')
plt.ylabel('InfoNCE Loss')
plt.legend()
plt.title('Multi-Session Training Loss')
plt.show()

# %% Consistency Computation
logger.info("Step 7: Computing consistency across animals...")

# Prepare labels for consistency computation
labels_for_consistency = [lbl[:, 0] for lbl in animal_continuous_label_list]

# Single-session consistency
scores, pairs, subjects = cebra.sklearn.metrics.consistency_score(
    embeddings=list(aligned_embeddings.values()),
    labels=labels_for_consistency,
    dataset_ids=animal_names,
    between='datasets'
)

# Multi-session consistency
multi_scores, multi_pairs, multi_subjects = cebra.sklearn.metrics.consistency_score(
    embeddings=list(multi_embeddings.values()),
    labels=labels_for_consistency,
    dataset_ids=animal_names,
    between='datasets'
)

# %% Consistency Maps Visualization
logger.info("Step 8: Visualizing consistency maps...")

fig2 = plt.figure(figsize=(11, 4))

ax1 = fig2.add_subplot(121)
cebra.plot_consistency(scores, pairs=pairs, datasets=subjects,
                       ax=ax1, title="Single-Session Aligned", colorbar_label=None)

ax2 = fig2.add_subplot(122)
cebra.plot_consistency(multi_scores, pairs=multi_pairs, datasets=multi_subjects,
                       ax=ax2, title="Multi-Session", colorbar_label=None)

plt.tight_layout()
plt.show()

# %% Cleanup
logger.info("Cleaning up temporary files...")
# If models were saved, remove them here if desired.