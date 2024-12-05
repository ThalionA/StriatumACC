# %% [markdown]
"""
# CEBRA Analysis Script

This script performs analysis using the CEBRA library on your neural and behavioral data.

- **Sections**:
  - Data Loading and Preparation
  - Model Training
  - Model Saving
  - Data Transformation and Embedding
  - Visualization
"""

# %% Import Necessary Libraries
import numpy as np
import h5py
import cebra
from cebra import CEBRA
from cebra import plot_embedding, plot_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import torch
import random
import logging
import os

# %% Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# %% Data Loading and Preparation
logger.info("Step 1: Loading and preparing your data...")

# Load your MATLAB v7.3 file using h5py
mat_file = h5py.File('cebra_mouse3data.mat', 'r')

# List variables in the file
print("Variables in the MATLAB file:", list(mat_file.keys()))

# Access your data
neural_data = np.array(mat_file['neural_data']).squeeze()
# Swap axes to get (bins, trials, neurons)
neural_data = np.transpose(neural_data, (2, 1, 0))  # From (trials, bins, neurons) to (bins, trials, neurons)
lick_data = np.array(mat_file['lick_data']).squeeze()
lick_data = np.transpose(lick_data, (1, 0))
lick_errors = np.array(mat_file['lick_errors']).squeeze()

# Close the file
mat_file.close()

# Reshape neural data to (time_steps, features)
neurons, spatial_bins, trials = neural_data.shape
neural_data = np.transpose(neural_data, (1, 2, 0)).reshape(-1, neurons)

# Reshape lick data to align with neural data
lick_data = lick_data.flatten()

# Expand lick errors per trial to align with neural data
lick_errors_per_time = np.repeat(lick_errors, spatial_bins)

# Combine labels if needed
continuous_label = np.column_stack((lick_data, lick_errors_per_time))

# Remove samples with NaNs
combined = np.hstack((neural_data, continuous_label))
non_nan_indices = ~np.isnan(combined).any(axis=1)
neural_data_clean = neural_data[non_nan_indices]
continuous_label_clean = continuous_label[non_nan_indices]

print("Original neural_data shape:", neural_data.shape)
print("Cleaned neural_data shape:", neural_data_clean.shape)
print("Number of samples removed due to NaNs:", neural_data.shape[0] - neural_data_clean.shape[0])

# %% Split Data into Training and Validation Sets
train_data, valid_data, train_continuous_label, valid_continuous_label = train_test_split(
    neural_data_clean,
    continuous_label_clean,
    test_size=0.3,
    random_state=42
)

# Standardize the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valid_data = scaler.transform(valid_data)

# %%
# Step 3: Initialize Multiple CEBRA Models

logger.info("Step 3: Initializing the CEBRA models...")

# CEBRA-Time Model
cebra_time_model = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    output_dimension=3,
    max_iterations=10000,
    time_offsets=10,
    temperature_mode="auto",
    conditional='time',
    hybrid=False,
    distance='cosine',
    device='cuda_if_available',
    verbose=True,
)

# CEBRA-Behavior Model
cebra_behavior_model = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    output_dimension=3,
    max_iterations=10000,
    time_offsets=10,
    temperature_mode="auto",
    conditional='time_delta',
    hybrid=False,
    distance='cosine',
    device='cuda_if_available',
    verbose=True,
)

# CEBRA-Hybrid Model
cebra_hybrid_model = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    output_dimension=3,
    max_iterations=10000,
    time_offsets=10,
    temperature_mode="auto",
    conditional='time_delta',
    hybrid=True,
    distance='cosine',
    device='cuda_if_available',
    verbose=True,
)

# %% Model Training
logger.info("Step 4: Training the CEBRA-Time model...")
cebra_time_model.fit(train_data)

logger.info("Step 4: Training the CEBRA-Behavior model...")
cebra_behavior_model.fit(train_data, train_continuous_label[:, 1])

logger.info("Step 4: Training the CEBRA-Hybrid model...")
cebra_hybrid_model.fit(train_data, train_continuous_label[:, 1])

# %% Save the Trained Models
logger.info("Step 5: Saving the trained models...")
tmp_dir = tempfile.gettempdir()
model_time_path = Path(tmp_dir, 'cebra_time_model.pt')
model_behavior_path = Path(tmp_dir, 'cebra_behavior_model.pt')
model_hybrid_path = Path(tmp_dir, 'cebra_hybrid_model.pt')

cebra_time_model.save(model_time_path)
cebra_behavior_model.save(model_behavior_path)
cebra_hybrid_model.save(model_hybrid_path)

logger.info(f"Models saved at: {tmp_dir}")

# %% Data Transformation and Embedding
logger.info("Step 6: Loading the trained models...")

# Load the models
cebra_time_model = CEBRA.load(model_time_path)
cebra_behavior_model = CEBRA.load(model_behavior_path)
cebra_hybrid_model = CEBRA.load(model_hybrid_path)

# Transform data into embeddings
logger.info("Step 7: Transforming data into the latent space...")

train_embedding_time = cebra_time_model.transform(train_data)
train_embedding_behavior = cebra_behavior_model.transform(train_data)
train_embedding_hybrid = cebra_hybrid_model.transform(train_data)

# Verify embedding dimensions
assert train_embedding_time.shape[0] == train_data.shape[0]
assert train_embedding_behavior.shape[0] == train_data.shape[0]
assert train_embedding_hybrid.shape[0] == train_data.shape[0]

# %% Visualization
logger.info("Step 8: Visualizing the embeddings...")

# Plot embeddings from different models
fig = plt.figure(figsize=(18, 6))

# CEBRA-Time embedding
ax1 = fig.add_subplot(131, projection='3d')
cebra.plot_embedding(
    train_embedding_time,
    embedding_labels=train_continuous_label[:, 0],
    markersize=5,
    title="CEBRA-Time Embedding with Lick Data",
    ax=ax1
)

# CEBRA-Behavior embedding
ax2 = fig.add_subplot(132, projection='3d')
cebra.plot_embedding(
    train_embedding_behavior,
    embedding_labels=train_continuous_label[:, 0],
    markersize=5,
    title="CEBRA-Behavior Embedding with Lick Data",
    ax=ax2
)

# CEBRA-Hybrid embedding
ax3 = fig.add_subplot(133, projection='3d')
cebra.plot_embedding(
    train_embedding_hybrid,
    embedding_labels=train_continuous_label[:, 0],
    markersize=5,
    title="CEBRA-Hybrid Embedding with Lick Data",
    ax=ax3
)

plt.show()


# Plot embeddings from different models
fig = plt.figure(figsize=(18, 6))

# CEBRA-Time embedding
ax1 = fig.add_subplot(131, projection='3d')
cebra.plot_embedding(
    train_embedding_time,
    embedding_labels=train_continuous_label[:, 1],
    markersize=5,
    title="CEBRA-Time Embedding with Lick Errors",
    ax=ax1
)

# CEBRA-Behavior embedding
ax2 = fig.add_subplot(132, projection='3d')
cebra.plot_embedding(
    train_embedding_behavior,
    embedding_labels=train_continuous_label[:, 1],
    markersize=5,
    title="CEBRA-Behavior Embedding with Lick Errors",
    ax=ax2
)

# CEBRA-Hybrid embedding
ax3 = fig.add_subplot(133, projection='3d')
cebra.plot_embedding(
    train_embedding_hybrid,
    embedding_labels=train_continuous_label[:, 1],
    markersize=5,
    title="CEBRA-Hybrid Embedding with Lick Errors",
    ax=ax3
)

plt.show()

# %% Visualize Training Losses
logger.info("Step 9: Visualizing the training losses...")

plt.figure(figsize=(8, 6))

# CEBRA-Time loss
cebra.plot_loss(
    cebra_time_model,
    color='blue',
    label='CEBRA-Time'
)

# CEBRA-Behavior loss
cebra.plot_loss(
    cebra_behavior_model,
    color='green',
    label='CEBRA-Behavior'
)

# CEBRA-Hybrid loss
cebra.plot_loss(
    cebra_hybrid_model,
    color='red',
    label='CEBRA-Hybrid'
)

plt.xlabel('Iterations')
plt.ylabel('InfoNCE Loss')
plt.legend()
plt.title('Training Loss Comparison')
plt.show()

# %% Remove saved models
logger.info("Cleaning up temporary files...")
os.remove(model_time_path)
os.remove(model_behavior_path)
os.remove(model_hybrid_path)