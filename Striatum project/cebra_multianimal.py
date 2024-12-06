# %% [markdown]
"""
# CEBRA Multi-Session Analysis Script (Using Lick Errors Only)

This script performs multi-session analysis using the CEBRA library on your neural and behavioral data from multiple mice, using only the **lick errors** as behavioral labels.

- **Sections**:
  - Data Loading and Preparation
  - Model Training
  - Model Saving
  - Data Transformation and Embedding
  - Visualization
  - Consistency Computation
"""

# %% Import Necessary Libraries
import numpy as np
import h5py
import cebra
from cebra import CEBRA
from cebra import plot_embedding, plot_loss, plot_consistency
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

# %% Data Loading and Preparation
logger.info("Step 1: Loading and preparing your data...")

animal_names = [f"Mouse {i}" for i in range(1, 9)]
num_animals = len(animal_names)

animal_neural_data_list = []
animal_continuous_label_list = []

for ianimal in range(1, 9):
    # Load your MATLAB v7.3 file using h5py
    filename = f'cebra_mouse{ianimal}data.mat'
    mat_file = h5py.File(filename, 'r')
    
    # List variables in the file
    print(f"Variables in {filename}:", list(mat_file.keys()))
    
    # Access your data
    neural_data = np.array(mat_file['neural_data']).squeeze()
    # Swap axes to get (bins, trials, neurons)
    neural_data = np.transpose(neural_data, (2, 1, 0))  # From (trials, bins, neurons) to (bins, trials, neurons)
    lick_errors = np.array(mat_file['lick_errors']).squeeze()
    
    # Close the file
    mat_file.close()
    
    # Reshape neural data to (time_steps, features)
    neurons, spatial_bins, trials = neural_data.shape
    neural_data = np.transpose(neural_data, (1, 2, 0)).reshape(-1, neurons)
    
    # Expand lick errors per trial to align with neural data
    lick_errors_per_time = np.repeat(lick_errors, spatial_bins)
    
    # Use only lick errors as labels
    continuous_label = lick_errors_per_time.reshape(-1, 1)
    
    # Remove samples with NaNs
    combined = np.hstack((neural_data, continuous_label))
    non_nan_indices = ~np.isnan(combined).any(axis=1)
    neural_data_clean = neural_data[non_nan_indices]
    continuous_label_clean = continuous_label[non_nan_indices]
    
    print(f"Original neural_data shape for Mouse {ianimal}:", neural_data.shape)
    print(f"Cleaned neural_data shape for Mouse {ianimal}:", neural_data_clean.shape)
    print(f"Number of samples removed due to NaNs for Mouse {ianimal}:", neural_data.shape[0] - neural_data_clean.shape[0])
    
    # Standardize the data
    scaler = StandardScaler()
    neural_data_clean = scaler.fit_transform(neural_data_clean)
    
    # Append to lists
    animal_neural_data_list.append(neural_data_clean)
    animal_continuous_label_list.append(continuous_label_clean)


# %% Model Training
"""
### **Note**: Run this section to train the multi-session model using only lick errors.
"""

logger.info("Step 2: Initializing the CEBRA multi-session model...")

# Initialize the CEBRA multi-session model
multi_cebra_model = CEBRA(
    model_architecture="offset10-model",
    batch_size=256,
    output_dimension=3,
    max_iterations=1000,
    time_offsets=10,
    temperature_mode="auto",
    conditional='time_delta',
    hybrid=False,  # Set hybrid=False if preferred
    distance='cosine',
    device='cuda_if_available',
    verbose=True,
)

logger.info("Step 3: Training the multi-session model...")

# Fit the model with the list of neural data and labels (using only lick errors)
multi_cebra_model.fit(animal_neural_data_list, animal_continuous_label_list)

# %% Model Saving
logger.info("Step 4: Saving the trained multi-session model...")
tmp_dir = tempfile.gettempdir()
multi_model_path = Path(tmp_dir, 'cebra_multi_model.pt')

multi_cebra_model.save(multi_model_path)

logger.info(f"Model saved at: {multi_model_path}")

# %% Data Transformation and Embedding
logger.info("Step 5: Loading the trained multi-session model...")

# Load the model
multi_cebra_model = CEBRA.load(multi_model_path)

# Transform data into embeddings for each animal
logger.info("Step 6: Transforming data into the latent space...")

animal_embeddings = []
for i in range(num_animals):
    embedding = multi_cebra_model.transform(animal_neural_data_list[i], session_id=i)
    animal_embeddings.append(embedding)

# Verify embedding dimensions
for i in range(num_animals):
    assert animal_embeddings[i].shape[0] == animal_neural_data_list[i].shape[0]

# %% Visualization
logger.info("Step 7: Visualizing the embeddings...")

# Plot embeddings for each animal
fig = plt.figure(figsize=(6 * num_animals, 6))

for i in range(num_animals):
    ax = fig.add_subplot(1, num_animals, i + 1, projection='3d')
    cebra.plot_embedding(
        animal_embeddings[i],
        embedding_labels=animal_continuous_label_list[i][:, 0],  # Using lick errors for coloring
        markersize=5,
        title=f"{animal_names[i]} Embedding",
        ax=ax
    )
    ax.axis('off')

plt.show()

# %% Visualize Training Losses
logger.info("Step 8: Visualizing the training losses...")

plt.figure(figsize=(8, 6))

cebra.plot_loss(
    multi_cebra_model,
    color='blue',
    label='CEBRA Multi-Session'
)

plt.xlabel('Iterations')
plt.ylabel('InfoNCE Loss')
plt.legend()
plt.title('Training Loss')
plt.show()

# %% Consistency Computation
logger.info("Step 9: Computing consistency across animals...")

# Prepare labels for consistency computation (using lick errors)
labels_for_consistency = [label[:, 0] for label in animal_continuous_label_list]

# Compute consistency scores for multi-animal embeddings
multi_scores, multi_pairs, multi_subjects = cebra.sklearn.metrics.consistency_score(
    embeddings=animal_embeddings,  # embeddings from multi-session model
    labels=labels_for_consistency,
    dataset_ids=animal_names,
    between='datasets'
)

# %% Visualization
logger.info("Step 10: Visualizing the embeddings and consistency maps...")

# Plot embeddings from multi-session model
fig2 = plt.figure(figsize=(6 * num_animals, 6))
for i, name in enumerate(animal_names):
    ax = fig2.add_subplot(1, num_animals, i + 1, projection='3d')
    cebra.plot_embedding(
        animal_embeddings[i],
        embedding_labels=animal_continuous_label_list[i][:, 0],  # Using lick errors for coloring
        markersize=5,
        title=f"{name} Multi-Session Embedding",
        ax=ax
    )
    ax.axis('off')
plt.show()

# Display consistency maps
fig3 = plt.figure(figsize=(11, 4))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1 = cebra.plot_consistency(
    scores,
    pairs=pairs,
    datasets=subjects,
    ax=ax1,
    title="Single-Animal Consistency",
    colorbar_label=None
)

ax2 = cebra.plot_consistency(
    multi_scores,
    pairs=multi_pairs,
    datasets=multi_subjects,
    ax=ax2,
    title="Multi-Animal Consistency",
    colorbar_label=None
)

plt.show()

# %% Remove Saved Model
logger.info("Cleaning up temporary files...")
os.remove(multi_model_path)