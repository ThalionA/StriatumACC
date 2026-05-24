"""Task geometry and parameter specification for the belief-state RL model.

Geometry mirrors `project_cfg.m` / `ProcessStriatumTask.m` so that model latents
align 1:1 with the neural `spatial_binned_fr` tensor (cells x bins x trials).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Task geometry (arbitrary units, a.u., matching the VR corridor)
# --------------------------------------------------------------------------
CORRIDOR_END_AU = 200.0
BIN_SIZE_AU = 4.0
N_BINS = int(CORRIDOR_END_AU / BIN_SIZE_AU)        # 50 spatial bins
AU_TO_CM = 1.25
BIN_SIZE_CM = BIN_SIZE_AU * AU_TO_CM               # 5 cm / bin

VISUAL_LANDMARK_AU = 80.0                          # VZ
REWARD_START_AU = 100.0                            # RZ start
REWARD_END_AU = 135.0                              # RZ end

# Bin centres (a.u.); bin t (0-indexed) spans [t*4, (t+1)*4)
BIN_CENTRES_AU = (np.arange(N_BINS) + 0.5) * BIN_SIZE_AU
RZ_MASK = (BIN_CENTRES_AU >= REWARD_START_AU) & (BIN_CENTRES_AU <= REWARD_END_AU)


@dataclass(frozen=True)
class TaskConfig:
    """Immutable container for task geometry passed into the agent."""

    n_bins: int = N_BINS
    bin_size_au: float = BIN_SIZE_AU
    bin_size_cm: float = BIN_SIZE_CM
    visual_landmark_au: float = VISUAL_LANDMARK_AU
    reward_start_au: float = REWARD_START_AU
    reward_end_au: float = REWARD_END_AU
    sigma0: float = 400.0      # fixed initial perceptual variance (a.u.^2; SD 20)
    reward_magnitude: float = 1.0  # value of water collected in the RZ


# --------------------------------------------------------------------------
# Free (subjective) parameters, fit per mouse.
# Each parameter is optimised in an unconstrained space and mapped through a
# transform.  `transform`:
#   'exp'      -> positive          (constrained = exp(u))
#   'sigmoid'  -> open interval (0,1) (constrained = 1/(1+exp(-u)))
#   'identity' -> real line
# `typical` is a plausible value, used to centre synthetic ground truth and to
# initialise the optimiser.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Param:
    name: str
    transform: str
    typical: float
    note: str


PARAMS: tuple[Param, ...] = (
    # --- Perceptual process ("where am I", fast) ---
    Param("Q",             "exp",     0.6,  "path-integration noise variance per a.u. travelled"),
    Param("R0",            "exp",     2.0,  "base visual observation noise variance"),
    Param("R_slope",       "exp",     0.2,  "growth of obs noise with distance from the landmark"),
    Param("iti_inflation", "exp",     5.0,  "perceptual variance injected between trials"),
    # --- Value process ("what is best", slow) ---
    Param("eta_w",         "sigmoid", 0.09, "critic (state-value) learning rate"),
    Param("gamma",         "sigmoid", 0.80, "TD discount factor (localises value to the RZ)"),
    Param("w_init",        "exp",     0.35, "flat initial critic value carried in from the prior random-reward task"),
    # --- Lick policy (Poisson-count readout of the critic) ---
    Param("beta",          "exp",     7.0,  "lick policy gain (inverse temperature)"),
    Param("theta",         "identity",0.28, "lick value threshold"),
    Param("lambda_max",    "exp",     3.0,  "saturation lick count per bin (Poisson rate ceiling)"),
    # --- Velocity policy (continuous, log-normal readout of the critic) ---
    Param("v_base",        "identity",3.30, "baseline log-velocity (cm/s in log space)"),
    Param("v_slope",       "identity",-0.9, "value -> log-velocity slope (slow when value high)"),
    Param("log_sigma_v",   "identity",-1.4, "log of velocity emission SD (log-velocity space)"),
)

PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PARAMS)
N_PARAMS = len(PARAMS)
