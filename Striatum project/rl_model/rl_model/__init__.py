"""Belief-state RL model of VR-corridor behaviour (StriatumACC project).

Workhorse: a two-process (perception + value) actor-critic, fit per mouse by
maximum likelihood, exporting per-(trial x bin) latents as neural regressors.
See UNDERSTANDING.md for the design rationale.
"""
# Double precision: the per-session log-likelihood sums thousands of terms and
# is fed to a gradient-based optimiser — float32 is not enough.
import os as _os
import jax as _jax
_jax.config.update("jax_enable_x64", True)
# Persistent XLA compilation cache: incremental fitting scripts are invoked many
# times; this lets each invocation reuse compiled code instead of recompiling.
_jax.config.update("jax_compilation_cache_dir",
                   _os.environ.get("RLMODEL_JAXCACHE", "/tmp/rlmodel_jaxcache"))
_jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
_jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from .config import PARAM_NAMES, PARAMS, TaskConfig
from .agent import (
    default_unconstrained,
    session_latents,
    session_loglik,
    simulate_session,
    to_constrained,
    to_unconstrained,
)
from .synthetic import make_cohort, sample_params
from .fitting import fit_mouse

__all__ = [
    "PARAM_NAMES", "PARAMS", "TaskConfig",
    "default_unconstrained", "session_latents", "session_loglik",
    "simulate_session", "to_constrained", "to_unconstrained",
    "make_cohort", "sample_params", "fit_mouse",
]
