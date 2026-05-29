#!/usr/bin/env python3
"""Generate the simulated population datasets for every coupling scenario.

For each scenario this writes, under ``data/generated/<scenario>/``:

- ``neural_<AREA>.npy``      -- (n_timesteps, n_neurons) observed activity.
- ``latents_<AREA>.npy``     -- (n_timesteps, n_latents) ground-truth latents.
- ``loadings_<AREA>.npy``    -- (n_neurons, n_latents) projection matrices.
- ``metadata.json``          -- config + ground-truth coupling description.

These are *generated* (synthetic) data and therefore live under
``data/generated/`` per the repository conventions, never confused with real
recordings. The arrays are stored as ``(n_timesteps, n_neurons)`` which maps
directly onto ``striatum_cca.AreaActivity`` for downstream CCA analysis.

Run from the ``popsim`` subproject directory::

    python scripts/generate_datasets.py
    python scripts/generate_datasets.py --observation poisson --dynamics lds
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from popsim import scenarios, simulate  # noqa: E402

HERE = os.path.dirname(__file__)
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "data", "generated"))


def write_scenario(name: str, out_root: str, n_timesteps: int, **area_kw) -> str:
    builder = scenarios.SCENARIOS[name]
    cfg = builder(n_timesteps=n_timesteps, **area_kw)
    result = simulate(cfg)

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    for area in result.area_names:
        np.save(os.path.join(out_dir, f"neural_{area}.npy"), result.neural[area])
        np.save(os.path.join(out_dir, f"latents_{area}.npy"), result.latents[area])
        np.save(os.path.join(out_dir, f"loadings_{area}.npy"), result.loadings[area])
    with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
        json.dump(result.metadata(), fh, indent=2)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help="output root directory")
    parser.add_argument("--n-timesteps", type=int, default=3000)
    parser.add_argument(
        "--dynamics", default="ar1", choices=["ar1", "lds", "oscillatory"]
    )
    parser.add_argument(
        "--observation", default="gaussian", choices=["gaussian", "poisson"]
    )
    parser.add_argument(
        "--scenarios", nargs="*", default=list(scenarios.SCENARIOS),
        help="subset of scenarios to generate",
    )
    args = parser.parse_args()

    area_kw = {"dynamics": args.dynamics, "observation": args.observation}
    for name in args.scenarios:
        out_dir = write_scenario(name, args.out, args.n_timesteps, **area_kw)
        print(f"wrote {name} -> {out_dir}")


if __name__ == "__main__":
    main()
