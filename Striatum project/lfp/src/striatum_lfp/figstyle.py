"""Shared figure conventions for the LFP figures.

Hoisted out of the two plotting drivers, which had grown their own copies of the
save helper, the area palette and the PNG size cap. One definition means a
recoloured area or a changed size cap cannot apply to half the figure set.

Colours match ``project_cfg.m`` ``cfg.area_colors`` so an LFP panel can sit
beside a MATLAB unit panel without the areas changing colour.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from . import config

# Longest side of a saved PNG, in pixels (global convention).
MAX_PNG_PX = 1600

AREA_ORDER: tuple[str, ...] = config.AREAS
AREA_COLOUR: dict[str, str] = {
    "DMS": "#0072b2",     # project_cfg cfg.area_colors row 1
    "DLS": "#77ac30",
    "ACC": "#d95319",
    "V1": "#7e2f8e",
    "CA1": "#cc1a33",
    "DG": "#33b3b3",
}

BAND_LABEL: dict[str, str] = {
    "theta": "theta 4–8 Hz",
    "beta": "beta 15–30 Hz",
    "low_gamma": "low gamma 30–80 Hz",
    "high_gamma": "high gamma 80–150 Hz",
    "total": "total 1–150 Hz",
}
# Bands that get their own panel. `total` is the 1/f denominator, not a result.
PLOT_BANDS: tuple[str, ...] = ("theta", "beta", "low_gamma", "high_gamma")


def save_pair(fig, stem: str) -> None:
    """Save ``stem.svg`` + ``stem.png`` into ``figures/``, PNG capped on its long side."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{stem}.svg")
    fig.savefig(config.FIGURES_DIR / f"{stem}.png",
                dpi=min(150, MAX_PNG_PX / max(fig.get_size_inches())))
    plt.close(fig)
    print(f"[plot] {stem}.svg + .png", flush=True)
