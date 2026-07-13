"""striatum_lfp -- LFP band-power analysis (drop-in for the spike firing-rate pipeline).

Front-end for the new 1 kHz, 384-channel Neuropixels LFP: out-of-core reading,
channel -> area mapping, channel QC, band-power extraction, and binning onto the
spike pipeline's position / trial / epoch grid. Feeds the MATLAB basic-activity
analyses (via a drop-in .mat) and the Python ``tcca`` temporal CCA.
"""

from __future__ import annotations

from . import align, audit, config, features, geometry, qc, reader, sanity

__all__ = ["align", "audit", "config", "features", "geometry", "qc", "reader", "sanity"]
