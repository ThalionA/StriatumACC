"""striatum_lfp -- LFP band-power analysis (drop-in for the spike firing-rate pipeline).

Front-end for the 1 kHz, 384-channel Neuropixels LFP exports: cohort discovery,
out-of-core reading, channel -> area mapping, band-power extraction and binning
onto the spike pipeline's trial / position grid, and the analysis arms built on
top of it. Submodules are imported explicitly (``from striatum_lfp import arms``).
"""
