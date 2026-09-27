# striatum_video: top-camera (face/body) features on the neural grid

Motion energy from the top camera (`video output/Top (02G55471207)` on the
datastore), binned by raw trial × 5 cm position exactly like
`spatial_binned_data` / `spatial_binned_fr_all`.

## Verified facts (2026-09-25)

- **The camera is triggered once per VR frame.** Frame i of the concatenated
  video parts is row i of the VR log. The counts are identical in 1105, 1106,
  1201 and 1206, and the pipeline's `VR_data` in `RawData/<id>_raw.mat` is the
  VR CSV row for row, with one `VR_times_synched` entry per row. So every frame
  has a Neuropixels-clock time with no fitting.
  - The filename start times are when the camera was *armed*, often about 11 min
    early. The header's 68 fps is meaningless.
- **Lock evidence** (`check_frame_lock.py`, all four sessions pass): video
  wheel-texture displacement vs VR displacement gives r ≥ 0.93 at zero frame
  shift in all 12 blocks of every session. This uses confidently tracked frames
  only (phase-correlation response > 0.5), because the texture blurs at speed.
  - **Sub-frame slide.** On frame-to-frame changes, early blocks peak sharply
    at shift 0 (r(0) about 0.6–0.8 vs r(+1) about 0–0.3). r(+1) then climbs
    smoothly through the session, reaching r(0) ≈ r(+1) by the end (1201 ends at
    0.47 vs 0.69). It is the same direction in all four sessions. It is
    gradual, not a step, so it is timing drift of under one frame (≤ ~30 ms),
    not a lost frame. That is negligible for 5 cm bins, which each take
    hundreds of ms, but not for frame-level event timing.
  - **Duplicate frames are benign.** 21–49 frames per session are identical to
    their predecessor. All occur at VR |v| = 0 in an already static scene
    (wheel ME before them is ~0.4), which is H.264 encoding "no change", not a
    repeated trigger.
- **The binning reproduces MATLAB's.** `binning.bin_session` durations equal
  `spatial_binned_data.durations` to 2e-15 s, with an identical NaN pattern
  (1105/1106/1201/1206, 41,940 bins).
- **1212 is not locked**: 346,286 frames vs 345,684 rows across its two VR
  files. It is excluded until that is explained.
- **The mouth and whisker ROIs carry running-related motion.** Both dip where
  the mouse stops (reward zone) and rise with running speed. The frame-level
  correlation of mouth ME with licks is only 0.08–0.23. An epoch difference in
  these features is confounded with speed until checked at matched speed.

## Pipeline

1. `python scripts/run_extract.py <ids>` reads each video part once (SMB, about
   5 min per session) and writes `results/<id>_video.npz`: per-frame ME for the
   wheel, mouth and whisker ROIs (`rois.py`), plus the wheel texture shift.
   It fails unless frames = VR rows.
2. `python scripts/check_frame_lock.py <ids>` is the gate: per-block lock, which
   writes `results/frame_lock.json` and `figures/frame_lock.*`.
3. `python scripts/run_bin.py <ids>` writes `results/<id>_binned.npz`: trials × 50
   bins of ME and |VR speed|, plus usable trials, epochs, LP and DP from
   `striatum_lfp.trials` (the project's one trial layer).
4. `python scripts/plot_binned.py <ids>` writes `figures/<id>_binned.*`.

Analyses, all on `results/<id>_binned.npz`:
- `run_face_learning.py`: speed-controlled face ME vs learning epochs. Negative:
  the mouth and whisker boxes don't track licks (partial r ≤ 0.13).
- `run_lick_maps.py` and `plot_lick_maps.py`: where licking moves pixels. It
  is the spout, and the spout box fails on held-out animals.
- `run_motion_svd.py` and `plot_motion_svd.py`: face motion SVD, the
  Stringer 2019 method in numpy (`motion_svd.py`). `run_bin.py` carries its
  top 10 components as `svd_1..10`.
- `run_movement_encoding.py` and `plot_movement_encoding.py`: per-unit
  cross-validated ΔR² of movement over position + slow drift, with a
  circular-shift null (empirical false-positive rate ~7%). Movement modulation
  is widespread but tiny (median ΔR² < 0.01). Video, whether ROIs or SVD, adds
  ~nothing beyond VR speed + licks, and there is no learning change.
- `run_temporal_encoding.py`: the same at the temporal CCA arm's 20 ms bins
  (spike column 0 = 2nd corridor VR row; `temporal.py`), with covariates at
  ±200 ms lags. 52–79% of units are modulated beyond VR, against 12–52% at
  5 cm. The per-unit ΔR² is still < 0.01.
- `run_cca_movement.py` and `plot_cca_movement.py`: cross-area CCA with
  movement regressed out (`cca/pipeline.prepare_pair_confounded`, partialled
  inside the folds), both arms, against 10 shifted controls per confound.
  Movement lowers CC1 below all controls in 24–44% of animal-pair-epochs
  (chance 9%). The median effect is ≤ 0.010. The naive → expert picture is
  unchanged.

The VR logs are cached in `results/vr_cache/` (git-ignored), because the SMB
share drops. Tests: `python -m pytest tests`.

## Gotcha

OpenCV 4.10's `cv2.phaseCorrelate` multiplies its inputs by the window IN PLACE.
Pass fresh arrays (see `motion.frame_features` and its regression test).
