function test_bin_comparison_metrics()
%TEST_BIN_COMPARISON_METRICS  Synthetic ground-truth checks for the
%   fine-vs-coarse spatial-binning comparison kernel.
%
%   Generative model: units with known spatial tuning, Poisson spike counts
%   on the FINE grid; the coarse data are the SUM of adjacent fine-bin counts
%   (exactly what real re-binning of the same spikes does), with occupancy
%   125 ms per 2.5 cm bin (20 cm/s). Two scenarios:
%     A) SMOOTH tuning (field sigma 25 cm  >>  5 cm bins). Ground truth:
%        coarse bins are more reliable, and the fine grid carries no
%        DETECTABLE structure beyond an interpolated coarse curve.
%     B) NARROW tuning (field sigma 2 cm  <  5 cm bins, strong fields).
%        Ground truth: the fine grid resolves real structure the coarse grid
%        smears away (higher r_fine than r_cross for most units, and much
%        higher adjacent-bin autocorrelation of the mean curve).
%
%   (First version's adjcorr-in-smooth assertion was refuted by direct MATLAB
%   measurement -- adjcorr is SNR-dependent, so it is asserted only in the
%   narrow scenario where the grid effect dominates. 2026-08-10.)

    rng(1, 'twister');

    corridor_cm  = 250;
    centres_fine = 1.25:2.5:corridor_cm;                    % 100 bins of 2.5 cm
    centres_coarse = mean(reshape(centres_fine, 2, []), 1); % 50 bins of 5 cm
    n_units  = 80;
    n_trials = 200;
    dt_fine   = 0.125;   % s per 2.5 cm bin at 20 cm/s
    dt_coarse = 0.250;

    med = @(x) median(x, 'omitnan');

    % --- Scenario A: smooth fields (baseline 0.5 Hz + 6 Hz field) ---
    M_smooth = run_scenario(25, 6);

    % 1. Coarse bins more reliable than fine (split-half r)
    assert(med(M_smooth.r_coarse) > med(M_smooth.r_fine), ...
        'smooth: coarse split-half reliability should beat fine');

    % 2. No detectable fine structure: interpolated coarse curve predicts the
    %    held-out fine data at least as well as the native fine curve does.
    assert(med(M_smooth.r_cross) > med(M_smooth.r_fine) - 0.02, ...
        'smooth: interp-coarse should predict held-out fine data as well as native fine');

    % 3. Fine bins sparser (higher zero fraction)
    assert(med(M_smooth.zero_frac_fine) > med(M_smooth.zero_frac_coarse), ...
        'fine bins must have a higher zero-spike fraction');

    % --- Scenario B: narrow, strong fields (real sub-5cm structure) ---
    M_narrow = run_scenario(2, 10);

    % 4. The fine grid resolves structure the coarse grid smears: the mean
    %    tuning curve is far more autocorrelated bin-to-bin on the fine grid.
    assert(med(M_narrow.adjcorr_fine) > med(M_narrow.adjcorr_coarse) + 0.1, ...
        'narrow: fine mean curve should be much smoother than the smeared coarse one');

    % 5. Structure detection: most units show r_fine > r_cross, and the
    %    median margin is positive.
    assert(mean(M_narrow.r_fine > M_narrow.r_cross, 'omitnan') > 0.6, ...
        'narrow: majority of units should show native-fine beating interp-coarse');
    assert(med(M_narrow.r_fine) - med(M_narrow.r_cross) > 0, ...
        'narrow: median r_fine should exceed median r_cross');

    fprintf('test_bin_comparison_metrics: ALL PASSED\n');

    % ---------------------------------------------------------------
    function M = run_scenario(sigma_cm, amp_hz)
        % Known tuning: baseline 0.5 Hz + field of amp_hz at a random centre
        field_centres = corridor_cm * rand(n_units, 1);
        rate_fine = 0.5 + amp_hz * exp(-(centres_fine - field_centres).^2 / (2*sigma_cm^2)); % U x 100

        counts_fine = poissrnd(repmat(rate_fine * dt_fine, 1, 1, n_trials)); % U x 100 x T
        fr_fine = counts_fine / dt_fine;

        % Coarse = re-binning of the SAME spikes: sum adjacent fine-bin pairs
        counts_coarse = counts_fine(:, 1:2:end, :) + counts_fine(:, 2:2:end, :);
        fr_coarse = counts_coarse / dt_coarse;

        M = bin_comparison_metrics(fr_fine, fr_coarse, centres_fine, centres_coarse);
    end
end
