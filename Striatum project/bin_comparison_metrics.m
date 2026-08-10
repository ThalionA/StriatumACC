function M = bin_comparison_metrics(fr_fine, fr_coarse, centres_fine, centres_coarse)
%BIN_COMPARISON_METRICS  Per-unit soundness metrics for two spatial binnings.
%   M = BIN_COMPARISON_METRICS(fr_fine, fr_coarse, centres_fine, centres_coarse)
%
%   fr_fine   : units x bins_fine x trials firing rates (Hz; NaN = unvisited)
%   fr_coarse : units x bins_coarse x trials, SAME units and trials
%   centres_* : 1 x bins bin-centre positions (same length unit, e.g. cm or au)
%
%   Returns per-unit column vectors in struct M:
%     r_fine, r_coarse : split-half (odd vs even trials) tuning reliability at
%                        each native resolution (Pearson r).
%     r_cross          : odd-trial FINE curve vs even-trial COARSE curve
%                        linearly interpolated onto the fine grid. The decisive
%                        metric, read ASYMMETRICALLY: r_fine > r_cross is sound
%                        evidence of real sub-coarse-scale structure; but
%                        r_cross >= r_fine only means no structure DETECTABLE
%                        at this trial count -- the interpolated coarse
%                        predictor carries less sampling noise, so the negative
%                        direction is expected even when moderate fine
%                        structure exists.
%     zero_frac_*      : fraction of visited (non-NaN) unit-bin-trial entries
%                        with zero spikes -- the sparsity cost of small bins.
%     nan_frac_*       : fraction of unvisited (NaN) entries.
%     adjcorr_*        : lag-1 autocorrelation of the all-trial mean tuning
%                        curve. NOTE: SNR-dependent, not purely a grid
%                        property -- compare within a noise regime, not across.
%
%   Split halves use disjoint trial sets (odd vs even) -- no leakage.
%
%   See also compare_bin_sizes, test_bin_comparison_metrics.

    [n_units, ~, n_trials] = size(fr_fine);
    assert(size(fr_coarse, 1) == n_units, 'unit count mismatch between binnings');
    assert(size(fr_coarse, 3) == n_trials, 'trial count mismatch between binnings');

    odd_tr  = 1:2:n_trials;
    even_tr = 2:2:n_trials;

    tune_fine_odd    = squeeze_mean(fr_fine(:, :, odd_tr));
    tune_fine_even   = squeeze_mean(fr_fine(:, :, even_tr));
    tune_coarse_odd  = squeeze_mean(fr_coarse(:, :, odd_tr));
    tune_coarse_even = squeeze_mean(fr_coarse(:, :, even_tr));
    tune_fine_all    = squeeze_mean(fr_fine);
    tune_coarse_all  = squeeze_mean(fr_coarse);

    M.r_fine   = nan(n_units, 1);
    M.r_coarse = nan(n_units, 1);
    M.r_cross  = nan(n_units, 1);
    M.zero_frac_fine   = nan(n_units, 1);
    M.zero_frac_coarse = nan(n_units, 1);
    M.nan_frac_fine    = nan(n_units, 1);
    M.nan_frac_coarse  = nan(n_units, 1);
    M.adjcorr_fine     = nan(n_units, 1);
    M.adjcorr_coarse   = nan(n_units, 1);

    for u = 1:n_units
        M.r_fine(u)   = safe_corr(tune_fine_odd(u, :)', tune_fine_even(u, :)');
        M.r_coarse(u) = safe_corr(tune_coarse_odd(u, :)', tune_coarse_even(u, :)');

        % Interpolate the held-out coarse curve onto the fine grid
        coarse_even_on_fine = interp1(centres_coarse, tune_coarse_even(u, :), ...
                                      centres_fine, 'linear', 'extrap');
        M.r_cross(u) = safe_corr(tune_fine_odd(u, :)', coarse_even_on_fine');

        % Non-finite rates (zero-duration bins upstream) are treated as
        % unvisited, not as data.
        ff = fr_fine(u, :, :);   ff(~isfinite(ff)) = NaN;
        fc = fr_coarse(u, :, :); fc(~isfinite(fc)) = NaN;
        M.zero_frac_fine(u)   = mean(ff(~isnan(ff)) == 0);
        M.zero_frac_coarse(u) = mean(fc(~isnan(fc)) == 0);
        M.nan_frac_fine(u)    = mean(isnan(ff(:)));
        M.nan_frac_coarse(u)  = mean(isnan(fc(:)));

        M.adjcorr_fine(u)   = safe_corr(tune_fine_all(u, 1:end-1)',   tune_fine_all(u, 2:end)');
        M.adjcorr_coarse(u) = safe_corr(tune_coarse_all(u, 1:end-1)', tune_coarse_all(u, 2:end)');
    end
end

function m = squeeze_mean(x)
% Mean over trials (dim 3), omitting NaNs; returns units x bins.
    m = mean(x, 3, 'omitnan');
end

function r = safe_corr(a, b)
% Pearson r over jointly non-NaN entries; NaN if degenerate.
    ok = isfinite(a) & isfinite(b);
    if nnz(ok) < 3 || std(a(ok)) == 0 || std(b(ok)) == 0
        r = NaN;
    else
        c = corrcoef(a(ok), b(ok));
        r = c(1, 2);
    end
end
