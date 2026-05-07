% File: plotTrialFactorVsBehaviorCorr.m
function plotTrialFactorVsBehaviorCorr(model, aligned_lick_errors, nFactors)
% Plots correlation between each trial factor and mean inverse lick error across all aligned trials.

    fprintf('Plotting correlation between trial factors and overall mean inverse lick error...\n');

    if isempty(model) || length(model.U) < 3 || isempty(model.U{3})
        warning('plotTrialFactorVsBehaviorCorr: Invalid model or trial factors missing.');
        return;
    end
    if isempty(aligned_lick_errors)
         warning('plotTrialFactorVsBehaviorCorr: aligned_lick_errors data missing.');
        return;
    end

    trial_loadings = model.U{3};
    n_aligned_trials = size(trial_loadings, 1);

    if size(aligned_lick_errors, 2) ~= n_aligned_trials
        warning('plotTrialFactorVsBehaviorCorr: Trial dimension mismatch between factors (%d) and lick errors (%d).', n_aligned_trials, size(aligned_lick_errors, 2));
        return;
    end

    % Calculate mean inverse lick error (higher = better performance)
    mean_inv_lick_error = -mean(aligned_lick_errors, 1, 'omitnan')'; % Result is nTrials x 1

    if all(isnan(mean_inv_lick_error))
        warning('plotTrialFactorVsBehaviorCorr: Mean inverse lick error is all NaN.');
        return;
    end

    figure;
    t = tiledlayout('flow', 'TileSpacing', 'compact');
    title(t, 'Trial Factor Loading vs. Mean Performance (Inverse Lick Error)');
    xlabel(t, 'Factor Loading');
    ylabel(t, 'Mean Inverse Lick Error');

    for r = 1:nFactors
        nexttile;
        factor_load = trial_loadings(:, r);

        % Filter NaN pairs for correlation and plotting
        valid_idx = ~isnan(factor_load) & ~isnan(mean_inv_lick_error);
        factor_load_filt = factor_load(valid_idx);
        behavior_filt = mean_inv_lick_error(valid_idx);

        if numel(factor_load_filt) < 3 % Need at least 3 points for correlation/line
            text(0.5, 0.5, 'Not enough data', 'HorizontalAlignment', 'center');
            title(sprintf('Factor %d', r));
            continue;
        end

        scatter(factor_load_filt, behavior_filt, 50, 'filled', 'MarkerFaceAlpha', 0.6);
        hold on;
        lsline; % Add linear regression line
        hold off;

        % Calculate and display correlation coefficient
        [R, P] = corr(factor_load_filt, behavior_filt, 'Type', 'Pearson'); % Or Spearman?
        title(sprintf('Factor %d (R=%.2f, p=%.3f)', r, R, P));
        box off;
        grid on;
    end
end