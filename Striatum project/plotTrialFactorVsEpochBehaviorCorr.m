% File: plotTrialFactorVsEpochBehaviorCorr.m
function plotTrialFactorVsEpochBehaviorCorr(model, aligned_lick_errors, cfg, nFactors)
% Calculates and plots heatmap of correlations between trial factors and
% mean inverse lick error *within* each learning epoch.

    fprintf('Plotting correlation heatmap: Trial factors vs within-epoch mean inverse lick error...\n');

    if isempty(model) || length(model.U) < 3 || isempty(model.U{3})
        warning('plotTrialFactorVsEpochBehaviorCorr: Invalid model or trial factors missing.');
        return;
    end
     if isempty(aligned_lick_errors)
         warning('plotTrialFactorVsEpochBehaviorCorr: aligned_lick_errors data missing.');
        return;
    end

    trial_loadings = model.U{3};
    n_aligned_trials = size(trial_loadings, 1);

     if size(aligned_lick_errors, 2) ~= n_aligned_trials
        warning('plotTrialFactorVsEpochBehaviorCorr: Trial dimension mismatch between factors (%d) and lick errors (%d).', n_aligned_trials, size(aligned_lick_errors, 2));
        return;
    end

    % Calculate mean inverse lick error
    mean_inv_lick_error = -mean(aligned_lick_errors, 1, 'omitnan')'; % nTrials x 1

    if all(isnan(mean_inv_lick_error))
        warning('plotTrialFactorVsEpochBehaviorCorr: Mean inverse lick error is all NaN.');
        return;
    end

    % --- Define Epochs (assuming 30 trials) ---
    if n_aligned_trials == 30
        n_early = numel(cfg.control_epoch_windows{1});
        n_pre = abs(diff(cfg.control_epoch_windows{2})) + 1;
        n_post = abs(diff(cfg.control_epoch_windows{3}));
        epoch_indices = {1:n_early, (n_early+1):(n_early+n_pre), (n_early+n_pre+1):(n_early+n_pre+n_post)};
        epoch_names = cfg.plot.epoch_names;
        nEpochs = numel(epoch_names);
    else
        warning('plotTrialFactorVsEpochBehaviorCorr: Expected 30 aligned trials, found %d. Cannot define epochs reliably.', n_aligned_trials);
        return;
    end

    % --- Calculate Correlations ---
    correlation_coeffs = nan(nFactors, nEpochs);
    p_values = nan(nFactors, nEpochs);

    for r = 1:nFactors % Loop through each factor
        for e = 1:nEpochs % Loop through each epoch
            current_trials = epoch_indices{e};
            factor_data_epoch = trial_loadings(current_trials, r);
            behavior_data_epoch = mean_inv_lick_error(current_trials);

            valid_pair_idx = ~isnan(factor_data_epoch) & ~isnan(behavior_data_epoch);
            factor_data_filt = factor_data_epoch(valid_pair_idx);
            behavior_data_filt = behavior_data_epoch(valid_pair_idx);

            if sum(valid_pair_idx) > 2 % Need > 2 points for correlation
                [R, P] = corr(factor_data_filt, behavior_data_filt, 'Type', 'Pearson'); % Or Spearman
                correlation_coeffs(r, e) = R;
                p_values(r, e) = P;
            end
        end
    end

    % --- Visualize Correlations as Heatmap ---
    figure;
    imagesc(correlation_coeffs);
    set(gca, 'YDir', 'normal'); % Factor 1 at bottom

    % Set colormap (use redblue from File Exchange if available, otherwise another diverging map)
    if ~isempty(which('redblue'))
        colormap(redblue);
    else
        warning('redblue colormap not found. Using coolwarm.');
        colormap(coolwarm); % Alternative diverging map
    end
    cb = colorbar;
    ylabel(cb, 'Pearson Correlation (R)');
    clim([-1 1]); % Center colorbar at 0

    % Add text labels (correlation values) and significance stars
    hold on;
    for r = 1:nFactors
        for e = 1:nEpochs
            corr_val = correlation_coeffs(r, e);
            p_val = p_values(r, e);

            if ~isnan(corr_val)
                 % Determine text color based on background
                text_color = 'k'; % Default black
                if abs(corr_val) > 0.6 % Heuristic for strong background color
                     text_color = 'w';
                end

                 % Add correlation value text
                 text_str = sprintf('%.2f', corr_val);
                 text(e, r, text_str, 'HorizontalAlignment', 'center', ...
                      'VerticalAlignment', 'bottom', 'FontSize', 9, 'Color', text_color, 'FontWeight','normal');

                 % Add significance stars below the text
                 star_str = '';
                 if ~isnan(p_val)
                    if p_val < 0.001
                        star_str = '***';
                    elseif p_val < 0.01
                        star_str = '**';
                    elseif p_val < 0.05
                        star_str = '*';
                    end
                 end
                 if ~isempty(star_str)
                      text(e, r, star_str, 'HorizontalAlignment', 'center', ...
                          'VerticalAlignment', 'top', 'FontSize', 12, 'Color', text_color, 'FontWeight', 'bold');
                 end
            end
        end
    end
    hold off;

    % Add labels and title
    xticks(1:nEpochs);
    xticklabels(epoch_names);
    xtickangle(30);
    yticks(1:nFactors);
    yticklabels(arrayfun(@(x) sprintf('F%d', x), 1:nFactors, 'UniformOutput', false)); % Label factors as F1, F2...
    xlabel('Learning Epoch');
    ylabel('Factor Index');
    title({'Correlation: Trial Factor vs. Mean Performance', '(Within Each Epoch)'});
end