% File: plotAlignedBehavior.m
function plotAlignedBehavior(aligned_lick_errors, cfg)
% Plots the mean +/- SEM of the aligned lick errors across animals.

    fprintf('Plotting aligned lick errors...\n');

    if isempty(aligned_lick_errors) || all(isnan(aligned_lick_errors), 'all')
        warning('plotAlignedBehavior: No valid aligned_lick_errors data to plot.');
        return;
    end

    n_aligned_trials = size(aligned_lick_errors, 2);
    if n_aligned_trials ~= 30
        warning('plotAlignedBehavior: Expected 30 aligned trials, found %d. Plotting anyway.', n_aligned_trials);
        % Define epoch boundaries based on actual trial count if not 30?
        % For now, assume boundaries at 1/3 and 2/3 points if not 30
        epoch1_end = round(n_aligned_trials / 3);
        epoch2_end = round(2 * n_aligned_trials / 3);
    else
        epoch1_end = 10;
        epoch2_end = 20;
    end

    mean_errors = mean(aligned_lick_errors, 1, 'omitnan');
    sem_errors = std(aligned_lick_errors, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(aligned_lick_errors), 1));

    figure;
    if ~isempty(which('shadedErrorBar')) % Check if shadedErrorBar exists
        shadedErrorBar(1:n_aligned_trials, mean_errors, sem_errors, 'lineprops', {'Color', 'k', 'LineWidth', 1.5}); % Black line, grey patch
    else
        warning('shadedErrorBar function not found. Plotting with basic errorbar.');
        errorbar(1:n_aligned_trials, mean_errors, sem_errors, '-k', 'LineWidth', 1.5);
    end
    hold on;
    % Add vertical lines for epoch boundaries
    xline(1, '--k', cfg.plot.epoch_names{1}, 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
    xline(epoch1_end, '--k', cfg.plot.epoch_names{2}, 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
    xline(epoch2_end, '--k', cfg.plot.epoch_names{3}, 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
    % Add line for z-score threshold used in LP definition
    yline(cfg.task_lp_zscore_threshold, 'r--', sprintf('LP Threshold (%.1f)', cfg.task_lp_zscore_threshold));
    

    xlabel('Aligned Trial Index');
    ylabel('Z-scored Lick Error');
    title('Mean Aligned Lick Errors Across Task Animals');
    axis tight;
    xlim([0, Inf])
    box off;
    hold off;

end