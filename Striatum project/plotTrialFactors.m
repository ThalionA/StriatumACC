% File: plotTrialFactors.m
function plotTrialFactors(model, cfg, nFactors)
% Plots the trial factors (Mode 3) with epoch boundaries.

fprintf('Plotting trial factors...\n');

if isempty(model) || length(model.U) < 3 || isempty(model.U{3})
    warning('plotTrialFactors: Invalid model or trial factors missing.');
    return;
end

trial_loadings = model.U{3};
n_aligned_trials = size(trial_loadings, 1);

% Determine epoch boundaries based on config (assuming 30 trials)
if n_aligned_trials == 30
    epoch1_end = numel(cfg.control_epoch_windows{1}); % e.g., 10
    epoch2_end = epoch1_end + abs(diff(cfg.control_epoch_windows{2})) + 1; % e.g., 10 + 10 = 20
else
    warning('plotTrialFactors: Expected 30 aligned trials, found %d. Boundaries may be inaccurate.', n_aligned_trials);
    epoch1_end = round(n_aligned_trials / 3);
    epoch2_end = round(2 * n_aligned_trials / 3);
end

figure;
t = tiledlayout('flow', 'TileSpacing', 'compact');
title(t, 'Trial Factors (Mode 3 - Aligned)');
xlabel(t, 'Aligned Trial Index');
ylabel(t, 'Loading');

for r = 1:nFactors
    nexttile;
    plot(1:n_aligned_trials, trial_loadings(:,r), 'k-', 'LineWidth', 1.5);
    hold on;
    current_ylim = ylim();
    ylim([current_ylim(1) - 0.1, current_ylim(2)])
    % Mark boundaries between segments
    xline(1, '--k', cfg.plot.epoch_names{1}, 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
    xline(epoch1_end, '--k', cfg.plot.epoch_names{2}, 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
    xline(epoch2_end, '--k', cfg.plot.epoch_names{3}, 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
    title(sprintf('Factor %d', r));
    % axis tight;
    box off;
    grid on;
    hold off;
end
linkaxes
end