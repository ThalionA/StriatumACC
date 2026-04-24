% File: plotTrialFactorScatter3D.m
function plotTrialFactorScatter3D(model, cfg, rank_to_plot)
% Creates a 3D scatter plot of the first 'rank_to_plot' trial factors,
% colored by learning epoch.

    fprintf('Plotting 3D scatter of first %d trial factors...\n', rank_to_plot);

    if isempty(model) || length(model.U) < 3 || isempty(model.U{3})
        warning('plotTrialFactorScatter3D: Invalid model or trial factors missing.');
        return;
    end
    if size(model.U{3}, 2) < rank_to_plot
        warning('plotTrialFactorScatter3D: Model has fewer than %d factors. Cannot create 3D plot.', rank_to_plot);
        return;
    end

    trial_loadings = model.U{3}; % Trials x nFactors
    n_aligned_trials = size(trial_loadings, 1);

     % --- Define Epochs and Colors (assuming 30 trials) ---
    if n_aligned_trials == 30
        n_early = numel(cfg.control_epoch_windows{1});
        n_pre = abs(diff(cfg.control_epoch_windows{2})) + 1;
        epoch_colors = [cfg.plot.colors.epoch_early; cfg.plot.colors.epoch_middle; cfg.plot.colors.epoch_expert];
        epoch_names = cfg.plot.epoch_names;
    else
        warning('plotTrialFactorScatter3D: Expected 30 aligned trials, found %d. Epoch coloring may be inaccurate.', n_aligned_trials);
        % Fallback colors if not 30 trials
        epoch_colors = [0 0 1; 1 0.5 0; 0 1 0];
        epoch_names = {'Early', 'Middle', 'Late'};
        n_early = round(n_aligned_trials / 3);
        n_pre = round(n_aligned_trials / 3);
    end


    figure;
    hold on;
    epoch_handles = []; % To store handles for legend

    % Plot points for each epoch with corresponding color
    idx_early = 1:n_early;
    h1 = scatter3(trial_loadings(idx_early, 1), trial_loadings(idx_early, 2), trial_loadings(idx_early, 3), ...
             75, 'filled', 'MarkerFaceColor', epoch_colors(1,:), 'MarkerEdgeColor', 'k');
     epoch_handles = [epoch_handles, h1];

    idx_middle = (n_early + 1):(n_early + n_pre);
    if ~isempty(idx_middle)
        h2 = scatter3(trial_loadings(idx_middle, 1), trial_loadings(idx_middle, 2), trial_loadings(idx_middle, 3), ...
                 75, 'filled', 'MarkerFaceColor', epoch_colors(2,:), 'MarkerEdgeColor', 'k');
        epoch_handles = [epoch_handles, h2];
    end

    idx_late = (n_early + n_pre + 1):n_aligned_trials;
     if ~isempty(idx_late)
        h3 = scatter3(trial_loadings(idx_late, 1), trial_loadings(idx_late, 2), trial_loadings(idx_late, 3), ...
                 75, 'filled', 'MarkerFaceColor', epoch_colors(3,:), 'MarkerEdgeColor', 'k');
         epoch_handles = [epoch_handles, h3];
     end


    xlabel('Factor 1 Loading');
    ylabel('Factor 2 Loading');
    zlabel('Factor 3 Loading');
    title(sprintf('Trial Factor Loadings (Factors 1-%d)', rank_to_plot));
    view(-15, 30); % Adjust view angle
    grid on;
    axis tight;

    % Add legend
    valid_epoch_names = epoch_names(~arrayfun(@isempty, {idx_early, idx_middle, idx_late})); % Only names for plotted epochs
    legend(epoch_handles, valid_epoch_names, 'Location', 'best');

    hold off;
end