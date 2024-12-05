for ianimal = 1:n_animals
    activity_to_analyse = preprocessed_data(ianimal).spatial_binned_fr_all;
    [n_units, n_bins, n_trials] = size(activity_to_analyse);

    % Reshape data for UMAP
    activity_reshaped = activity_to_analyse(:, :)';

    % Run UMAP
    [reduced_data, umap_obj] = run_umap(activity_reshaped, ...
        'n_components', 3, ...
        'n_neighbors', 15, ...
        'min_dist', 0.1, ...
        'metric', 'euclidean');

    reshaped_reduced_data = reshape(reduced_data, [n_bins, n_trials, 3]);

    % Plotting

    % Generate bin and trial indices
    bin_indices = repmat((1:n_bins)', n_trials, 1);
    trial_indices = repelem((1:n_trials)', n_bins);

    % Prepare behavioral labels
    behaviour_labels = nan(1, n_trials);
    lick_errors = preprocessed_data(ianimal).zscored_lick_errors;
    behaviour_labels(lick_errors <= -2) = 1;    % Precise
    behaviour_labels(lick_errors > -2 & lick_errors <= -1) = 0; % Medium
    behaviour_labels(lick_errors > -1) = -1;    % Imprecise
    behaviour_labels(isnan(lick_errors)) = 0;   % Default to Medium if NaN

    % Define behavior labels in the order of progression
    behaviour_labels_list = [-1, 0, 1]; % -1: Imprecise, 0: Medium, 1: Precise

    % Corresponding behavior names
    behaviour_names = {'Imprecise', 'Medium', 'Precise'};

    % Define colors for behaviors to reflect progression
    behaviour_colors = [
        1, 0, 0;   % Red for Imprecise (-1)
        1, 1, 0;   % Yellow for Medium (0)
        0, 1, 0    % Green for Precise (1)
        ];

    % Create a mapping from behavior labels to colors
    behaviour_color_map = containers.Map(num2cell(behaviour_labels_list), num2cell(behaviour_colors, 2));

    % Plotting
    figure;
    subplot(1, 2, 1)
    hold on;

    legend_handles = [];
    legend_labels = {};

    % Loop over behaviors in the defined order
    for i = 1:length(behaviour_labels_list)
        behaviour = behaviour_labels_list(i);
        behaviour_trials = find(behaviour_labels == behaviour);
        behaviour_color = behaviour_color_map(behaviour);
        behaviour_name = behaviour_names{i};

        subplot(1, 2, 1)
        % Plot an invisible point to capture the handle for the legend
        h = scatter3(NaN, NaN, NaN, 20, behaviour_color, 'filled');
        legend_handles(end+1) = h;
        legend_labels{end+1} = behaviour_name;

        % Plot trials for the current behavior
        for itrial = behaviour_trials
            obs_indices = trial_indices == itrial;
            trial_data = reduced_data(obs_indices, :);
            trial_bins = bin_indices(obs_indices);

            % Modulate color brightness for bin progression
            trial_bins_normalized = (trial_bins - 1) / (n_bins - 1);
            colors_combined = bsxfun(@times, trial_bins_normalized, behaviour_color) + ...
                bsxfun(@times, (1 - trial_bins_normalized), [1, 1, 1]);

            % Plot data points with combined colors
            scatter3(trial_data(:, 1), trial_data(:, 2), trial_data(:, 3), 20, ...
                colors_combined, 'filled');
        end

        mean_behavioural_trajectory = squeeze(mean(reshaped_reduced_data(:, behaviour_trials, :), 2));
        subplot(1, 2, 2)
        hold on
        plot3(mean_behavioural_trajectory(:, 1), mean_behavioural_trajectory(:, 2), mean_behavioural_trajectory(:, 3), 'Color', behaviour_color,...
            'LineWidth', 3)
        hold off

    end
    linkaxes

    xlabel('UMAP 1');
    ylabel('UMAP 2');
    zlabel('UMAP 3');
    sgtitle(sprintf('UMAP - mouse %d', ianimal));

    legend(legend_handles, legend_labels);
end