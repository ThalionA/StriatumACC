function visualize_performance_correlations(preprocessed_data, decoded_positions, decoder_performance, bin_size, neuron_counts_to_plot)
n_animals = length(preprocessed_data);
all_neuron_counts = decoder_performance(1).neuron_counts;

% Ensure neuron_counts_to_plot includes all counts specified and available
neuron_counts_to_plot = intersect(neuron_counts_to_plot, all_neuron_counts);
n_neuron_counts = length(neuron_counts_to_plot);

if isempty(neuron_counts_to_plot)
    error('None of the specified neuron counts are available in decoder_performance.');
end

% Loop over each specified neuron count to create a figure with trial-by-trial correlations
for icount = 1:n_neuron_counts
    neuron_count = neuron_counts_to_plot(icount);
    fprintf('\nPlotting trial-level correlations for neuron count: %d\n', neuron_count);

    % Find index of neuron_count in the full neuron_counts array
    icount_full = find(all_neuron_counts == neuron_count);

    % Create a figure with tiled layout for each neuron count
    figure('Name', sprintf('Trial-by-Trial Correlation (Neuron Count: %d)', neuron_count), 'Position', [100 100 1200 800]);
    t = tiledlayout(ceil(n_animals / 2), 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Loop over animals
    for ianimal = 1:n_animals
        if is_neuron_count_available(decoded_positions, decoder_performance, ianimal, neuron_count)
            % Retrieve lick errors and decoding performance for best bootstrap
            lick_errors = preprocessed_data(ianimal).zscored_lick_errors;
            [~, best_bootstrap_idx] = max(decoder_performance(ianimal).r2(icount_full, :), [], 'omitnan');
            decoded_pos_trial = decoded_positions{ianimal}{icount_full, best_bootstrap_idx};
            true_positions = (1:size(decoded_pos_trial, 1))' * bin_size;

            n_trials = length(lick_errors);
            trial_r2_values = nan(1, n_trials);

            % Calculate R² for each trial in the best bootstrap
            for itrial = 1:n_trials
                pred_pos = decoded_pos_trial(:, itrial);
                trial_r2_values(itrial) = calculate_r2(pred_pos, true_positions);
            end

            % Plotting for each animal's trials as scatter points
            nexttile
            scatter(trial_r2_values, lick_errors, 'b', 'filled', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75);
            lsline % Least-squares line for trial-by-trial data
            [rho, pval] = corr(trial_r2_values', lick_errors', "Rows", "complete");
            legend(sprintf('rho = %.2f, pval = %.3f', rho, pval))
            title(sprintf('Animal %d', ianimal))
            axis tight
        end
    end

    % Shared x and y labels for the entire figure
    xlabel(t, 'Decoder Performance (R²)');
    ylabel(t, 'Behavioral Performance (Z-scored Lick Errors)');

    % Add a main title for each neuron count figure
    title(t, sprintf('Trial-by-Trial Correlation for Neuron Count: %d', neuron_count), 'FontSize', 14);
end
end

function is_available = is_neuron_count_available(decoded_positions, decoder_performance, ianimal, neuron_count)
% Retrieve the list of neuron counts that were used for decoding for this specific animal
neuron_counts = decoder_performance(ianimal).neuron_counts;

% Check if the specified neuron count is in this list
if any(neuron_counts == neuron_count)
    % Find the index of this neuron count in the neuron's list
    icount_animal = find(neuron_counts == neuron_count);

    % Check if the decoded positions contain any non-NaN data for this neuron count
    % decoded_positions{ianimal}{icount_animal, :} contains cells for each bootstrap
    decoded_pos_bootstraps = decoded_positions{ianimal}(icount_animal, :);

    % Verify if at least one bootstrap has non-NaN values in decoded positions
    is_available = any(cellfun(@(x) ~all(isnan(x(:))), decoded_pos_bootstraps));
else
    is_available = false;
end
end

function r2_value = calculate_r2(predicted, true_values)
% Calculate R² for a given set of predictions and true values
ss_res = sum((true_values - predicted).^2, 'omitnan');
ss_tot = sum((true_values - mean(true_values, 'omitnan')).^2, 'omitnan');
r2_value = 1 - ss_res / ss_tot;
end