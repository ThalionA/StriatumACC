function visualize_trial_evolution(decoded_positions, decoder_performance, bin_size, neuron_counts_to_visualize)
    n_animals = length(decoded_positions);
    neuron_counts = decoder_performance(1).neuron_counts;

    % Ensure neuron_counts_to_visualize includes only counts that are actually available
    if nargin < 4
        neuron_counts_to_visualize = neuron_counts;
    else
        neuron_counts_to_visualize = intersect(neuron_counts_to_visualize, neuron_counts);
    end

    % Loop over each neuron count to visualize trial evolution
    for n_idx = 1:length(neuron_counts_to_visualize)
        neuron_count = neuron_counts_to_visualize(n_idx);
        icount = find(neuron_counts == neuron_count);

        fprintf('\nVisualizing trial evolution for neuron count: %d\n', neuron_count);

        % Check which animals have valid data for this neuron count
        valid_animals = [];
        for ianimal = 1:n_animals
            if is_neuron_count_available(decoded_positions, decoder_performance, ianimal, neuron_count)
                valid_animals = [valid_animals, ianimal];
            end
        end

        if isempty(valid_animals)
            fprintf('No animals have data for neuron count %d. Skipping visualization.\n', neuron_count);
            continue;
        end

        % Initialize variables to store trial metrics
        trial_metrics = struct();

        for idx = 1:length(valid_animals)
            ianimal = valid_animals(idx);
            % Collect decoded positions across bootstraps
            decoded_pos_bootstraps = decoded_positions{ianimal}(icount, :);
            n_bootstraps = length(decoded_pos_bootstraps);

            % Stack decoded positions for averaging
            n_pos_bins = size(decoded_pos_bootstraps{1}, 1);
            n_trials = size(decoded_pos_bootstraps{1}, 2);
            decoded_pos_matrix = zeros(n_pos_bins, n_trials, n_bootstraps);
            for ibootstrap = 1:n_bootstraps
                decoded_pos_matrix(:, :, ibootstrap) = decoded_pos_bootstraps{ibootstrap};
            end

            % Get true positions
            true_positions = repmat((1:n_pos_bins)' * bin_size, 1, n_trials);

            % Initialize metrics for each trial
            trial_metrics(idx).r2 = zeros(n_trials, n_bootstraps);
            trial_metrics(idx).rmse = zeros(n_trials, n_bootstraps);
            trial_metrics(idx).mae = zeros(n_trials, n_bootstraps);

            % Calculate metrics for each trial and bootstrap
            for itrial = 1:n_trials
                for ibootstrap = 1:n_bootstraps
                    pred_pos = decoded_pos_matrix(:, itrial, ibootstrap);
                    true_pos = true_positions(:, itrial);

                    % R-squared
                    ss_res = sum((true_pos - pred_pos).^2);
                    ss_tot = sum((true_pos - mean(true_pos)).^2);
                    trial_metrics(idx).r2(itrial, ibootstrap) = 1 - ss_res / ss_tot;

                    % RMSE
                    trial_metrics(idx).rmse(itrial, ibootstrap) = sqrt(mean((pred_pos - true_pos).^2));
                end
            end
        end

        % Create figure for this neuron count
        figure

        % 1. R-squared evolution
        subplot(2, 1, 1);
        plot_metric_evolution(trial_metrics, 'r2', 'R²');
        xticks([])
        xlabel('')

        % 2. RMSE evolution
        subplot(2, 1, 2);
        plot_metric_evolution(trial_metrics, 'rmse', 'RMSE (cm)');

        % Add overall title
        sgtitle(sprintf('Evolution of Decoding Performance Across Trials (Neuron Count: %d, N = %d)', neuron_count, length(valid_animals)), 'FontSize', 14);
    end
end

function plot_metric_evolution(trial_metrics, metric_field, metric_name)
    n_animals = length(trial_metrics);

    % Initialize cell array to store mean metrics for each animal
    metrics_mean_cell = cell(1, n_animals);
    max_n_trials = 0;

    for idx = 1:n_animals
        metrics = trial_metrics(idx).(metric_field);
        % Mean across bootstraps for each trial
        mean_metric = mean(metrics, 2, 'omitnan');
        metrics_mean_cell{idx} = mean_metric;
        max_n_trials = max(max_n_trials, length(mean_metric));
    end

    % Initialize arrays for mean and SEM across animals
    mean_across_animals = nan(max_n_trials, 1);
    sem_across_animals = nan(max_n_trials, 1);

    for itrial = 1:max_n_trials
        % Collect metrics from animals that have data for this trial
        metrics_at_trial = [];
        for idx = 1:n_animals
            if length(metrics_mean_cell{idx}) >= itrial
                metrics_at_trial = [metrics_at_trial; metrics_mean_cell{idx}(itrial)];
            end
        end
        if ~isempty(metrics_at_trial)
            mean_across_animals(itrial) = mean(metrics_at_trial, 'omitnan');
            sem_across_animals(itrial) = std(metrics_at_trial, [], 'omitnan') / sqrt(length(metrics_at_trial));
        end
    end

    % Plot mean with confidence bands
    shadedErrorBar(1:max_n_trials, mean_across_animals, sem_across_animals, 'lineprops', {'Color', 'k', 'LineWidth', 1})

    xlabel('Trial #');
    ylabel(metric_name);
    title(['Trial-by-trial ' metric_name]);
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