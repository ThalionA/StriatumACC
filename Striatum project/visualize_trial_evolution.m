function visualize_trial_evolution(decoded_positions, decoder_performance, bin_size)
    n_animals = length(decoded_positions);
    neuron_counts = decoder_performance(1).neuron_counts;
    n_neuron_counts = length(neuron_counts);

    % Choose neuron counts to visualize (you can adjust this list)
    neuron_counts_to_visualize = neuron_counts;

    for n_idx = 1:length(neuron_counts_to_visualize)
        neuron_count = neuron_counts_to_visualize(n_idx);
        icount = find(neuron_counts == neuron_count);

        fprintf('\nVisualizing trial evolution for neuron count: %d\n', neuron_count);

        % Check which animals have valid data for this neuron count
        valid_animals = [];
        for ianimal = 1:n_animals
            if all(~isnan(decoder_performance(ianimal).r2(icount, :)))
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

                    % MAE
                    trial_metrics(idx).mae(itrial, ibootstrap) = mean(abs(pred_pos - true_pos));
                end
            end
        end

        % Create figure for this neuron count
        figure('Name', sprintf('Trial Evolution (Neuron Count: %d)', neuron_count), 'Position', [100 100 1200 800]);

        % 1. R-squared evolution
        subplot(1,2,1);
        plot_metric_evolution(trial_metrics, 'r2', 'R²');

        % 2. RMSE evolution
        subplot(1,2,2);
        plot_metric_evolution(trial_metrics, 'rmse', 'RMSE (cm)');

        % Add overall title
        sgtitle(sprintf('Evolution of Decoding Performance Across Trials (Neuron Count: %d)', neuron_count), 'FontSize', 14);
    end
end

function plot_metric_evolution(trial_metrics, metric_field, metric_name)
    n_animals = length(trial_metrics);
    n_trials = size(trial_metrics(1).(metric_field), 1);

    % Initialize matrix to store mean metrics across bootstraps
    metrics_mean = zeros(n_trials, n_animals);
    metrics_sem = zeros(n_trials, n_animals);

    % Colors for plotting
    colors = lines(n_animals);

    % Plot individual animals
    for idx = 1:n_animals
        metrics = trial_metrics(idx).(metric_field);
        % Mean and SEM across bootstraps for each trial
        mean_metric = mean(metrics, 2, 'omitnan');
        sem_metric = std(metrics, [], 2, 'omitnan') / sqrt(size(metrics, 2));

        metrics_mean(:, idx) = mean_metric;
        metrics_sem(:, idx) = sem_metric;

        % Plot mean metric for each animal
        plot(mean_metric, 'Color', colors(idx, :), 'LineWidth', 1);
        hold on;
    end

    % Plot mean and SEM across animals
    mean_across_animals = mean(metrics_mean, 2, 'omitnan');
    sem_across_animals = std(metrics_mean, [], 2, 'omitnan') / sqrt(n_animals);

    % Plot mean with confidence bands
    shadedErrorBar(1:n_trials, mean_across_animals, sem_across_animals, 'lineprops', {'LineWidth', 1})

    xlabel('Trial Number');
    ylabel(metric_name);
    title(['Trial-by-Trial ' metric_name]);
    legend([arrayfun(@(x) ['Animal ' num2str(x)], 1:n_animals, 'UniformOutput', false), {'Mean'}], 'Location', 'best');
end