function visualize_trial_evolution(decoded_positions, bin_size)
    n_animals = length(decoded_positions);
    n_trials = size(decoded_positions{1}, 2);
    
    % Calculate trial-by-trial performance metrics
    trial_metrics = struct();
    for ianimal = 1:n_animals
        true_positions = repmat((1:size(decoded_positions{1}, 1))' * bin_size, 1, n_trials);
        
        % Calculate metrics for each trial
        trial_metrics(ianimal).r2 = zeros(n_trials, 1);
        trial_metrics(ianimal).rmse = zeros(n_trials, 1);
        trial_metrics(ianimal).mae = zeros(n_trials, 1);
        
        for itrial = 1:n_trials
            pred_pos = decoded_positions{ianimal}(:, itrial);
            true_pos = true_positions(:, itrial);
            
            % R-squared
            trial_metrics(ianimal).r2(itrial) = 1 - sum((true_pos - pred_pos).^2) / ...
                sum((true_pos - mean(true_pos)).^2);
            
            % RMSE
            trial_metrics(ianimal).rmse(itrial) = sqrt(mean((pred_pos - true_pos).^2));
            
            % MAE
            trial_metrics(ianimal).mae(itrial) = mean(abs(pred_pos - true_pos));
        end
    end
    
    % Create figure
    figure('Position', [100 100 1200 800]);
    
    % 1. R-squared evolution
    subplot(2,2,1);
    plot_metric_evolution([trial_metrics.r2], 'R²');
    
    % 2. RMSE evolution
    subplot(2,2,2);
    plot_metric_evolution([trial_metrics.rmse], 'RMSE (cm)');
    
    % 3. MAE evolution
    subplot(2,2,3);
    plot_metric_evolution([trial_metrics.mae], 'MAE (cm)');
    
    % 4. Learning curve analysis
    subplot(2,2,4);
    window_size = 10; % trials to average over
    for ianimal = 1:n_animals
        smoothed_r2 = movmean(trial_metrics(ianimal).r2, window_size, 'omitnan');
        plot(smoothed_r2, 'LineWidth', 1.5);
        hold on;
    end
    
    % Add mean across animals
    all_r2 = cat(2, trial_metrics.r2);
    mean_r2 = mean(all_r2, 2, 'omitnan');
    smoothed_mean_r2 = movmean(mean_r2, window_size, 'omitnan');
    plot(smoothed_mean_r2, 'k-', 'LineWidth', 3);
    
    xlabel('Trial Number');
    ylabel('R² (10-trial moving average)');
    title('Learning Curves');
    legend([arrayfun(@(x) ['Animal ' num2str(x)], 1:n_animals, 'UniformOutput', false), {'Mean'}]);

    % Add overall title
    sgtitle('Evolution of Decoding Performance Across Trials', 'FontSize', 14);
end

function plot_metric_evolution(metrics, metric_name)
    n_animals = size(metrics, 2);
    
    % Plot individual animals
    plot(metrics, 'LineWidth', 1);
    hold on;
    
    % Plot mean and SEM
    mean_metric = mean(metrics, 2, 'omitnan');
    sem_metric = std(metrics, [], 2, 'omitnan') / sqrt(n_animals);
    
    % Plot mean with confidence bands
    x = 1:length(mean_metric);
    fill([x fliplr(x)], [mean_metric+sem_metric; flipud(mean_metric-sem_metric)], ...
        'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(mean_metric, 'k-', 'LineWidth', 2);
    
    xlabel('Trial Number');
    ylabel(metric_name);
    title(['Trial-by-Trial ' metric_name]);
end