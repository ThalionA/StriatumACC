function visualize_performance_vs_neuron_count(decoder_performance)
    n_animals = length(decoder_performance);
    
    % Define the performance metrics to plot
    metrics = {'r2'};
    metric_titles = {'R²'};
    
    % Create a figure for each metric
    for m = 1:length(metrics)
        figure('Name', sprintf('Performance Scaling - %s', metric_titles{m}), 'Position', [100 100 1200 800]);
        
        % Loop over each animal
        for ianimal = 1:n_animals
            neuron_counts = decoder_performance(ianimal).neuron_counts;
            
            % Extract performance data for the current metric, ignoring NaNs
            metric_data = decoder_performance(ianimal).(metrics{m});
            mean_metric_data = mean(metric_data, 2, 'omitnan'); % Average across bootstraps
            sem_metric_data = std(metric_data, [], 2, 'omitnan') / sqrt(size(metric_data, 2)); % SEM across bootstraps
            
            % Plot the metric vs neuron count
            subplot(ceil(n_animals / 2), 2, ianimal);
            errorbar(neuron_counts, mean_metric_data, sem_metric_data, '-o', 'MarkerSize', 6, 'LineWidth', 1.5);
            hold on;
            
            xlabel('Neuron Count');
            ylabel(metric_titles{m});
            title(sprintf('Animal %d', ianimal));
            axis tight;
            box off
        end
        linkaxes
        
        % Add a main title for each metric
        sgtitle(sprintf('Performance Scaling with Neuron Count (%s)', metric_titles{m}), 'FontSize', 14);
    end
end