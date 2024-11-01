function visualize_decoding_results(decoded_positions, decoder_performance, bin_size)
    n_animals = length(decoded_positions);
    
    % Create figure with subplots for each animal plus summary
    figure('Position', [100 100 1200 800]);
    
    % Individual animal plots
    for ianimal = 1:n_animals
        subplot(2, ceil((n_animals+1)/2), ianimal);
        
        % Get true positions
        n_pos_bins = size(decoded_positions{ianimal}, 1);
        true_positions = (1:n_pos_bins)' * bin_size;
        
        % Calculate mean and SEM of decoded positions
        mean_decoded = mean(decoded_positions{ianimal}, 2);
        sem_decoded = std(decoded_positions{ianimal}, [], 2) / sqrt(size(decoded_positions{ianimal}, 2));
        
        % Plot true vs decoded positions with error bars
        errorbar(true_positions, mean_decoded, sem_decoded, 'b.', 'MarkerSize', 15);
        hold on;
        % plot([0 max(true_positions)], [0 max(true_positions)], 'k--'); % Unity line
        identity_line
        xline([100, 125])
        yline([100, 125])
        
        xlabel('True Position (cm)');
        ylabel('Decoded Position (cm)');
        title(sprintf('Animal %d (R^2 = %.2f)', ianimal, decoder_performance(ianimal).r2));
        axis square;
    end
    
    % Summary plot
    subplot(2, ceil((n_animals+1)/2), n_animals + 1);
    
    % Aggregate position errors across animals
    all_position_errors = zeros(length(decoder_performance(1).position_errors), n_animals);
    for ianimal = 1:n_animals
        all_position_errors(:, ianimal) = decoder_performance(ianimal).position_errors;
    end
    
    % Calculate mean and SEM of position errors
    mean_position_errors = mean(all_position_errors, 2);
    sem_position_errors = std(all_position_errors, [], 2) / sqrt(n_animals);
    positions = (1:length(mean_position_errors))' * bin_size;
    
    % Plot average error by position
    errorbar(positions, mean_position_errors, sem_position_errors, 'r.', 'MarkerSize', 15);
    xlabel('Position (cm)');
    ylabel('Mean Absolute Error (cm)');
    title('Average Decoding Error by Position');
    axis square;
    xline([100, 125])
    
    % Print summary statistics
    fprintf('\nDecoding Performance Summary:\n');
    fprintf('---------------------------\n');
    all_r2 = [decoder_performance.r2];
    all_rmse = [decoder_performance.rmse];
    all_mae = [decoder_performance.mae];
    
    fprintf('Mean R² (± SEM): %.3f ± %.3f\n', mean(all_r2), std(all_r2)/sqrt(n_animals));
    fprintf('Mean RMSE (± SEM): %.3f ± %.3f cm\n', mean(all_rmse), std(all_rmse)/sqrt(n_animals));
    fprintf('Mean MAE (± SEM): %.3f ± %.3f cm\n', mean(all_mae), std(all_mae)/sqrt(n_animals));
end