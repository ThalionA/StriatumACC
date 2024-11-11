function visualize_decoding_results(decoded_positions, decoder_performance, bin_size)
    n_animals = length(decoded_positions);
    
    % Loop over neuron counts to visualize
    neuron_counts = unique([decoder_performance(:).neuron_counts]);  % Collect unique neuron counts across animals
    n_neuron_counts = length(neuron_counts);
    
    for icount = 1:n_neuron_counts
        neuron_count = neuron_counts(icount);
        fprintf('\nVisualizing results for neuron count: %d\n', neuron_count);
        
        % Check which animals have valid data for this neuron count
        valid_animals = [];
        for ianimal = 1:n_animals
            if ismember(neuron_count, decoder_performance(ianimal).neuron_counts) && ...
               all(~isnan(decoder_performance(ianimal).r2(find(decoder_performance(ianimal).neuron_counts == neuron_count), :)))
                valid_animals = [valid_animals, ianimal];
            end
        end
        
        if isempty(valid_animals)
            fprintf('No animals have data for neuron count %d. Skipping visualization.\n', neuron_count);
            continue;
        end
        
        % Create figure for this neuron count
        figure('Name', sprintf('Decoding Results (Neuron Count: %d)', neuron_count), 'Position', [100 100 1200 800]);
        
        % Individual animal plots
        n_valid_animals = length(valid_animals);
        for idx = 1:n_valid_animals
            ianimal = valid_animals(idx);
            subplot(2, ceil((n_valid_animals+1)/2), idx);
            
            % Find index for the current neuron count
            icount_animal = find(decoder_performance(ianimal).neuron_counts == neuron_count);
            
            % Collect decoded positions across bootstraps
            decoded_pos_bootstraps = decoded_positions{ianimal}(icount_animal, :);
            n_bootstraps = length(decoded_pos_bootstraps);
            
            % Stack decoded positions for averaging
            n_pos_bins = size(decoded_pos_bootstraps{1}, 1);
            n_trials = size(decoded_pos_bootstraps{1}, 2);
            decoded_pos_matrix = zeros(n_pos_bins, n_trials, n_bootstraps);
            for ibootstrap = 1:n_bootstraps
                decoded_pos_matrix(:, :, ibootstrap) = decoded_pos_bootstraps{ibootstrap};
            end
            
            % Calculate mean and SEM across bootstraps
            mean_decoded = mean(decoded_pos_matrix, 3, 'omitnan');
            sem_decoded = std(decoded_pos_matrix, [], 3, 'omitnan') / sqrt(n_bootstraps);
            
            % Get true positions
            true_positions = (1:n_pos_bins)' * bin_size;
            
            % Plot true vs decoded positions with error bars
            errorbar(true_positions, mean(mean_decoded, 2), mean(sem_decoded, 2), 'b.', 'MarkerSize', 15);
            hold on;
            plot([min(true_positions), max(true_positions)], [min(true_positions), max(true_positions)], 'k--'); % Unity line
            xlabel('True Position (cm)');
            ylabel('Decoded Position (cm)');
            xline([100, 125]);
            yline([100, 125]);
            
            % Calculate mean R² across bootstraps
            mean_r2 = mean(decoder_performance(ianimal).r2(icount_animal, :), 'omitnan');
            
            title(sprintf('Animal %d (Mean R^2 = %.2f)', ianimal, mean_r2));
            axis square;
        end
        
        % Summary plot (e.g., average error by position across animals)
        subplot(2, ceil((n_valid_animals+1)/2), n_valid_animals + 1);
        
        % Aggregate position errors across valid animals
        all_position_errors = [];
        for ianimal = valid_animals
            icount_animal = find(decoder_performance(ianimal).neuron_counts == neuron_count);
            position_errors = decoder_performance(ianimal).position_errors(:, icount_animal, :);
            all_position_errors = cat(3, all_position_errors, position_errors);
        end
        
        % Calculate mean and SEM of position errors across animals and bootstraps
        mean_position_errors = mean(all_position_errors, [3], 'omitnan');
        sem_position_errors = std(all_position_errors, [], 3, 'omitnan') / sqrt(length(valid_animals));
        positions = (1:length(mean_position_errors))' * bin_size;
        
        % Plot average error by position
        errorbar(positions, mean(mean_position_errors, 2), sem_position_errors, 'r.', 'MarkerSize', 15);
        xlabel('Position (cm)');
        ylabel('Mean Absolute Error (cm)');
        title('Average Decoding Error by Position');
        axis square;
        
        % Print summary statistics
        fprintf('\nDecoding Performance Summary (Neuron Count: %d):\n', neuron_count);
        fprintf('------------------------------------------------\n');
        all_r2 = [];
        all_rmse = [];
        all_mae = [];
        for ianimal = valid_animals
            icount_animal = find(decoder_performance(ianimal).neuron_counts == neuron_count);
            all_r2 = [all_r2; decoder_performance(ianimal).r2(icount_animal, :)'];
            all_rmse = [all_rmse; decoder_performance(ianimal).rmse(icount_animal, :)'];
            all_mae = [all_mae; decoder_performance(ianimal).mae(icount_animal, :)'];
        end
        
        fprintf('Mean R² (± SEM): %.3f ± %.3f\n', mean(all_r2, 'omitnan'), std(all_r2, 'omitnan')/sqrt(length(all_r2)));
        fprintf('Mean RMSE (± SEM): %.3f ± %.3f cm\n', mean(all_rmse, 'omitnan'), std(all_rmse, 'omitnan')/sqrt(length(all_rmse)));
        fprintf('Mean MAE (± SEM): %.3f ± %.3f cm\n', mean(all_mae, 'omitnan'), std(all_mae, 'omitnan')/sqrt(length(all_mae)));
    end
end