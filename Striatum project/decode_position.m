function [decoded_positions, decoder_performance] = decode_position(preprocessed_data, options)
    % Default options
    if nargin < 2
        options.cv_folds = 5;
        options.model_type = 'ridge';  % 'ridge', 'linear', 'svm', or 'ann'
        options.bin_size = 4;
        options.area = 'all';  % 'all', 'DMS', 'DLS', or 'ACC'
        options.n_bootstraps = 100;
        options.neuron_counts = [1, 5, 10, 20, 50, 100];
    end

    n_animals = length(preprocessed_data);
    decoder_performance = struct();
    decoded_positions = cell(1, n_animals);

    % Get neuron counts per animal after area selection
    num_neurons_per_animal = zeros(n_animals, 1);
    for ianimal = 1:n_animals
        spatial_binned_fr = preprocessed_data(ianimal).z_spatial_binned_fr_all;
        
        % Select neurons based on area
        switch options.area
            case 'DMS'
                neuron_idx = preprocessed_data(ianimal).is_dms;
            case 'DLS'
                neuron_idx = preprocessed_data(ianimal).is_dls;
            case 'ACC'
                neuron_idx = preprocessed_data(ianimal).is_acc;
            otherwise
                neuron_idx = true(size(spatial_binned_fr, 1), 1);
        end
        
        num_neurons_per_animal(ianimal) = sum(neuron_idx);
    end

    for ianimal = 1:n_animals
        fprintf('Decoding positions for animal %d...\n', ianimal);
        
        % Get binned neural data and select neurons based on area
        spatial_binned_fr = preprocessed_data(ianimal).z_spatial_binned_fr_all;
        
        % Select neurons based on area
        switch options.area
            case 'DMS'
                neuron_idx = preprocessed_data(ianimal).is_dms;
            case 'DLS'
                neuron_idx = preprocessed_data(ianimal).is_dls;
            case 'ACC'
                neuron_idx = preprocessed_data(ianimal).is_acc;
            otherwise
                neuron_idx = true(size(spatial_binned_fr, 1), 1);
        end
        
        spatial_binned_fr = spatial_binned_fr(neuron_idx, :, :);
        [n_neurons, n_pos_bins, n_trials] = size(spatial_binned_fr);
        
        % Initialize storage for decoder performance and decoded positions
        neuron_counts = options.neuron_counts;
        n_counts = length(neuron_counts);
        n_bootstraps = options.n_bootstraps;
        decoder_performance(ianimal).neuron_counts = neuron_counts;
        decoder_performance(ianimal).rmse = nan(n_counts, n_bootstraps);
        decoder_performance(ianimal).r2 = nan(n_counts, n_bootstraps);
        decoder_performance(ianimal).mae = nan(n_counts, n_bootstraps);
        decoder_performance(ianimal).position_errors = nan(n_pos_bins, n_counts, n_bootstraps);
        
        decoded_positions{ianimal} = cell(n_counts, n_bootstraps);
        
        % Loop over neuron counts
        for icount = 1:n_counts
            n_selected_neurons = neuron_counts(icount);
            
            if n_neurons >= n_selected_neurons
                fprintf('Decoding with %d neurons...\n', n_selected_neurons);
                
                % Bootstrap iterations
                for ibootstrap = 1:n_bootstraps
                    % Randomly select neurons
                    selected_neuron_idx = randperm(n_neurons, n_selected_neurons);
                    spatial_binned_fr_boot = spatial_binned_fr(selected_neuron_idx, :, :);
                    
                    % Reshape data for decoding
                    X = reshape(spatial_binned_fr_boot, n_selected_neurons, [])'; % [n_timepoints x n_neurons]
                    true_positions = repmat(1:n_pos_bins, 1, n_trials)' * options.bin_size;
                    
                    % Cross-validation
                    cv_partition = cvpartition(size(X,1), 'KFold', options.cv_folds);
                    predicted_positions = zeros(size(X,1), 1);
                    
                    for ifold = 1:options.cv_folds
                        train_idx = training(cv_partition, ifold);
                        test_idx = test(cv_partition, ifold);
                        
                        % Train decoder based on selected model type
                        switch options.model_type
                            case 'ridge'
                                mdl = fitrlinear(X(train_idx,:), true_positions(train_idx), ...
                                    'Learner', 'leastsquares', 'Regularization', 'ridge', ...
                                    'Lambda', 1, 'Solver', 'lbfgs');
                                
                            case 'linear'
                                mdl = fitrlinear(X(train_idx,:), true_positions(train_idx), ...
                                    'Learner', 'leastsquares', 'Solver', 'lbfgs');
                                
                            case 'svm'
                                mdl = fitrsvm(X(train_idx,:), true_positions(train_idx), ...
                                    'KernelFunction', 'gaussian', ...
                                    'Standardize', true);
                                
                            case 'ann'
                                % Create and train neural network
                                net = fitnet([round(n_selected_neurons/2), round(n_selected_neurons/4)]);
                                net.trainParam.showWindow = false;
                                net.divideParam.trainRatio = 0.7;
                                net.divideParam.valRatio = 0.15;
                                net.divideParam.testRatio = 0.15;
                                net.performParam.normalization = 'standard';
                                [net, ~] = train(net, X(train_idx,:)', true_positions(train_idx)');
                        end
                        
                        % Test decoder
                        if strcmp(options.model_type, 'ann')
                            predicted_positions(test_idx) = net(X(test_idx,:)')';
                        else
                            predicted_positions(test_idx) = predict(mdl, X(test_idx,:));
                        end
                    end
                    
                    % Reshape predictions back to trials x positions
                    decoded_positions{ianimal}{icount, ibootstrap} = reshape(predicted_positions, n_pos_bins, n_trials);
                    
                    % Compute performance metrics
                    rmse = sqrt(mean((predicted_positions - true_positions).^2));
                    r2 = 1 - sum((true_positions - predicted_positions).^2) / ...
                        sum((true_positions - mean(true_positions)).^2);
                    mae = mean(abs(predicted_positions - true_positions));
                    
                    % Store performance metrics
                    decoder_performance(ianimal).rmse(icount, ibootstrap) = rmse;
                    decoder_performance(ianimal).r2(icount, ibootstrap) = r2;
                    decoder_performance(ianimal).mae(icount, ibootstrap) = mae;
                    
                    % Calculate error as function of position
                    position_errors = nan(n_pos_bins, 1);
                    for ibin = 1:n_pos_bins
                        bin_idx = true_positions == (ibin * options.bin_size);
                        position_errors(ibin) = mean(abs(predicted_positions(bin_idx) - true_positions(bin_idx)));
                    end
                    decoder_performance(ianimal).position_errors(:, icount, ibootstrap) = position_errors;
                end % End of bootstrap iterations
            else
                % Not enough neurons; store NaNs
                fprintf('Not enough neurons (%d) for decoding with %d neurons. Storing NaNs.\n', n_neurons, n_selected_neurons);
                for ibootstrap = 1:n_bootstraps
                    decoded_positions{ianimal}{icount, ibootstrap} = nan(n_pos_bins, n_trials);
                end
                decoder_performance(ianimal).rmse(icount, :) = nan;
                decoder_performance(ianimal).r2(icount, :) = nan;
                decoder_performance(ianimal).mae(icount, :) = nan;
                decoder_performance(ianimal).position_errors(:, icount, :) = nan;
            end
        end % End of neuron counts
        fprintf('Done with animal %d\n', ianimal);
    end % End of animals
end