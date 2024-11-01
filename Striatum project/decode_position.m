function [decoded_positions, decoder_performance] = decode_position(preprocessed_data, options)
    % Default options
    if nargin < 2
        options.cv_folds = 5;
        options.model_type = 'ridge';  % 'ridge', 'linear', 'svm', or 'ann'
        options.bin_size = 4;
        options.area = 'all';  % 'all', 'DMS', 'DLS', or 'ACC'
    end

    n_animals = length(preprocessed_data);
    decoder_performance = struct();
    decoded_positions = cell(1, n_animals);
    
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
        
        % Reshape data for decoding
        X = reshape(spatial_binned_fr, n_neurons, [])'; % [n_timepoints x n_neurons]
        true_positions = repmat(1:n_pos_bins, 1, n_trials)' * options.bin_size;
        
        % Z-score features
        % X = zscore(X, 0, 1); % Z-score across samples (rows)
        
        % Initialize storage for cross-validation results
        cv_partition = cvpartition(size(X,1), 'KFold', options.cv_folds);
        predicted_positions = zeros(size(X,1), 1);
        
        % Cross-validation loop
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
                    net = fitnet([round(n_neurons/2), round(n_neurons/4)]);
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
        decoded_positions{ianimal} = reshape(predicted_positions, n_pos_bins, n_trials);
        
        % Calculate performance metrics
        decoder_performance(ianimal).rmse = sqrt(mean((predicted_positions - true_positions).^2));
        decoder_performance(ianimal).r2 = 1 - sum((true_positions - predicted_positions).^2) / ...
            sum((true_positions - mean(true_positions)).^2);
        decoder_performance(ianimal).mae = mean(abs(predicted_positions - true_positions));
        
        % Calculate error as function of position
        position_errors = nan(n_pos_bins, 1);
        for ibin = 1:n_pos_bins
            bin_idx = true_positions == (ibin * options.bin_size);
            position_errors(ibin) = mean(abs(predicted_positions(bin_idx) - true_positions(bin_idx)));
        end
        decoder_performance(ianimal).position_errors = position_errors;                
        
        fprintf('Done with animal %d\n', ianimal);
    end
end

