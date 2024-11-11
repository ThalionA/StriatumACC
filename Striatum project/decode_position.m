function [decoded_positions, decoder_performance] = decode_position(preprocessed_data, options)
    % Set default options
    default_options = struct('model_type', 'ridge', ...
                             'bin_size', 5, ...
                             'area', 'all', ...
                             'n_bootstraps', 10, ...
                             'neuron_counts', [15, 50, 100]);
    
    if nargin < 2
        options = default_options;
    else
        % Merge default options with provided options
        options = merge_options(default_options, options);
    end

    % Validate the model_type option
    valid_model_types = {'ridge', 'linear'};
    if ~ismember(lower(options.model_type), valid_model_types)
        error('Invalid model_type: %s. Valid options are ''ridge'' or ''linear''.', options.model_type);
    end

    n_animals = length(preprocessed_data);
    decoder_performance = struct();
    decoded_positions = cell(1, n_animals);

    % Decode positions for each animal
    for ianimal = 1:n_animals
        fprintf('Decoding positions for animal %d using leave-one-out cross-validation...\n', ianimal);

        % Get binned neural data and select neurons based on area
        spatial_binned_fr = preprocessed_data(ianimal).spatial_binned_fr_all;

        % Select neurons based on area
        switch lower(options.area)
            case 'dms'
                neuron_idx = preprocessed_data(ianimal).is_dms;
            case 'dls'
                neuron_idx = preprocessed_data(ianimal).is_dls;
            case 'acc'
                neuron_idx = preprocessed_data(ianimal).is_acc;
            otherwise
                neuron_idx = true(size(spatial_binned_fr, 1), 1);
        end

        spatial_binned_fr = spatial_binned_fr(neuron_idx, :, :);
        [n_neurons, n_pos_bins, n_trials] = size(spatial_binned_fr);

        % Handle cases where there are no neurons in the specified area
        if n_neurons == 0
            warning('Animal %d has no neurons in area %s. Skipping.', ianimal, options.area);
            continue;
        end

        % Determine neuron counts to use for this animal
        neuron_counts = options.neuron_counts(options.neuron_counts <= n_neurons);
        if isempty(neuron_counts)
            neuron_counts = n_neurons;
        elseif ~ismember(n_neurons, neuron_counts)
            neuron_counts = [neuron_counts, n_neurons];
        end
        neuron_counts = unique(neuron_counts);

        n_counts = length(neuron_counts);
        n_bootstraps = options.n_bootstraps;

        % Initialize storage
        decoder_performance(ianimal).neuron_counts = neuron_counts;
        decoder_performance(ianimal).rmse = nan(n_counts, n_bootstraps);
        decoder_performance(ianimal).r2 = nan(n_counts, n_bootstraps);
        decoder_performance(ianimal).mae = nan(n_counts, n_bootstraps);
        decoder_performance(ianimal).position_errors = nan(n_pos_bins, n_counts, n_bootstraps);

        decoded_positions{ianimal} = cell(n_counts, n_bootstraps);

        % Loop over neuron counts
        for icount = 1:n_counts
            n_selected_neurons = neuron_counts(icount);
            fprintf('Animal %d: Decoding with %d neurons...\n', ianimal, n_selected_neurons);

            if n_selected_neurons == n_neurons
                n_iterations = 1;  % Only one iteration needed
            else
                n_iterations = n_bootstraps;
            end

            % Bootstrap iterations
            for ibootstrap = 1:n_iterations
                % Randomly select neurons
                if n_selected_neurons == n_neurons
                    selected_neuron_idx = 1:n_neurons;  % Use all neurons
                else
                    selected_neuron_idx = randperm(n_neurons, n_selected_neurons);
                end
                spatial_binned_fr_boot = spatial_binned_fr(selected_neuron_idx, :, :);

                % Prepare data for decoding
                X = reshape(spatial_binned_fr_boot, n_selected_neurons, [])';
                true_positions = repmat((1:n_pos_bins)', n_trials, 1) * options.bin_size;

                % Indices to map samples to trials
                trial_indices = repmat(1:n_trials, n_pos_bins, 1);
                trial_indices = trial_indices(:);

                % Initialize predicted positions
                predicted_positions = nan(size(true_positions));

                % Leave-One-Out Cross-Validation over trials
                for ileaveout = 1:n_trials
                    % Training and test indices
                    train_trials = true(n_trials, 1);
                    train_trials(ileaveout) = false;
                    test_trials = ~train_trials;

                    train_idx = ismember(trial_indices, find(train_trials));
                    test_idx = ismember(trial_indices, find(test_trials));

                    % Check if there are any training samples
                    if sum(train_idx) == 0
                        warning('No training samples available after leaving out trial %d. Skipping this iteration.', ileaveout);
                        continue;
                    end

                    % Train decoder based on selected model type
                    mdl = train_decoder(X(train_idx, :), true_positions(train_idx), options);

                    % Test decoder
                    predicted_positions(test_idx) = test_decoder(mdl, X(test_idx, :), options);
                end

                % Reshape predictions back to positions x trials
                decoded_positions_matrix = reshape(predicted_positions, n_pos_bins, n_trials);

                % Determine the index for storing results
                if n_selected_neurons == n_neurons
                    store_idx = 1;  % Only one iteration
                else
                    store_idx = ibootstrap;
                end

                decoded_positions{ianimal}{icount, store_idx} = decoded_positions_matrix;

                % Compute performance metrics
                [rmse, r2, mae, position_errors] = compute_performance_metrics(predicted_positions, true_positions, n_pos_bins);

                % Store performance metrics
                decoder_performance(ianimal).rmse(icount, store_idx) = rmse;
                decoder_performance(ianimal).r2(icount, store_idx) = r2;
                decoder_performance(ianimal).mae(icount, store_idx) = mae;
                decoder_performance(ianimal).position_errors(:, icount, store_idx) = position_errors;
            end % End of bootstrap iterations
        end % End of neuron counts
        fprintf('Done with animal %d\n', ianimal);
    end % End of animals
end

function options = merge_options(default_options, user_options)
    options = default_options;
    fields = fieldnames(user_options);
    for i = 1:length(fields)
        options.(fields{i}) = user_options.(fields{i});
    end
end

function mdl = train_decoder(X_train, y_train, options)
    switch lower(options.model_type)
        case 'ridge'
            mdl = fitrlinear(X_train, y_train, ...
                'Learner', 'leastsquares', 'Regularization', 'ridge', ...
                'Lambda', 1, 'Solver', 'lbfgs', 'Verbose', 0);
        case 'linear'
            mdl = fitrlinear(X_train, y_train, ...
                'Learner', 'leastsquares', 'Solver', 'lbfgs', 'Verbose', 0);
        otherwise
            error('Unknown model type: %s', options.model_type);
    end
end

function y_pred = test_decoder(mdl, X_test, options)
    y_pred = predict(mdl, X_test);
end

function [rmse, r2, mae, position_errors] = compute_performance_metrics(y_pred, y_true, n_pos_bins)
    rmse = sqrt(mean((y_pred - y_true).^2, 'omitnan'));
    ss_res = sum((y_true - y_pred).^2, 'omitnan');
    ss_tot = sum((y_true - mean(y_true, 'omitnan')).^2, 'omitnan');
    r2 = 1 - ss_res / ss_tot;
    mae = mean(abs(y_pred - y_true), 'omitnan');

    % Calculate error as a function of position
    position_errors = nan(n_pos_bins, 1);
    positions = unique(y_true);
    for i = 1:length(positions)
        idx = y_true == positions(i);
        position_errors(i) = mean(abs(y_pred(idx) - y_true(idx)), 'omitnan');
    end
end