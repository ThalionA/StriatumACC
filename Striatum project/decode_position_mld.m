function [decoded_positions, decoder_performance] = decode_position_mld(preprocessed_data, options)
    % Set default options
    default_options = struct('bin_size', 5, ...
                             'area', 'all', ...
                             'n_bootstraps', 100, ...
                             'neuron_counts', [15, 50, 100]);

    if nargin < 2
        options = default_options;
    else
        % Merge default options with provided options
        options = merge_options(default_options, options);
    end

    n_animals = length(preprocessed_data);
    decoder_performance = struct();
    decoded_positions = cell(1, n_animals);

    % Decode positions for each animal
    for ianimal = 1:n_animals
        fprintf('Decoding positions for animal %d using Gaussian MLD...\n', ianimal);

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
        decoder_performance(ianimal).per_trial_accuracy = nan(n_trials, n_counts, n_bootstraps);
        decoder_performance(ianimal).overall_accuracy = nan(n_counts, n_bootstraps);

        decoded_positions{ianimal} = cell(n_counts, n_bootstraps);

        % Loop over neuron counts
        for icount = 1:n_counts
            n_selected_neurons = neuron_counts(icount);
            fprintf('Animal %d: Decoding with %d neurons...\n', ianimal, n_selected_neurons);

            % Adjust n_bootstraps if using all neurons
            if n_selected_neurons == n_neurons
                n_iterations = 1;  % Only one iteration needed
            else
                n_iterations = n_bootstraps;
            end

            % Bootstrap iterations
            for ibootstrap = 1:n_iterations
                % Randomly select neurons if not using all neurons
                if n_selected_neurons == n_neurons
                    selected_neuron_idx = 1:n_neurons;  % Use all neurons
                else
                    selected_neuron_idx = randperm(n_neurons, n_selected_neurons);
                end
                spatial_binned_fr_boot = spatial_binned_fr(selected_neuron_idx, :, :);

                % Initialize decoded positions
                decoded_positions_matrix = nan(n_pos_bins, n_trials);

                % Leave-One-Out Cross-Validation over trials
                for ileaveout = 1:n_trials
                    % Indices for training and test
                    train_trials = true(n_trials, 1);
                    train_trials(ileaveout) = false;
                    test_trial = ileaveout;

                    % Compute tuning curves (mean and std) using training trials
                    fr_train = spatial_binned_fr_boot(:, :, train_trials);  % n_neurons x n_pos_bins x n_train_trials
                    mu = mean(fr_train, 3);  % Mean firing rates
                    sigma = std(fr_train, 0, 3);  % Standard deviations

                    % Handle zero or near-zero standard deviations
                    epsilon = 1e-6;
                    sigma(sigma < epsilon) = epsilon;

                    % Get observed firing rates for the test trial
                    fr_test = spatial_binned_fr_boot(:, :, test_trial);  % n_neurons x n_pos_bins

                    % Decode each position bin in the test trial
                    for ibin = 1:n_pos_bins
                        % Observed firing rates for this bin
                        fr_observed = fr_test(:, ibin);

                        % Compute log-likelihoods for each position
                        log_likelihoods = zeros(n_pos_bins, 1);

                        for ipos = 1:n_pos_bins
                            % Expected mean and std from tuning curves
                            mu_i = mu(:, ipos);
                            sigma_i = sigma(:, ipos);

                            % Compute Gaussian log-likelihood
                            diff = fr_observed - mu_i;
                            log_likelihoods(ipos) = -0.5 * sum(log(2 * pi * sigma_i.^2) + (diff.^2) ./ (sigma_i.^2));
                        end

                        % Decode the position with the maximum log-likelihood
                        [~, decoded_pos_bin] = max(log_likelihoods);

                        % Store the decoded position
                        decoded_positions_matrix(ibin, test_trial) = decoded_pos_bin * options.bin_size;
                    end
                end

                % Determine the index for storing results
                if n_selected_neurons == n_neurons
                    store_idx = 1;  % Only one iteration
                else
                    store_idx = ibootstrap;
                end

                decoded_positions{ianimal}{icount, store_idx} = decoded_positions_matrix;

                % Prepare true positions
                true_positions_matrix = repmat((1:n_pos_bins)' * options.bin_size, 1, n_trials);
                decoded_positions_vector = decoded_positions_matrix(:);
                true_positions_vector = true_positions_matrix(:);

                % Compute performance metrics
                [rmse, r2, mae, position_errors] = compute_performance_metrics(decoded_positions_vector, true_positions_vector, n_pos_bins);

                % Store performance metrics
                decoder_performance(ianimal).rmse(icount, store_idx) = rmse;
                decoder_performance(ianimal).r2(icount, store_idx) = r2;
                decoder_performance(ianimal).mae(icount, store_idx) = mae;
                decoder_performance(ianimal).position_errors(:, icount, store_idx) = position_errors;

                % Compute per-trial accuracy
                per_trial_accuracy = nan(n_trials, 1);
                for trial_index = 1:n_trials
                    decoded_positions_trial = decoded_positions_matrix(:, trial_index);
                    true_positions_trial = true_positions_matrix(:, trial_index);
                    % Compare decoded positions to true positions
                    correct_decodings = decoded_positions_trial == true_positions_trial;
                    per_trial_accuracy(trial_index) = sum(correct_decodings) / n_pos_bins;
                end

                % Store per-trial accuracy
                decoder_performance(ianimal).per_trial_accuracy(:, icount, store_idx) = per_trial_accuracy;

                % Compute overall accuracy
                overall_accuracy = sum(decoded_positions_vector == true_positions_vector) / numel(decoded_positions_vector);

                % Store overall accuracy
                decoder_performance(ianimal).overall_accuracy(icount, store_idx) = overall_accuracy;

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