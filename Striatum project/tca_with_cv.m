function tca_with_cv(data, tca_type, normalisation_to_apply, cross_validation_fold, maxNumFactors, max_iterations)
    % Default values for optional parameters
    if nargin < 2 || isempty(tca_type)
        tca_type = 'cp_als';  % default TCA method
    end

    if nargin < 3 || isempty(normalisation_to_apply)
        normalisation_to_apply = 'none';  % default normalization
    end

    if nargin < 4 || isempty(cross_validation_fold)
        cross_validation_fold = 5;  % default number of cross-validation folds
    end

    if nargin < 5 || isempty(maxNumFactors)
        maxNumFactors = 10;
    end

    if nargin < 6 || isempty(max_iterations)
        max_iterations = 200;  % default number of iterations for TCA
    end
    
    % Extract size of the data
    [num_neurons, num_bins, num_trials] = size(data);

    % Normalisation
    switch lower(normalisation_to_apply)
        case 'none'
            data_in = data;
        case 'z-score'
            data_reshaped = reshape(data, num_neurons, []);
            data_zscored_reshaped = zscore(data_reshaped, 0, 2);
            data_in = reshape(data_zscored_reshaped, num_neurons, num_bins, num_trials);
        case 'min-max'
            data_reshaped = reshape(data, num_neurons, []);
            data_min = min(data_reshaped, [], 2);
            data_max = max(data_reshaped, [], 2);
            data_scaled = (data_reshaped - data_min) ./ (data_max - data_min);
            data_scaled(isnan(data_scaled)) = 0;  % Replace NaNs
            data_in = reshape(data_scaled, num_neurons, num_bins, num_trials);
        otherwise
            error('Unknown normalization method');
    end

    % Cross-validation setup
    K = cross_validation_fold;
    c = cvpartition(num_trials, 'KFold', K);

    % Cross-validation loop
    
    cv_errors = zeros(maxNumFactors, K);
    options = struct('maxiters', max_iterations, 'tol', 1e-6, 'printitn', 0, 'stop_orth', 10);

    for nFactors = 1:maxNumFactors
        fprintf('Testing %d factors...\n', nFactors);
        for ifold = 1:K
            fprintf('  Fold %d/%d\n', ifold, K);
            % Split data into training and testing sets
            test_idx = c.test(ifold);
            train_idx = c.training(ifold);
            
            data_train = tensor(data_in(:, :, train_idx));
            data_test = tensor(data_in(:, :, test_idx));

            % Choose TCA method based on input
            switch lower(tca_type)
                case 'cp_als'
                    P = cp_als(data_train, nFactors, options);
                case 'cp_nmu'
                    P = cp_nmu(data_train, nFactors, options);
                case 'cp_orth_als'
                    P = cp_orth_als(data_train, nFactors, options);
                otherwise
                    error('Unknown TCA method');
            end

            % Extract factor matrices
            A = P.U{1};  % Neurons factors
            B = P.U{2};  % Spatial bins factors

            % Fit trial factors for validation data
            data_test_unfold = tenmat(data_test, 3);  % Unfold along trials
            AB_kr = khatrirao(A, B);  % Khatri-Rao product

            num_test_trials = sum(test_idx);
            C_test = zeros(num_test_trials, nFactors);  % Initialize C_test

            for t = 1:num_test_trials
                data_vec = data_test_unfold.data(t, :)';
                C_test(t, :) = lsqnonneg(AB_kr, data_vec)';  % Solve using NNLS
            end

            % Construct the model for validation data
            P_test = ktensor({A, B, C_test});

            % Compute reconstruction error on validation data
            error = norm(data_test - tensor(P_test))^2 / norm(data_test)^2;
            cv_errors(nFactors, ifold) = error;
        end
    end

    % Compute mean and SEM of cross-validation errors
    mean_cv_errors = mean(cv_errors, 2);
    sem_cv_errors = std(cv_errors, 0, 2) / sqrt(K);

    % Plot results
    figure
    shadedErrorBar(1:maxNumFactors, mean_cv_errors, sem_cv_errors);
    xlabel('Number of Factors');
    ylabel('Mean CV Reconstruction Error');
    title('Cross-Validation Error vs. Number of Factors');

    % Select the best number of factors
    [~, best_nFactors] = min(mean_cv_errors);

    % Fit the final model using the best number of factors
    switch lower(tca_type)
        case 'cp_als'
            best_mdl = cp_als(tensor(data_in), best_nFactors, options);
        case 'cp_nmu'
            best_mdl = cp_nmu(tensor(data_in), best_nFactors, options);
        case 'cp_orth_als'
            best_mdl = cp_orth_als(tensor(data_in), best_nFactors, options);
    end

    % Plot factor matrices (example: neurons and spatial bins)
    figure
    t = tiledlayout(best_nFactors, 1);
    for iFactor = 1:best_nFactors
        nexttile
        bar(best_mdl.U{1}(:, iFactor))
        xlabel('Neuron #')
        axis tight
    end
    xlabel(t, 'Neuron Factors');

    figure
    t = tiledlayout(best_nFactors, 1);
    for iFactor = 1:best_nFactors
        nexttile
        plot(best_mdl.U{2}(:, iFactor))
        xlabel('Spatial Bin #')
        axis tight
    end
    xlabel(t, 'Spatial Bin Factors');
end