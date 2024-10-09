function tca_with_cv(data, tca_type, normalisation_to_apply, cross_validation_fold, maxNumFactors, max_iterations, xlines_to_plot)
    % tca_with_cv performs tensor component analysis with cross-validation.
    
    % Input argument validation and default values
    arguments
        data
        tca_type (1,:) char {mustBeMember(tca_type, {'cp_als', 'cp_nmu', 'cp_orth_als'})} = 'cp_als'
        normalisation_to_apply (1,:) char {mustBeMember(normalisation_to_apply, {'none', 'z-score', 'min-max'})} = 'none'
        cross_validation_fold (1,1) double {mustBeInteger, mustBePositive} = 5
        maxNumFactors (1,1) double {mustBeInteger, mustBePositive} = 10
        max_iterations (1,1) double {mustBeInteger, mustBePositive} = 200
        xlines_to_plot (1,3) double = nan(1,3)
    end
    
    clc

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
    options = struct('maxiters', max_iterations, 'tol', 1e-6, 'printitn', 0);
    
    for nFactors = 1:maxNumFactors
        fprintf('Testing %d factors...\n', nFactors);
        for ifold = 1:K
            fprintf('  Fold %d/%d\n', ifold, K);
            % Split data into training and testing sets
            test_idx = c.test(ifold);
            train_idx = c.training(ifold);
    
            data_train = tensor(data_in(:, :, train_idx));
            data_test = tensor(data_in(:, :, test_idx));

            min_error = Inf;
            num_initializations = 5; % Number of times to fit the model per fold
            for init = 1:num_initializations

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
                train_error = norm(data_train - tensor(P))^2 / norm(data_train)^2;
                if train_error < min_error
                    min_error = train_error;
                    best_P = P;
                end
            end
    
            % Extract factor matrices
            A = best_P.U{1};  % Neurons factors
            B = best_P.U{2};  % Spatial bins factors
    
            % Fit trial factors for validation data
            data_test_unfold = tenmat(data_test, 3);  % Unfold along trials
            AB_kr = khatrirao(A, B);  % Khatri-Rao product
    
            num_test_trials = sum(test_idx);
            C_test = zeros(num_test_trials, nFactors);  % Initialize C_test
    
            for t = 1:num_test_trials
                data_vec = data_test_unfold.data(t, :)';
                
                switch lower(tca_type)
                    case 'cp_als'
                        % Solve using regular least squares
                        C_test(t, :) = (AB_kr \ data_vec)';
                    case 'cp_nmu'
                        % Solve using non-negative least squares
                        C_test(t, :) = lsqnonneg(AB_kr, data_vec)';
                    case 'cp_orth_als'
                        % Project data onto orthonormal basis
                        C_test(t, :) = (AB_kr' * data_vec)';
                    otherwise
                        error('Unknown TCA method');
                end
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
    [~, best_nFactors] = min(mean_cv_errors(2:end));
    best_nFactors = best_nFactors + 1;
    
    num_initializations = 10; % Number of times to fit the final model
    best_error = Inf;
    for init = 1:num_initializations
            % Fit the final model using the best number of factors
            switch lower(tca_type)
                case 'cp_als'
                    P = cp_als(tensor(data_in), best_nFactors, options);
                case 'cp_nmu'
                    P = cp_nmu(tensor(data_in), best_nFactors, options);
                case 'cp_orth_als'
                    P = cp_orth_als(tensor(data_in), best_nFactors, options);
            end
        total_error = norm(tensor(data_in) - tensor(P))^2 / norm(tensor(data_in))^2;
        if total_error < best_error
            best_error = total_error;
            best_mdl = P;
        end
    end



    % Compute variance explained by each component and reorder
    component_contributions = zeros(best_nFactors, 1);
    total_ss = norm(tensor(data_in))^2;

    for r = 1:best_nFactors
        lambda_r = best_mdl.lambda(r);
        a_r = best_mdl.U{1}(:, r);
        b_r = best_mdl.U{2}(:, r);
        c_r = best_mdl.U{3}(:, r);

        component_tensor = ktensor(lambda_r, {a_r, b_r, c_r});
        component_ss = norm(tensor(component_tensor))^2;
        component_contributions(r) = component_ss;
    end

    variance_explained = component_contributions / total_ss;
    [~, sorted_indices] = sort(variance_explained, 'descend');

    % Reorder the model's factor matrices and weights
    best_mdl.lambda = best_mdl.lambda(sorted_indices);
    best_mdl.U{1} = best_mdl.U{1}(:, sorted_indices);
    best_mdl.U{2} = best_mdl.U{2}(:, sorted_indices);
    best_mdl.U{3} = best_mdl.U{3}(:, sorted_indices);
    variance_explained = variance_explained(sorted_indices);

    % Display variance explained by each component
    fprintf('Variance explained by each component:\n');
    for r = 1:best_nFactors
        fprintf('Component %d: %.2f%%\n', r, variance_explained(r) * 100);
    end
    
    % Plot factor matrices (ordered by variance explained)
    % Plot neuron factors
    figure
    t = tiledlayout(best_nFactors, 1, "TileSpacing", "compact");
    for iFactor = 1:best_nFactors
        nexttile
        bar(best_mdl.U{1}(:, iFactor))
        xline(xlines_to_plot(1))
        axis tight
        if iFactor ~= best_nFactors
            xticks([])
        end
        ylabel(sprintf('Comp %d', iFactor))
        title(sprintf('Variance Explained: %.2f%%', variance_explained(iFactor) * 100))
        linkaxes
    end
    xlabel(t, 'Neuron #');
    title(t, tca_type)
    
    % Plot spatial factors
    figure
    t = tiledlayout(best_nFactors, 1, "TileSpacing", "compact");
    for iFactor = 1:best_nFactors
        nexttile
        plot(best_mdl.U{2}(:, iFactor))
        xline(xlines_to_plot(2))
        axis tight
        if iFactor ~= best_nFactors
            xticks([])
        end
        ylabel(sprintf('Comp %d', iFactor))
        title(sprintf('Variance Explained: %.2f%%', variance_explained(iFactor) * 100))
        linkaxes
    end
    xlabel(t, 'Spatial Bin');
    title(t, tca_type)
    
    % Plot trial factors
    figure
    t = tiledlayout(best_nFactors, 1, "TileSpacing", "compact");
    for iFactor = 1:best_nFactors
        nexttile
        shadedErrorBar(1:num_trials, movmean(best_mdl.U{3}(:, iFactor), 10), ...
                       movstd(best_mdl.U{3}(:, iFactor), 10)/sqrt(10))
        xline(xlines_to_plot(3))
        axis tight
        if iFactor ~= best_nFactors
            xticks([])
        end
        ylabel(sprintf('Comp %d', iFactor))
        title(sprintf('Variance Explained: %.2f%%', variance_explained(iFactor) * 100))
        linkaxes
    end
    xlabel(t, 'Trial #')
    title(t, tca_type)
    
end