function [best_mdl, variance_explained, bic_values, best_nFactors] = tca_with_bic(data, tca_type, normalisation_to_apply, maxNumFactors, max_iterations, num_initializations)
% tca_with_bic performs tensor component analysis using BIC for model selection.
% Instead of simply choosing the model with the lowest BIC, it selects the model
% with the fewest factors (greater than 1) whose BIC is within a specified tolerance 
% of the minimum BIC, analogous to selecting an "elbow" in the curve.
%
% INPUTS:
%   data                   - 3D array (neurons x bins x trials)
%   tca_type               - TCA method ('cp_als', 'cp_nmu', 'cp_orth_als')
%   normalisation_to_apply - Normalization method ('none', 'z-score', 'min-max')
%   maxNumFactors          - Maximum number of factors to test
%   max_iterations         - Maximum iterations for fitting
%   num_initializations    - Number of initialisations
%
% OUTPUTS:
%   best_mdl         - Best TCA model (ktensor) selected based on BIC and simplicity
%   variance_explained - Variance explained by each component (ordered descending)
%   bic_values       - BIC values for models with 1 to maxNumFactors factors
%   best_nFactors    - The selected number of factors
%
arguments
    data
    tca_type (1,:) char {mustBeMember(tca_type, {'cp_als', 'cp_nmu', 'cp_orth_als'})} = 'cp_als'
    normalisation_to_apply (1,:) char {mustBeMember(normalisation_to_apply, {'none', 'z-score', 'min-max'})} = 'none'
    maxNumFactors (1,1) double {mustBeInteger, mustBePositive} = 10
    max_iterations (1,1) double {mustBeInteger, mustBePositive} = 200
    num_initializations (1, 1) double {mustBeInteger, mustBePositive} = 20
end

% Precompute lowercase strings to avoid repeated calls to lower()
tca_method = lower(tca_type);
norm_method = lower(normalisation_to_apply);

% Get dimensions of the data
[I, J, K] = size(data);

% Normalize the data
data_in = apply_normalisation(data, norm_method, I, J, K);

% Convert data to a tensor (only once)
tensor_data = tensor(data_in);

% Preallocate arrays for BIC values and store candidate models
bic_values = nan(maxNumFactors, 1);
candidate_models = cell(maxNumFactors, 1);

% Number of observations for BIC calculation
n = numel(data_in);

% Options for TCA fitting
options = struct('maxiters', max_iterations, 'tol', 1e-6, 'printitn', 0);

% Loop over candidate number of factors
for nFactors = 2:maxNumFactors
    fprintf('Testing model with %d factors...\n', nFactors);
    best_error = Inf;
    best_P = [];
    
    % Multiple initializations to mitigate local minima
    for init = 1:num_initializations
        P = fit_tca_model(tensor_data, nFactors, tca_method, options);
        % Reconstruction error (relative to full data)
        rec_error = norm(tensor_data - tensor(P))^2 / norm(tensor_data)^2;
        if rec_error < best_error
            best_error = rec_error;
            best_P = P;
        end
    end
    
    % Compute the residual sum of squares (RSS)
    RSS = norm(tensor_data - tensor(best_P))^2;
    
    % Estimate the number of free parameters: p = nFactors*(I + J + K) - 2*nFactors
    p = nFactors * (I + J + K);
    
    % Compute BIC: BIC = n * log(RSS/n) + p * log(n)
    bic = n * log(RSS/n) + p * log(n);
    bic_values(nFactors) = bic;
    
    % Store the best model for this candidate number of factors
    candidate_models{nFactors} = best_P;
end

tol = 0.01;

% Find minimum BIC and its index (considering only factors >= 2)
valid_bics = bic_values(2:end);
if isempty(valid_bics) || all(isnan(valid_bics))
     error('BIC calculation failed for all factor numbers.');
end
[min_bic_val, min_idx_local] = min(valid_bics);
min_idx_global = min_idx_local + 1; % Adjust index back to original array (2 to maxNumFactors)

% Define the threshold (e.g., min_bic + 1% of its magnitude)
% Use abs() in case BIC values are negative, though usually positive.
bic_threshold = min_bic_val + abs(min_bic_val) * tol;

% Find indices of models within the tolerance threshold (starting from index 2)
candidate_indices = find(bic_values(2:maxNumFactors) <= bic_threshold);

if isempty(candidate_indices)
    % Fallback: If no model is within tolerance (unlikely with revised logic), choose the minimum BIC model.
    best_nFactors = min_idx_global;
    fprintf('Warning: No model found within tolerance. Selecting model with minimum BIC.\n');
else
    % Select the smallest factor number among the candidates
    best_nFactors = min(candidate_indices) + 1; % Adjust index back to original array
end

% Retrieve the model (ensure candidate_models corresponds to indices 1:maxNumFactors)
best_mdl = candidate_models{best_nFactors};

fprintf('Selected model with %d factors (BIC = %.2f, within %.1f%% of minimum BIC = %.2f).\n', ...
    best_nFactors, bic_values(best_nFactors), tol*100, min_bic);

% Compute variance explained by each component
component_contributions = zeros(best_nFactors, 1);
total_ss = norm(tensor_data)^2;
for r = 1:best_nFactors
    lambda_r = best_mdl.lambda(r);
    a_r = best_mdl.U{1}(:, r);
    b_r = best_mdl.U{2}(:, r);
    c_r = best_mdl.U{3}(:, r);
    
    component_tensor = ktensor(lambda_r, {a_r, b_r, c_r});
    component_contributions(r) = norm(tensor(component_tensor))^2;
end
variance_explained = component_contributions / total_ss;

% Reorder components in descending order of variance explained
[~, sorted_indices] = sort(variance_explained, 'descend');
best_mdl.lambda = best_mdl.lambda(sorted_indices);
best_mdl.U{1} = best_mdl.U{1}(:, sorted_indices);
best_mdl.U{2} = best_mdl.U{2}(:, sorted_indices);
best_mdl.U{3} = best_mdl.U{3}(:, sorted_indices);
variance_explained = variance_explained(sorted_indices);

end

%% Helper Functions

function data_out = apply_normalisation(data, norm_method, I, J, K)
    % Apply the chosen normalization method to the data.
    switch norm_method
        case 'none'
            data_out = data;
        case 'z-score'
            data_reshaped = reshape(data, I, []);
            data_z = zscore(data_reshaped, 0, 2);
            data_out = reshape(data_z, I, J, K);
        case 'min-max'
            data_reshaped = reshape(data, I, []);
            data_min = min(data_reshaped, [], 2);
            data_max = max(data_reshaped, [], 2);
            data_scaled = (data_reshaped - data_min) ./ (data_max - data_min);
            data_scaled(isnan(data_scaled)) = 0;
            data_out = reshape(data_scaled, I, J, K);
        otherwise
            error('Unknown normalization method');
    end
end

function P = fit_tca_model(tensor_data, nFactors, tca_method, options)
    % Fit a TCA model using the specified method and options.
    switch tca_method
        case 'cp_als'
            P = cp_als(tensor_data, nFactors, options);
        case 'cp_nmu'
            P = cp_nmu(tensor_data, nFactors, options);
        case 'cp_orth_als'
            P = cp_orth_als(tensor_data, nFactors, options);
        otherwise
            error('Unknown TCA method');
    end
end