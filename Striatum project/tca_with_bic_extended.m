function [best_mdl, variance_explained, bic_values, best_nFactors, all_best_models, recon_errors, init_similarity_scores] = tca_with_bic_extended(data, tca_type, normalisation_to_apply, maxNumFactors, max_iterations, num_initializations)
% tca_with_bic_extended performs TCA, selects model via BIC, and provides extended outputs.
% Selects the simplest model (fewest factors > 1) whose BIC is within a tolerance
% of the minimum BIC. Also returns best models, errors, and initialization similarity
% for all tested factor numbers.
%
% INPUTS:
%   data                   - 3D array (neurons x bins x trials)
%   tca_type               - TCA method ('cp_als', 'cp_nmu', 'cp_orth_als')
%   normalisation_to_apply - Normalization method ('none', 'z-score', 'min-max')
%   maxNumFactors          - Maximum number of factors to test (must be >= 2)
%   max_iterations         - Maximum iterations for fitting
%   num_initializations    - Number of random initializations for each factor count (default: 25)
%
% OUTPUTS:
%   best_mdl              - Best TCA model (ktensor) selected based on BIC and simplicity.
%   variance_explained    - Variance explained by each component of best_mdl (descending).
%   bic_values            - BIC values for models with 1 to maxNumFactors factors (index 1 is NaN).
%   best_nFactors         - The selected number of factors based on BIC criterion.
%   all_best_models       - Cell array {maxNumFactors x 1} containing the best ktensor model found for each number of factors (index 1 is empty).
%   recon_errors          - Vector (maxNumFactors x 1) of relative reconstruction errors for the best model at each number of factors (index 1 is NaN).
%   init_similarity_scores- Vector (maxNumFactors x 1) of average pairwise similarity ('score') between models from different initializations for each factor number (index 1 is NaN). Range [0, 1] or NaN.
%
% REQUIRES: MATLAB Tensor Toolbox

arguments
    data
    tca_type (1,:) char {mustBeMember(tca_type, {'cp_als', 'cp_nmu', 'cp_orth_als'})} = 'cp_nmu'
    normalisation_to_apply (1,:) char {mustBeMember(normalisation_to_apply, {'none', 'z-score', 'min-max'})} = 'none'
    maxNumFactors (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(maxNumFactors, 2)} = 10 % Ensure at least 2 factors are tested
    max_iterations (1,1) double {mustBeInteger, mustBePositive} = 200
    num_initializations (1,1) double {mustBeInteger, mustBePositive} = 20 
end

% --- Preprocessing ---
tca_method = lower(tca_type);
norm_method = lower(normalisation_to_apply);
[I, J, K] = size(data);
data_in = apply_normalisation(data, norm_method, I, J, K);
tensor_data = tensor(data_in); % Convert data to a tensor object
n_data_norm_sq = norm(tensor_data)^2; % Precompute for relative error calculation
n = numel(data_in); % Number of observations for BIC

% --- Initialization ---
bic_values = nan(maxNumFactors, 1);
all_best_models = cell(maxNumFactors, 1); % Stores best model for each nFactor
recon_errors = nan(maxNumFactors, 1); % Stores reconstruction error for best model per nFactor
init_similarity_scores = nan(maxNumFactors, 1); % Stores avg similarity score per nFactor

options = struct('maxiters', max_iterations, 'tol', 1e-6, 'printitn', 0); % TCA options

% --- Main Loop: Iterate over number of factors ---
fprintf('Starting TCA analysis for factors 2 to %d...\n', maxNumFactors);
for nFactors = 2:maxNumFactors
    fprintf('  Testing model with %d factors (%d initializations)...\n', nFactors, num_initializations);

    % Store results from all initializations for this nFactors
    models_this_round = cell(1, num_initializations);
    errors_this_round = nan(1, num_initializations);

    % --- Inner Loop: Multiple Initializations ---
    % Consider using parfor here if Parallel Computing Toolbox is available and data is large
    % parfor init = 1:num_initializations
    for init = 1:num_initializations
        % Fit the model
        P = fit_tca_model(tensor_data, nFactors, tca_method, options);

        % Store the resulting model (ktensor object)
        models_this_round{init} = P;

        % Calculate relative reconstruction error
        rec_error_sq = norm(tensor_data - tensor(P))^2; % Squared error
        errors_this_round(init) = rec_error_sq / n_data_norm_sq; % Relative error
    end % End initialization loop

    % --- Process Results for this nFactors ---

    % 1. Find the best model (minimum reconstruction error) from initializations
    [best_error_this_nFactor, best_init_idx] = min(errors_this_round);
    best_P = models_this_round{best_init_idx}; % Best ktensor for this nFactors

    % 2. Calculate average similarity score across initializations
    similarity_score = calculate_U2_factor_correlation(models_this_round, true);
    fprintf('    Avg. pairwise similarity score across %d inits: %.4f\n', num_initializations, similarity_score);

    % 3. Store results for this nFactors
    all_best_models{nFactors} = best_P;
    recon_errors(nFactors) = best_error_this_nFactor;
    init_similarity_scores(nFactors) = similarity_score;

    % 4. Calculate BIC for the best model found
    RSS = best_error_this_nFactor * n_data_norm_sq; % Residual Sum of Squares = relative_error * data_norm^2
    % Using standard parameter count p = R*(I+J+K). Review if different p needed.
    p = nFactors * (I + J + K);
    % Handle potential log(0) or negative RSS/n if error is extremely small or data is zero
    log_term = log(max(RSS/n, eps)); % Use max with eps to avoid log(0)
    if isnan(log_term) || isinf(log_term)
         warning('BIC calculation issue for nFactors=%d. RSS/n = %g. Setting BIC to NaN.', nFactors, RSS/n);
         bic = NaN;
    else
        bic = n * log_term + p * log(n);
    end
    bic_values(nFactors) = bic;
    fprintf('    Best reconstruction error: %.4e, BIC: %.4f\n', best_error_this_nFactor, bic);

end % End nFactors loop
fprintf('Finished testing all factor numbers.\n');

% --- Model Selection based on BIC ---
valid_bics = bic_values(2:end); % BICs for factors 2 to maxNumFactors
if all(isnan(valid_bics))
    error('BIC calculation failed for all factor numbers. Cannot select model.');
end

[min_bic_val, min_idx_local] = min(valid_bics);
min_idx_global = min_idx_local + 1; % Index in the original 1:maxNumFactors array

% Define BIC tolerance (e.g., 0.1% relative increase allowed) - ADJUST AS NEEDED
tol = 0.001; % 1% tolerance - Make this an input argument?
bic_threshold = min_bic_val + abs(min_bic_val) * tol;

% Find indices (>=2) of models within the tolerance threshold
candidate_indices_local = find(valid_bics <= bic_threshold); % Indices relative to 2:maxNumFactors range

if isempty(candidate_indices_local)
    % Fallback: If no model is within tolerance, choose the minimum BIC model.
    best_nFactors = min_idx_global;
    fprintf('Warning: No model found within BIC tolerance (%.1f%%). Selecting model with minimum BIC.\n', tol*100);
else
    % Select the smallest factor number among the candidates
    best_nFactors_local = min(candidate_indices_local);
    best_nFactors = best_nFactors_local + 1; % Adjust index back to original array (2:maxNumFactors)
end

best_mdl = all_best_models{best_nFactors}; % Retrieve the selected best overall model

fprintf('Selected model with %d factors (BIC = %.2f, threshold = %.2f, min BIC = %.2f at %d factors).\n', ...
    best_nFactors, bic_values(best_nFactors), bic_threshold, min_bic_val, min_idx_global);

% --- Calculate and Reorder Variance Explained for the Final Selected Model ---
if ~isempty(best_mdl)
    component_contributions = zeros(best_nFactors, 1);
    total_ss = n_data_norm_sq; % Use precomputed norm

    % Ensure best_mdl factors are normalized before calculating individual contributions
    % (cp_als usually returns normalized factors, but good to ensure)
    best_mdl = normalize(best_mdl); % Normalize absorbs lambda into factors U

    for r = 1:best_nFactors
        % Create a ktensor for the single component
        lambda_r = 1; % Since normalize sets lambda to 1
        U_r = cell(1,3);
        U_r{1} = best_mdl.U{1}(:, r);
        U_r{2} = best_mdl.U{2}(:, r);
        U_r{3} = best_mdl.U{3}(:, r);
        component_tensor = ktensor(lambda_r, U_r); % Use U_r, not individual columns

        % Calculate contribution - using norm^2 of the single component tensor
        % Note: This is a heuristic for non-orthogonal CP decomposition
        component_contributions(r) = norm(component_tensor)^2;
    end

    variance_explained = component_contributions / total_ss;

    % Reorder components in descending order of variance explained
    [variance_explained, sorted_indices] = sort(variance_explained, 'descend');
    best_mdl = arrange(best_mdl, sorted_indices); % Use arrange function for ktensors
else
    warning('Final selected model (best_mdl) is empty. Cannot compute variance explained.');
    variance_explained = [];
end

end % End main function

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         Helper Functions
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function data_out = apply_normalisation(data, norm_method, I, J, K)
    % Apply the chosen normalization method to the data.
    switch norm_method
        case 'none'
            data_out = data;
        case 'z-score'
            data_reshaped = reshape(data, I, []);
            data_z = zscore(data_reshaped, 0, 2); % Z-score across bins*trials for each neuron
            data_out = reshape(data_z, I, J, K);
        case 'min-max'
            data_reshaped = reshape(data, I, []);
            data_min = min(data_reshaped, [], 2);
            data_max = max(data_reshaped, [], 2);
            data_range = data_max - data_min;
            data_range(data_range == 0) = 1; % Avoid division by zero if activity is constant
            data_scaled = (data_reshaped - data_min) ./ data_range;
            % data_scaled(isnan(data_scaled)) = 0; % Should be handled by data_range check now
            data_out = reshape(data_scaled, I, J, K);
        otherwise
            error('Unknown normalization method: %s', norm_method);
    end
end

% -------------------------------------------------------------------------

function P = fit_tca_model(tensor_data, nFactors, tca_method, options)
    % Fit a TCA model using the specified method and options.
    try
        switch tca_method
            case 'cp_als'
                P = cp_als(tensor_data, nFactors, options);
            case 'cp_nmu' % Requires Non-negative data
                 if any(double(tensor_data) < 0, 'all')
                     warning('Data contains negative values, but cp_nmu requested. Results may be invalid.');
                 end
                 P = cp_nmu(tensor_data, nFactors, options);
            case 'cp_orth_als' % Requires Orthogonal factors (often specific use cases)
                P = cp_orth_als(tensor_data, nFactors, options);
            otherwise
                error('Unknown TCA method: %s', tca_method);
        end
    catch ME
         fprintf('ERROR fitting TCA model (method: %s, nFactors: %d): %s\n', tca_method, nFactors, ME.message);
         % Return an empty ktensor or rethrow error? Returning empty might allow loop to continue.
         P = ktensor(); % Return an empty ktensor on failure
         % Alternatively: rethrow(ME);
    end
end

% -------------------------------------------------------------------------

function avg_similarity = calculate_model_similarity_score(models_cell_array)
    % Calculates the average pairwise similarity score between ktensor models
    % using the Tensor Toolbox 'score' function.
    % models_cell_array: Cell array containing ktensor objects.
    % Returns: Average similarity [0, 1], or NaN if < 2 models or errors occur.

    num_inits = numel(models_cell_array);

    % Handle cases with fewer than 2 initializations or empty models
    valid_models_idx = find(~cellfun(@isempty, models_cell_array) & cellfun(@(x) isa(x, 'ktensor'), models_cell_array));
    valid_models = models_cell_array(valid_models_idx);
    num_valid_models = numel(valid_models);

    if num_valid_models < 2
        avg_similarity = NaN; % Cannot compute similarity
        return;
    end

    total_score = 0;
    num_pairs = 0;

    for i = 1:num_valid_models
        for j = i+1:num_valid_models
            try
                % score function computes similarity, handling permutation/scaling.
                pair_score = score(valid_models{i}, valid_models{j});
                 % Score might return NaN if models are incompatible (e.g., different sizes, although shouldn't happen here)
                 if isnan(pair_score)
                     warning('score() returned NaN for pair (%d, %d). Skipping.', i,j);
                     continue; % Skip this pair if score is NaN
                 end
            catch ME
                warning('Could not compute score between models %d and %d: %s. Skipping pair.', valid_models_idx(i), valid_models_idx(j), ME.message);
                continue; % Skip this pair if score calculation fails
            end

            total_score = total_score + pair_score;
            num_pairs = num_pairs + 1;
        end
    end

    if num_pairs > 0
        avg_similarity = total_score / num_pairs;
    else
        % This case might happen if all pairwise comparisons failed or if num_valid_models was < 2 initially
        avg_similarity = NaN;
    end
end

% -------------------------------------------------------------------------
% % Additional helper function potentially needed if using matchpairs
function avg_similarity = calculate_factor_similarity_manual(factor_matrices_cell)
    % Example using matchpairs for spatial factors (U{1}) - REQUIRES Statistics Toolbox
    num_inits = numel(factor_matrices_cell);
    if num_inits < 2
        avg_similarity = NaN; return;
    end
    total_similarity = 0; num_pairs = 0;
    for i = 1:num_inits
        for j = i+1:num_inits
            U1_i = normalize(factor_matrices_cell{i}.U{2}, 'norm', 2); % Normalize columns
            U1_j = normalize(factor_matrices_cell{j}.U{2}, 'norm', 2);
            if size(U1_i, 2) ~= size(U1_j, 2); continue; end % Skip if R differs
            R = size(U1_i, 2);
            cost_matrix = 1 - abs(U1_i' * U1_j); % Cost = 1 - abs(cosine_similarity)
            [matches, ~] = matchpairs(cost_matrix, 1); % Find best matches minimizing cost
            matched_similarity = 0;
            for k = 1:size(matches, 1)
                matched_similarity = matched_similarity + abs(U1_i(:, matches(k,1))' * U1_j(:, matches(k,2)));
            end
            pair_avg_similarity = matched_similarity / R;
            total_similarity = total_similarity + pair_avg_similarity;
            num_pairs = num_pairs + 1;
        end
    end
    avg_similarity = total_similarity / max(1, num_pairs);
end


function avg_correlation = calculate_U2_factor_correlation(models_cell_array, use_matchpairs)
    % Calculates the average pairwise absolute CORRELATION between the second
    % factor matrices (U{2}) of ktensor models in a cell array, after
    % matching factors based on maximal correlation.
    % Assumes U{2} corresponds to the factors of interest.
    %
    % INPUTS:
    %   models_cell_array: Cell array containing ktensor objects from initializations.
    %   use_matchpairs: Logical. If true (default), uses matchpairs function
    %                   (requires Statistics and Machine Learning Toolbox).
    %                   If false or toolbox unavailable, uses a greedy matching approach.
    %
    % OUTPUTS:
    %   avg_correlation: Average absolute correlation score [0, 1] across pairs, or NaN.

    arguments
        models_cell_array cell
        use_matchpairs logical = true % Default to using matchpairs if available
    end

    % Filter out empty or non-ktensor entries
    valid_models_idx = find(~cellfun(@isempty, models_cell_array) & cellfun(@(x) isa(x, 'ktensor'), models_cell_array));
    valid_models = models_cell_array(valid_models_idx);
    num_valid_models = numel(valid_models);

    if num_valid_models < 2
        avg_correlation = NaN; % Cannot compute similarity
        return;
    end

    % fprintf('  Using %s for factor matching.\n', MfileName); % Optional debug info

    total_correlation = 0;
    num_pairs = 0;

    for i = 1:num_valid_models
        for j = i+1:num_valid_models
            ktensor_i = valid_models{i};
            ktensor_j = valid_models{j};

            U2_i = ktensor_i.U{2}; % Extract second factor matrix (e.g., Time x Rank)
            U2_j = ktensor_j.U{2};
            R = size(U2_i, 2); % Number of factors (rank)

            if R == 0
                pair_avg_correlation = NaN; % Handle rank 0 case
            else
                % --- Calculate pairwise absolute correlations between columns ---
                try
                    % corr(X,Y) computes matrix of correlation coefficients between columns of X and Y
                    correlation_matrix = abs(corr(U2_i, U2_j)); % Results in an R x R matrix
                    % Handle potential NaNs (e.g., if a column has zero variance)
                    correlation_matrix(isnan(correlation_matrix)) = 0; % Treat NaN correlation as 0 for matching purpose
                catch ME_corr
                    warning('MATLAB:calculate_U2_factor_correlation:corrFailed', ...
                            'corr function failed for pair (%d, %d): %s. Skipping pair.', valid_models_idx(i), valid_models_idx(j), ME_corr.message);
                    continue; % Skip this pair if correlation calculation fails
                end

                % --- Match factors to maximize correlation ---
                matched_correlation_sum = 0;
                all_matched = false; % Flag to check if matching succeeded for all R factors

                if use_matchpairs
                    cost_matrix = 1 - correlation_matrix; % Cost = 1 - absolute correlation
                    try
                        % Find pairs [row_idx, col_idx] that minimize total cost
                        % Set costOfNonAssignment > 1 (max possible cost) to ensure matching R pairs if possible
                        [matches, ~] = matchpairs(cost_matrix, 1.01);
                         if size(matches,1) == R % Check if all R factors were matched
                             all_matched = true;
                             for k = 1:size(matches, 1)
                                % Use the original correlation_matrix for the score
                                matched_correlation_sum = matched_correlation_sum + correlation_matrix(matches(k,1), matches(k,2));
                             end
                         else
                            warning('MATLAB:calculate_U2_factor_correlation:matchpairsIncomplete', ...
                                    'matchpairs did not return R matches for pair (%d, %d). R=%d, Matches=%d', valid_models_idx(i), valid_models_idx(j), R, size(matches,1));
                         end
                    catch ME_matchpairs
                         warning('MATLAB:calculate_U2_factor_correlation:matchpairsError', ...
                                 'matchpairs failed for pair (%d, %d): %s. Using NaN for this pair.', valid_models_idx(i), valid_models_idx(j), ME_matchpairs.message);
                         % matched_correlation_sum remains 0, all_matched remains false
                    end
                else % ----- Greedy matching -----
                    temp_corr_matrix = correlation_matrix;
                    current_match_sum = 0;
                    matched_j = false(1, R);
                    num_greedy_matches = 0;
                    for factor_i = 1:R
                         best_corr_for_i = -Inf; % Correlations are between [-1, 1], absolute is [0, 1]
                         best_j_idx = -1;
                         % Find best *available* match in j for current factor i
                         available_j_indices = find(~matched_j);
                         if isempty(available_j_indices), break; end % Stop if no more j's available

                         [max_corr_val, local_idx] = max(temp_corr_matrix(factor_i, available_j_indices));

                         % Find corresponding global index in j
                         best_j_global_idx = available_j_indices(local_idx);

                         current_match_sum = current_match_sum + max_corr_val; % Add the absolute correlation
                         matched_j(best_j_global_idx) = true; % Mark this j column as used
                         num_greedy_matches = num_greedy_matches + 1;
                    end
                    if num_greedy_matches == R % Check if all factors were matched
                        all_matched = true;
                        matched_correlation_sum = current_match_sum;
                    else
                        warning('MATLAB:calculate_U2_factor_correlation:greedyIncomplete', ...
                                'Greedy matching did not match all factors for pair (%d, %d). R=%d, Matches=%d', valid_models_idx(i), valid_models_idx(j), R, num_greedy_matches);
                    end
                end % ----- End matching logic -----

                % --- Calculate average correlation for the pair ---
                if all_matched
                    pair_avg_correlation = matched_correlation_sum / R;
                else
                    pair_avg_correlation = NaN; % Use NaN if matching failed or was incomplete
                end
            end % End if R > 0

            % --- Accumulate results ---
            if ~isnan(pair_avg_correlation)
                total_correlation = total_correlation + pair_avg_correlation;
                num_pairs = num_pairs + 1;
            end

        end % End loop j
    end % End loop i

    % --- Calculate final average ---
    if num_pairs > 0
        avg_correlation = total_correlation / num_pairs;
    else
        avg_correlation = NaN; % If no valid pairs were compared
        % fprintf('  Similarity calculation resulted in NaN: No valid pairs compared for this factor count.\n');
    end
end % End helper function
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%