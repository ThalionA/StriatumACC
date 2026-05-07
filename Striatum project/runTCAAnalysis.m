% File: runTCAAnalysis.m
function [best_mdl, best_n_factors, results] = runTCAAnalysis(tensor_data, tca_cfg)
% Runs Tensor Component Analysis (TCA) using the tca_with_bic_extended function.
% This function acts as a wrapper to call tca_with_bic_extended, passing
% configuration parameters and returning the selected model, factor count,
% and detailed results.
%
% INPUTS:
%   tensor_data - The input data tensor (Neurons x Bins x Trials) after NaN handling.
%   tca_cfg     - Configuration structure containing TCA parameters:
%                 .method              : TCA method ('cp_nmu', 'cp_als', etc.)
%                 .normalization       : Normalization ('min-max', 'z-score', 'none')
%                 .max_factors         : Maximum number of factors to test
%                 .max_iterations      : Max iterations for TCA fitting
%                 .num_initialisations : Number of random initializations per factor count
%
% OUTPUTS:
%   best_mdl        - The final ktensor model selected by tca_with_bic_extended.
%   best_n_factors  - The number of factors in the selected best_mdl.
%   results         - Structure containing detailed outputs from tca_with_bic_extended:
%                     .variance_explained    : Variance explained for best_mdl components.
%                     .bic_values            : BIC values for factors 1:maxNumFactors (NaN at 1).
%                     .all_best_models       : Cell array of best models for each factor count.
%                     .recon_errors          : Reconstruction errors for each factor count.
%                     .init_similarity_scores: Initialization similarity scores.

    best_mdl = [];
    best_n_factors = [];
    results = struct(); % Initialize results struct

    if isempty(tensor_data)
        warning('runTCAAnalysis: Input tensor data is empty.');
        return;
    end

    % --- Extract Parameters from cfg ---
    % Validate required fields exist in tca_cfg first (optional but good practice)
    required_fields = {'method', 'normalization', 'max_factors', 'max_iterations', 'num_initialisations'};
    if ~all(isfield(tca_cfg, required_fields))
        error('runTCAAnalysis: Missing one or more required fields in tca_cfg structure.');
    end

    tca_type = tca_cfg.method;
    normalisation_to_apply = tca_cfg.normalization;
    maxNumFactors = tca_cfg.max_factors;
    max_iterations = tca_cfg.max_iterations;
    num_initialisations = tca_cfg.num_initialisations;
    % Note: Selection method (e.g., 'bic', 'fixed') is now handled *within* tca_with_bic_extended

    fprintf('--- Calling tca_with_bic_extended ---\n');
    fprintf('  Method: %s, Normalization: %s\n', tca_type, normalisation_to_apply);
    fprintf('  Max Factors: %d, Iterations: %d, Initializations: %d\n', maxNumFactors, max_iterations, num_initialisations);

    % --- Check if the target function exists ---
    fit_function_name = 'tca_with_bic_extended';
    if isempty(which(fit_function_name))
        warning('TCA function "%s" not found in MATLAB path. Cannot run TCA.', fit_function_name);
        return; % Exit if the function isn't available
    end

    % --- Execute the TCA function ---
    try
        % Call the specialized function, capturing all its outputs
        [selected_mdl, variance_explained, bic_values, selected_nFactors, ...
         all_models, recon_errors, init_similarity_scores] = ...
            tca_with_bic_extended(tensor_data, tca_type, normalisation_to_apply, ...
                                  maxNumFactors, max_iterations, num_initialisations);

        % --- Assign Outputs ---
        % The function already performs the model selection based on its internal criteria
        best_mdl = selected_mdl;
        best_n_factors = selected_nFactors;

        % --- Store Detailed Results ---
        % These are useful for plotting and diagnostics (e.g., BIC curve)
        results.variance_explained = variance_explained; % For the final selected model
        results.bic_values = bic_values;             % Array [maxNumFactors x 1], NaN at index 1
        results.all_best_models = all_models;        % Cell array {maxNumFactors x 1}, empty at index 1
        results.recon_errors = recon_errors;         % Array [maxNumFactors x 1], NaN at index 1
        results.init_similarity_scores = init_similarity_scores; % Array [maxNumFactors x 1], NaN at index 1

        % --- Report Outcome ---
        if isempty(best_mdl) || isempty(best_n_factors)
             % This might happen if tca_with_bic_extended encounters an error preventing selection
             fprintf('  %s completed but did not return a valid selected model or factor count.\n', fit_function_name);
        else
             fprintf('  %s selected %d factors based on its criteria.\n', fit_function_name, best_n_factors);
        end

    catch ME
        warning('Error during execution of %s: %s', fit_function_name, ME.getReport);
        % Ensure outputs remain empty/default on error
        best_mdl = [];
        best_n_factors = [];
        results = struct(); % Clear results struct as well
    end

    fprintf('--- TCA Analysis Function Finished ---\n');

end % End function runTCAAnalysis