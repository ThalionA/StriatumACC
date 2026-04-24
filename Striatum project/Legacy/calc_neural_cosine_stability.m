function [neural_stability] = calc_neural_cosine_stability(spike_tensor, window_size)
    % spike_tensor: [n_neurons x n_spatial_bins x n_trials]
    % window_size: integer (e.g., 5)
    
    [n_neurons, n_bins, n_trials] = size(spike_tensor);
    neural_stability = nan(n_trials, 1);
    half_win = floor(window_size/2);

    % Flatten: [features x trials]
    flat_data = reshape(permute(spike_tensor, [2 1 3]), [n_bins*n_neurons, n_trials]);
    
    % GUARD 1: Handle input NaNs (if any slipped through filtering)
    % Option A: Treat NaNs as zeros (no activity)
    flat_data(isnan(flat_data)) = 0; 
    
    for t = 1:n_trials
        t_start = max(1, t - half_win);
        t_end = min(n_trials, t + half_win);
        idx = t_start:t_end;
        
        if length(idx) < 2, continue; end
        
        current_vectors = flat_data(:, idx);
        
        % Calculate Norms
        vec_norms = vecnorm(current_vectors);
        
        % GUARD 2: Handle Zero-Norm Vectors (Silent Trials)
        % If norm is 0, we cannot divide. We create a 'valid' mask.
        % Vectors with 0 norm will result in 0 similarity (orthogonal to everything)
        valid_norm_mask = vec_norms > eps; % eps is floating-point relative accuracy
        
        % Initialize unit vectors with zeros
        unit_vectors = zeros(size(current_vectors));
        
        % Only normalize vectors that have non-zero length
        unit_vectors(:, valid_norm_mask) = current_vectors(:, valid_norm_mask) ./ vec_norms(valid_norm_mask);
        
        % Matrix multiplication for Cosine Similarity
        sim_matrix = unit_vectors' * unit_vectors;
        
        % GUARD 3: Handling the diagonal and result NaNs
        % If a vector was 0-norm, its correlations will be 0. 
        % However, if *both* vectors are 0-norm, dot product is 0. 
        % This is mathematically safe now, but we check final output just in case.
        
        % Extract unique off-diagonal pairs
        mask = triu(true(size(sim_matrix)), 1);
        unique_sims = sim_matrix(mask);
        
        % Final safety calculation
        neural_stability(t) = mean(unique_sims, 'omitnan');
    end
end