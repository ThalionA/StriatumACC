function v = estimate_trialwise_variance(data)

[n_neurons, n_bins, n_trials] = size(data);

% Preallocate an array to store the generalized variance for each trial
genVar = zeros(1, n_trials);

for trial = 1:n_trials
    % Extract data for the current trial (size: n_neurons x n_spatialbins)
    trialData = data(:, :, trial);
    
    % Transpose to get variables (neurons) in columns
    trialDataT = trialData';
    
    % Compute the covariance matrix
    covMatrix = cov(trialDataT);
    
    % Compute the determinant of the covariance matrix
    genVar(trial) = log(det(covMatrix + 1*eye(size(covMatrix))));
end

genVar = genVar/n_neurons;

window_size = 5;


% shadedErrorBar(1:n_trials, movmean(genVar, window_size, 'omitmissing'), movstd(genVar, window_size, 'omitmissing')/sqrt(window_size))


v = genVar;