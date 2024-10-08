function v = estimate_trialwise_variance(data, change_point_mean)

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
    genVar(trial) = log(det(covMatrix + eps*eye(size(covMatrix))));
end

figure
shadedErrorBar(1:n_trials, movmean(genVar, 3), movstd(genVar, 3)/sqrt(3))
xline(change_point_mean)

figure
bar([mean(genVar(1:3)), mean(genVar(4:125)), mean(genVar(126:end))])