%% Load

load("preprocessed_data.mat")
task_data = preprocessed_data;
load("preprocessed_data_control.mat")
control_data = preprocessed_data;
load("preprocessed_data_control2.mat")
control2_data = preprocessed_data;

%% Find "learning" point

n_animals_task = numel(task_data);

zscored_lick_errors = {task_data(:).zscored_lick_errors};
learning_point = cellfun(@(x) find(movsum(x <= -2, [0,9]) >= 8, 1), zscored_lick_errors, 'UniformOutput', false);
has_learning_point = ~cellfun(@isempty, learning_point);

% Find "stability" point
avg_lick_corrs = cell(1, n_animals_task);
avg_occupancy_corrs = cell(1, n_animals_task);

for ianimal = 1:n_animals_task
    n_trials = task_data(ianimal).n_trials;
    
    lick_data = task_data(ianimal).spatial_binned_data.licks(1:n_trials, :);
    occupancy_data = task_data(ianimal).spatial_binned_data.durations(1:n_trials, :);

    window_size = 10;
    half_window = floor(window_size / 2);
    trials = n_trials;

    avg_lick_corrs_animal = nan(1, trials);
    avg_occupancy_corrs_animal = nan(1, trials);

    for t = 1:trials
        % Define the window of trials centered on trial t
        trial_start = max(1, t - half_window);
        trial_end = min(trials, t + half_window);
        trials_in_window = trial_start:trial_end;
        num_trials_in_window = length(trials_in_window);

        % Proceed only if we have at least two trials to correlate
        if num_trials_in_window > 1
            % Extract the data for the current neuron and window
            lick_data_block = squeeze(lick_data(trials_in_window, :)');
            occ_data_block = squeeze(occupancy_data(trials_in_window, :)');

            % Compute the correlation matrices
            R_lick = corrcoef(lick_data_block);
            R_occ = corrcoef(occ_data_block);

            % Extract the upper triangle values
            upper_triangle_lick = triu(R_lick, 1);
            upper_vals_lick = upper_triangle_lick(upper_triangle_lick ~= 0);

            upper_triangle_occ = triu(R_occ, 1);
            upper_vals_occ = upper_triangle_occ(upper_triangle_occ ~= 0);

            % Compute the average correlation
            avg_lick_corrs_animal(t) = mean(upper_vals_lick, 'omitmissing');
            avg_occupancy_corrs_animal(t) = mean(upper_vals_occ, 'omitmissing');
        end
    end

    avg_lick_corrs{ianimal} = avg_lick_corrs_animal;
    avg_occupancy_corrs{ianimal} = avg_occupancy_corrs_animal;
end

stability_point_lick = cellfun(@(x) find(x > 0.75), avg_lick_corrs, 'UniformOutput', false);

%% Filter out bad animals
task_data = task_data(has_learning_point);
%% Define colors for each area
color_dms = [0, 0.4470, 0.7410];       % Deep Blue for DMS
color_dls =  [0.4660, 0.6740, 0.1880];  % Forest Green for DLS
color_acc = [0.8500, 0.3250, 0.0980];  % Crimson Red for ACC

%% Analysis
n_animals_task = size(task_data, 2);
n_animals_control = size(control_data, 2);
n_animals_control2 = size(control2_data, 2);

neuron_counts_task = cellfun(@(x) size(x,1), {task_data(:).spatial_binned_fr_all});
target_neurons = min(neuron_counts_task);

for ianimal = 1:n_animals_task

    is_dms = task_data(ianimal).is_dms;
    is_dls = task_data(ianimal).is_dls;
    is_acc = task_data(ianimal).is_acc;

    current_activity = task_data(ianimal).spatial_binned_fr_all;
    [neurons, bins, trials] = size(current_activity);

    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end

    current_activity = current_activity(:, :, 1:change_point);

    current_activity_dms = current_activity(is_dms, :, :);
    current_activity_dls = current_activity(is_dls, :, :);
    current_activity_acc = current_activity(is_acc, :, :);

    all_activity_dms_task{ianimal} = current_activity_dms;
    all_activity_dls_task{ianimal} = current_activity_dls;
    all_activity_acc_task{ianimal} = current_activity_acc;


    zscored_lick_errors = task_data(ianimal).zscored_lick_errors(1:trials);

    for ibin = 1:bins
        p = polyfit(mean(current_activity(:, ibin, :), 3, 'omitmissing'), var(current_activity(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_task{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_dms(:, ibin, :), 3, 'omitmissing'), var(current_activity_dms(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_task_dms{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_dls(:, ibin, :), 3, 'omitmissing'), var(current_activity_dls(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_task_dls{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_acc(:, ibin, :), 3, 'omitmissing'), var(current_activity_acc(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_task_acc{ianimal}(ibin) = p(1);
    end

    genvar_task{ianimal} = estimate_trialwise_variance(current_activity(:, :, 1:change_point));
    genvar_task_dms{ianimal} = estimate_trialwise_variance(current_activity_dms(:, :, 1:change_point));
    genvar_task_dls{ianimal} = estimate_trialwise_variance(current_activity_dls(:, :, 1:change_point));
    genvar_task_acc{ianimal} = estimate_trialwise_variance(current_activity_acc(:, :, 1:change_point));

    mask_none = false(1, neurons);
    doShuffle = false;  % set to true for shuffle control
    ablation_mask = mask_none;

    %----- DECODING ANALYSIS WITH SUBSAMPLING (TASK) -----
    % If this animal has more neurons than the target, perform 10 iterations
    % of decoding using a random subsample of neurons of size target_neurons.
    if neurons > target_neurons
        predicted_bins_all = NaN(change_point, bins, 10);
        for iter = 1:10
            sample_idx = randsample(neurons, target_neurons);
            % Use the subsampled neurons from current_activity
            current_activity_sub = current_activity(sample_idx, :, :);
            predicted_bins_sub = NaN(change_point, bins);
            for testTrial = 1:change_point
                fprintf('Decoding trial %d/%d (Animal %d, iter %d)\n', testTrial, change_point, ianimal, iter)
                trainTrials = setdiff(1:change_point, testTrial);
                trainData = current_activity_sub(:, :, trainTrials);
                meanFR = mean(trainData, 3, 'omitnan');
                if doShuffle
                    for ineuron = 1:size(meanFR,1)
                        meanFR(ineuron, :) = meanFR(ineuron, randperm(bins));
                    end
                end
                testData = current_activity_sub(:, :, testTrial);
                for iBin = 1:bins
                    observedCounts = testData(:, iBin);
                    logLikelihood = zeros(bins, 1);
                    for c = 1:bins
                        lambda_c = meanFR(:, c);
                        lambda_c(lambda_c<=0) = 1e-6;
                        ll = observedCounts .* log(lambda_c) - lambda_c;
                        logLikelihood(c) = sum(ll, 'omitnan');
                    end
                    % Instead of max, break ties randomly:
                    maxVal = max(logLikelihood);
                    maxInds = find(logLikelihood == maxVal);
                    if numel(maxInds) > 1
                        bestBin = maxInds(randi(numel(maxInds)));
                    else
                        bestBin = maxInds;
                    end
                    predicted_bins_sub(testTrial, iBin) = bestBin;
                end
            end
            predicted_bins_all(:,:,iter) = predicted_bins_sub;
        end
        % Take the mode across the 10 iterations as the final prediction.
        predicted_bins = mode(predicted_bins_all, 3);
    else
        % If no subsampling is needed, perform decoding as usual.
        predicted_bins = NaN(change_point, bins);
        for testTrial = 1:change_point
            fprintf('Decoding trial %d/%d (Animal %d)\n', testTrial, change_point, ianimal)
            trainTrials = setdiff(1:change_point, testTrial);
            useNeurons = ~ablation_mask;
            trainData = current_activity(useNeurons, :, trainTrials);
            meanFR = mean(trainData, 3, 'omitnan');
            if doShuffle
                for ineuron = 1:size(meanFR,1)
                    meanFR(ineuron, :) = meanFR(ineuron, randperm(bins));
                end
            end
            testData = current_activity(useNeurons, :, testTrial);
            for iBin = 1:bins
                observedCounts = testData(:, iBin);
                logLikelihood = zeros(bins, 1);
                for c = 1:bins
                    lambda_c = meanFR(:, c);
                    lambda_c(lambda_c<=0) = 1e-6;
                    ll = observedCounts .* log(lambda_c) - lambda_c;
                    logLikelihood(c) = sum(ll, 'omitnan');
                end
                % Instead of max, break ties randomly:
                maxVal = max(logLikelihood);
                maxInds = find(logLikelihood == maxVal);
                if numel(maxInds) > 1
                    bestBin = maxInds(randi(numel(maxInds)));
                else
                    bestBin = maxInds;
                end
                predicted_bins(testTrial, iBin) = bestBin;
            end
        end
    end

    actual_bins = repmat(1:bins, [change_point, 1]);  % ground truth: bin i is labelled i
    errors_task{ianimal} = predicted_bins - actual_bins;
    predicted_bins_task{ianimal} = predicted_bins;
    actual_bins_task{ianimal} = actual_bins;


    window_size = 5;
    half_window = floor(window_size / 2);

    % Preallocate the output matrix with NaNs to handle cases with insufficient data
    avg_corrs = NaN(neurons, change_point);

    % Loop over each neuron
    for n = 1:neurons
        % Loop over each trial
        for t = 1:change_point
            % Define the window of trials centered on trial t
            trial_start = max(1, t - half_window);
            trial_end = min(change_point, t + half_window);
            trials_in_window = trial_start:trial_end;
            num_trials_in_window = length(trials_in_window);

            % Proceed only if we have at least two trials to correlate
            if num_trials_in_window > 1
                % Extract the data for the current neuron and window
                data_block = squeeze(current_activity(n, :, trials_in_window)); % Size: [spatial_bins x num_trials_in_window]

                % Compute the correlation matrix for the trials in the current window
                R = corr(data_block);

                % Extract the upper triangle of the correlation matrix (excluding the diagonal)
                upper_triangle = triu(R, 1);
                upper_vals = upper_triangle(upper_triangle ~= 0);

                % Compute the average correlation
                avg_corrs(n, t) = mean(upper_vals);

            else
                % If only one trial is available, set the average correlation to NaN
                avg_corrs(n, t) = NaN;
            end
        end
    end

    avg_corrs_task{ianimal} = avg_corrs;

    % Loop over each area (DMS, DLS, ACC)
    areas = {'dms', 'dls', 'acc'};
    activities = {current_activity_dms, current_activity_dls, current_activity_acc};

    for a = 1:length(areas)
        area_activity = activities{a};
        [neurons_area, ~, ~] = size(area_activity);

        % Preallocate NaN matrix for current area
        avg_corrs = NaN(neurons_area, change_point);

        % Loop over each neuron in the current area
        for n = 1:neurons_area
            % Loop over each trial
            for t = 1:change_point
                % Define the window of trials centered on trial t
                trial_start = max(1, t - half_window);
                trial_end = min(change_point, t + half_window);
                trials_in_window = trial_start:trial_end;
                num_trials_in_window = length(trials_in_window);

                % Proceed only if we have at least two trials to correlate
                if num_trials_in_window > 1
                    % Extract the data for the current neuron and window
                    data_block = squeeze(area_activity(n, :, trials_in_window)); % Size: [spatial_bins x num_trials_in_window]

                    % Compute the correlation matrix for the trials in the current window
                    R = corr(data_block);

                    % Extract the upper triangle of the correlation matrix (excluding the diagonal)
                    upper_triangle = triu(R, 1);
                    upper_vals = upper_triangle(upper_triangle ~= 0);

                    % Compute the average correlation
                    avg_corrs(n, t) = mean(upper_vals);
                else
                    % If only one trial is available, set the average correlation to NaN
                    avg_corrs(n, t) = NaN;
                end
            end
        end

        % Store results for the current area and animal
        avg_corrs_task_area{a}{ianimal} = avg_corrs;
    end

end
% Assign the final matrices to variables
avg_corrs_dms_task = avg_corrs_task_area{1};
avg_corrs_dls_task = avg_corrs_task_area{2};
avg_corrs_acc_task = avg_corrs_task_area{3};

all_activity_task = {task_data(:).spatial_binned_fr_all};

mean_activity_task_spatial_neurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_task, 'UniformOutput', false);
mean_activity_task_spatial_neurons = cat(1, mean_activity_task_spatial_neurons{:});

mean_activity_task = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_task, 'UniformOutput', false);
mean_activity_task_spatial = cellfun(@(x) squeeze(mean(x, 2, 'omitmissing'))', mean_activity_task, 'UniformOutput', false);
mean_activity_task_spatial = cat(1, mean_activity_task_spatial{:});

mean_activity_task_spatial_dmsneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_dms_task, 'UniformOutput', false);
mean_activity_task_spatial_dmsneurons = cat(1, mean_activity_task_spatial_dmsneurons{:});

mean_activity_task_dms = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_dms_task, 'UniformOutput', false);
mean_activity_task_dms_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_task_dms, 'UniformOutput', false);
mean_activity_task_dms_spatial = cat(1, mean_activity_task_dms_spatial{:});

mean_activity_task_spatial_dlsneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_dls_task, 'UniformOutput', false);
mean_activity_task_spatial_dlsneurons = cat(1, mean_activity_task_spatial_dlsneurons{:});

mean_activity_task_dls = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_dls_task, 'UniformOutput', false);
mean_activity_task_dls_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_task_dls, 'UniformOutput', false);
mean_activity_task_dls_spatial = cat(1, mean_activity_task_dls_spatial{:});

mean_activity_task_spatial_accneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_acc_task, 'UniformOutput', false);
mean_activity_task_spatial_accneurons = cat(1, mean_activity_task_spatial_accneurons{:});

mean_activity_task_acc = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_acc_task, 'UniformOutput', false);
mean_activity_task_acc_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_task_acc, 'UniformOutput', false);
mean_activity_task_acc_spatial = cat(1, mean_activity_task_acc_spatial{:});


for ianimal = 1:n_animals_control

    is_dms = control_data(ianimal).is_dms;
    is_dls = control_data(ianimal).is_dls;
    is_acc = control_data(ianimal).is_acc;

    current_activity = control_data(ianimal).spatial_binned_fr_all;
    current_activity_dms = current_activity(is_dms, :, :);
    current_activity_dls = current_activity(is_dls, :, :);
    current_activity_acc = current_activity(is_acc, :, :);

    all_activity_dms_control{ianimal} = current_activity_dms;
    all_activity_dls_control{ianimal} = current_activity_dls;
    all_activity_acc_control{ianimal} = current_activity_acc;

    [neurons, bins, trials] = size(current_activity);

    try
        change_point = min([control_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end

    for ibin = 1:bins
        p = polyfit(mean(current_activity(:, ibin, :), 3, 'omitmissing'), var(current_activity(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_control{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_dms(:, ibin, :), 3, 'omitmissing'), var(current_activity_dms(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_control_dms{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_dls(:, ibin, :), 3, 'omitmissing'), var(current_activity_dls(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_control_dls{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_acc(:, ibin, :), 3, 'omitmissing'), var(current_activity_acc(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_control_acc{ianimal}(ibin) = p(1);
    end

    genvar_control{ianimal} = estimate_trialwise_variance(current_activity(:, :, 1:change_point));
    genvar_control_dms{ianimal} = estimate_trialwise_variance(current_activity_dms(:, :, 1:change_point));
    genvar_control_dls{ianimal} = estimate_trialwise_variance(current_activity_dls(:, :, 1:change_point));
    genvar_control_acc{ianimal} = estimate_trialwise_variance(current_activity_acc(:, :, 1:change_point));

    mask_none = false(1, neurons);

    doShuffle = false;  % set to true for shuffle control
    ablation_mask = mask_none;  % set to mask_dms, mask_dls, mask_acc, etc.

    predicted_bins = NaN(change_point, bins);  % store final predictions
    actual_bins = repmat(1:bins, [change_point, 1]);  % ground truth: bin i is labelled i

    all_log_likelihoods = nan(change_point, bins, bins);

    for testTrial = 1:change_point
        fprintf('decoding trial %d/%d\n', testTrial, change_point)
        trainTrials = setdiff(1:change_point, testTrial);
        % ablation_mask is 1 for neurons we want to exclude
        useNeurons = ~ablation_mask;  % useNeurons is 1 for kept neurons

        % Extract the training data for these neurons
        trainData = current_activity(useNeurons, :, trainTrials);
        % trainData size is [nUsedNeurons x nBins x (nTrials-1)]

        meanFR = mean(trainData, 3, 'omitnan');

        if doShuffle
            for ineuron = 1:size(meanFR,1)
                % random permutation of bin axis for that neuron
                meanFR(ineuron, :) = meanFR(ineuron, randperm(bins));
            end
        end

        %----------------------------------------------------------------------
        % 5) Decode the left-out trial: for each bin in testTrial, compute
        %    likelihood for each candidate bin, choose the best.
        %----------------------------------------------------------------------
        testData = current_activity(useNeurons, :, testTrial);  % [nUsedNeurons x nBins]

        for iBin = 1:bins
            % Observed firing in that bin: [nUsedNeurons x 1]
            observedCounts = testData(:, iBin);

            % We'll compute log-likelihood under each candidate bin
            % for each neuron, then sum across neurons.
            % Poisson: p(r|lambda) = exp(-lambda) * lambda^r / r!
            % We'll do log(p(r|lambda)) to avoid underflow:
            % log p(r|lambda) = -lambda + r*log(lambda) - log(r!)
            % We'll skip the log(r!) term as it doesn't affect argmax.

            % meanFR is [nUsedNeurons x nBins], so for candidate bin c
            % the mean rate is meanFR(:, c).

            logLikelihood = zeros(bins, 1);  % store log-likelihood for each candidate bin
            for c = 1:bins
                lambda_c = meanFR(:, c);  % [nUsedNeurons x 1]
                % Avoid zero or negative
                lambda_c(lambda_c<=0) = 1e-6;

                % observedCounts is r, also [nUsedNeurons x 1]
                % Summation over neurons of r*log(lambda) - lambda
                ll = observedCounts .* log(lambda_c) - lambda_c;
                logLikelihood(c) = sum(ll, 'omitnan');
            end


            % Instead of max, break ties randomly:
            maxVal = max(logLikelihood);
            maxInds = find(logLikelihood == maxVal);
            if numel(maxInds) > 1
                bestBin = maxInds(randi(numel(maxInds)));
            else
                bestBin = maxInds;
            end
            predicted_bins(testTrial, iBin) = bestBin;

            all_log_likelihoods(testTrial, iBin, :) = logLikelihood;
        end
    end

    errors_control{ianimal} = predicted_bins - actual_bins;  % [nTrials x nBins]
    predicted_bins_control{ianimal} = predicted_bins;
    actual_bins_control{ianimal} = actual_bins;


    window_size = 5;
    half_window = floor(window_size / 2);

    % Preallocate the output matrix with NaNs to handle cases with insufficient data
    avg_corrs = NaN(neurons, change_point);

    % Loop over each neuron
    for n = 1:neurons
        % Loop over each trial
        for t = 1:change_point
            % Define the window of trials centered on trial t
            trial_start = max(1, t - half_window);
            trial_end = min(trials, t + half_window);
            trials_in_window = trial_start:trial_end;
            num_trials_in_window = length(trials_in_window);

            % Proceed only if we have at least two trials to correlate
            if num_trials_in_window > 1
                % Extract the data for the current neuron and window
                data_block = squeeze(current_activity(n, :, trials_in_window)); % Size: [spatial_bins x num_trials_in_window]

                % Compute the correlation matrix for the trials in the current window
                R = corr(data_block);

                % Extract the upper triangle of the correlation matrix (excluding the diagonal)
                upper_triangle = triu(R, 1);
                upper_vals = upper_triangle(upper_triangle ~= 0);

                % Compute the average correlation
                avg_corrs(n, t) = mean(upper_vals);

            else
                % If only one trial is available, set the average correlation to NaN
                avg_corrs(n, t) = NaN;
            end
        end
    end

    avg_corrs_control{ianimal} = avg_corrs;

    % Loop over each area (DMS, DLS, ACC)
    areas = {'dms', 'dls', 'acc'};
    activities = {current_activity_dms, current_activity_dls, current_activity_acc};

    for a = 1:length(areas)
        area_activity = activities{a};
        [neurons_area, ~, ~] = size(area_activity);

        % Preallocate NaN matrix for current area
        avg_corrs = NaN(neurons_area, change_point);

        % Loop over each neuron in the current area
        for n = 1:neurons_area
            % Loop over each trial
            for t = 1:change_point
                % Define the window of trials centered on trial t
                trial_start = max(1, t - half_window);
                trial_end = min(trials, t + half_window);
                trials_in_window = trial_start:trial_end;
                num_trials_in_window = length(trials_in_window);

                % Proceed only if we have at least two trials to correlate
                if num_trials_in_window > 1
                    % Extract the data for the current neuron and window
                    data_block = squeeze(area_activity(n, :, trials_in_window)); % Size: [spatial_bins x num_trials_in_window]

                    % Compute the correlation matrix for the trials in the current window
                    R = corr(data_block);

                    % Extract the upper triangle of the correlation matrix (excluding the diagonal)
                    upper_triangle = triu(R, 1);
                    upper_vals = upper_triangle(upper_triangle ~= 0);

                    % Compute the average correlation
                    avg_corrs(n, t) = mean(upper_vals);
                else
                    % If only one trial is available, set the average correlation to NaN
                    avg_corrs(n, t) = NaN;
                end
            end
        end

        % Store results for the current area and animal
        avg_corrs_control_area{a}{ianimal} = avg_corrs;
    end

end
% Assign the final matrices to variables
avg_corrs_dms_control = avg_corrs_control_area{1};
avg_corrs_dls_control = avg_corrs_control_area{2};
avg_corrs_acc_control = avg_corrs_control_area{3};

all_activity_control = {control_data(:).spatial_binned_fr_all};
mean_activity_control_spatial_neurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_control, 'UniformOutput', false);
mean_activity_control_spatial_neurons = cat(1, mean_activity_control_spatial_neurons{:});

mean_activity_control = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_control, 'UniformOutput', false);
mean_activity_control_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_control, 'UniformOutput', false);
mean_activity_control_spatial = cat(1, mean_activity_control_spatial{:});

mean_activity_control_spatial_dmsneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_dms_control, 'UniformOutput', false);
mean_activity_control_spatial_dmsneurons = cat(1, mean_activity_control_spatial_dmsneurons{:});

mean_activity_control_dms = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_dms_control, 'UniformOutput', false);
mean_activity_control_dms_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_control_dms, 'UniformOutput', false);
mean_activity_control_dms_spatial = cat(1, mean_activity_control_dms_spatial{:});

mean_activity_control_spatial_dlsneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_dls_control, 'UniformOutput', false);
mean_activity_control_spatial_dlsneurons = cat(1, mean_activity_control_spatial_dlsneurons{:});

mean_activity_control_dls = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_dls_control, 'UniformOutput', false);
mean_activity_control_dls_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_control_dls, 'UniformOutput', false);
mean_activity_control_dls_spatial = cat(1, mean_activity_control_dls_spatial{:});

mean_activity_control_spatial_accneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_acc_control, 'UniformOutput', false);
mean_activity_control_spatial_accneurons = cat(1, mean_activity_control_spatial_accneurons{:});

mean_activity_control_acc = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_acc_control, 'UniformOutput', false);
mean_activity_control_acc_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_control_acc, 'UniformOutput', false);
mean_activity_control_acc_spatial = cat(1, mean_activity_control_acc_spatial{:});

for ianimal = 1:n_animals_control2

    is_dms = control2_data(ianimal).is_dms;
    is_dls = control2_data(ianimal).is_dls;
    is_acc = control2_data(ianimal).is_acc;

    current_activity = control2_data(ianimal).firing_rates_per_bin;
    current_activity_dms = current_activity(is_dms, :, :);
    current_activity_dls = current_activity(is_dls, :, :);
    current_activity_acc = current_activity(is_acc, :, :);

    all_activity_dms_control2{ianimal} = current_activity_dms;
    all_activity_dls_control2{ianimal} = current_activity_dls;
    all_activity_acc_control2{ianimal} = current_activity_acc;

    [neurons, bins, trials] = size(current_activity);

    try
        change_point = min([control2_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end

    for ibin = 1:bins
        p = polyfit(mean(current_activity(:, ibin, :), 3, 'omitmissing'), var(current_activity(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_control2{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_dms(:, ibin, :), 3, 'omitmissing'), var(current_activity_dms(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_control2_dms{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_dls(:, ibin, :), 3, 'omitmissing'), var(current_activity_dls(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_control2_dls{ianimal}(ibin) = p(1);

        p = polyfit(mean(current_activity_acc(:, ibin, :), 3, 'omitmissing'), var(current_activity_acc(:, ibin, :), [], 3, 'omitmissing'), 1);  % Fit a 1st-degree polynomial (a straight line)
        population_ff_control2_acc{ianimal}(ibin) = p(1);
    end

    genvar_control2{ianimal} = estimate_trialwise_variance(current_activity(:, :, 1:change_point));
    genvar_control2_dms{ianimal} = estimate_trialwise_variance(current_activity_dms(:, :, 1:change_point));
    genvar_control2_dls{ianimal} = estimate_trialwise_variance(current_activity_dls(:, :, 1:change_point));
    genvar_control2_acc{ianimal} = estimate_trialwise_variance(current_activity_acc(:, :, 1:change_point));

    mask_none = false(1, neurons);

    doShuffle = false;  % set to true for shuffle control
    ablation_mask = mask_none;  % set to mask_dms, mask_dls, mask_acc, etc.

    predicted_bins = NaN(change_point, bins);  % store final predictions
    actual_bins = repmat(1:bins, [change_point, 1]);  % ground truth: bin i is labelled i

    all_log_likelihoods = nan(change_point, bins, bins);

    for testTrial = 1:change_point
        fprintf('decoding trial %d/%d\n', testTrial, change_point)
        trainTrials = setdiff(1:change_point, testTrial);
        % ablation_mask is 1 for neurons we want to exclude
        useNeurons = ~ablation_mask;  % useNeurons is 1 for kept neurons

        % Extract the training data for these neurons
        trainData = current_activity(useNeurons, :, trainTrials);
        % trainData size is [nUsedNeurons x nBins x (nTrials-1)]

        meanFR = mean(trainData, 3, 'omitnan');

        if doShuffle
            for ineuron = 1:size(meanFR,1)
                % random permutation of bin axis for that neuron
                meanFR(ineuron, :) = meanFR(ineuron, randperm(bins));
            end
        end

        %----------------------------------------------------------------------
        % 5) Decode the left-out trial: for each bin in testTrial, compute
        %    likelihood for each candidate bin, choose the best.
        %----------------------------------------------------------------------
        testData = current_activity(useNeurons, :, testTrial);  % [nUsedNeurons x nBins]

        for iBin = 1:bins
            % Observed firing in that bin: [nUsedNeurons x 1]
            observedCounts = testData(:, iBin);

            % We'll compute log-likelihood under each candidate bin
            % for each neuron, then sum across neurons.
            % Poisson: p(r|lambda) = exp(-lambda) * lambda^r / r!
            % We'll do log(p(r|lambda)) to avoid underflow:
            % log p(r|lambda) = -lambda + r*log(lambda) - log(r!)
            % We'll skip the log(r!) term as it doesn't affect argmax.

            % meanFR is [nUsedNeurons x nBins], so for candidate bin c
            % the mean rate is meanFR(:, c).

            logLikelihood = zeros(bins, 1);  % store log-likelihood for each candidate bin
            for c = 1:bins
                lambda_c = meanFR(:, c);  % [nUsedNeurons x 1]
                % Avoid zero or negative
                lambda_c(lambda_c<=0) = 1e-6;

                % observedCounts is r, also [nUsedNeurons x 1]
                % Summation over neurons of r*log(lambda) - lambda
                ll = observedCounts .* log(lambda_c) - lambda_c;
                logLikelihood(c) = sum(ll, 'omitnan');
            end

            % Choose the bin c that maximises the log-likelihood
            % Instead of max, break ties randomly:
            maxVal = max(logLikelihood);
            maxInds = find(logLikelihood == maxVal);
            if numel(maxInds) > 1
                bestBin = maxInds(randi(numel(maxInds)));
            else
                bestBin = maxInds;
            end
            predicted_bins(testTrial, iBin) = bestBin;

            all_log_likelihoods(testTrial, iBin, :) = logLikelihood;
        end
    end

    errors_control2{ianimal} = predicted_bins - actual_bins;  % [nTrials x nBins]
    predicted_bins_control2{ianimal} = predicted_bins;
    actual_bins_control2{ianimal} = actual_bins;

    window_size = 5;
    half_window = floor(window_size / 2);

    % Preallocate the output matrix with NaNs to handle cases with insufficient data
    avg_corrs = NaN(neurons, change_point);

    % Loop over each neuron
    for n = 1:neurons
        % Loop over each trial
        for t = 1:change_point
            % Define the window of trials centered on trial t
            trial_start = max(1, t - half_window);
            trial_end = min(trials, t + half_window);
            trials_in_window = trial_start:trial_end;
            num_trials_in_window = length(trials_in_window);

            % Proceed only if we have at least two trials to correlate
            if num_trials_in_window > 1
                % Extract the data for the current neuron and window
                data_block = squeeze(current_activity(n, :, trials_in_window)); % Size: [spatial_bins x num_trials_in_window]

                % Compute the correlation matrix for the trials in the current window
                R = corr(data_block);

                % Extract the upper triangle of the correlation matrix (excluding the diagonal)
                upper_triangle = triu(R, 1);
                upper_vals = upper_triangle(upper_triangle ~= 0);

                % Compute the average correlation
                avg_corrs(n, t) = mean(upper_vals);

            else
                % If only one trial is available, set the average correlation to NaN
                avg_corrs(n, t) = NaN;
            end
        end
    end

    avg_corrs_control2{ianimal} = avg_corrs;

    % Loop over each area (DMS, DLS, ACC)
    areas = {'dms', 'dls', 'acc'};
    activities = {current_activity_dms, current_activity_dls, current_activity_acc};

    for a = 1:length(areas)
        area_activity = activities{a};
        [neurons_area, ~, ~] = size(area_activity);

        % Preallocate NaN matrix for current area
        avg_corrs = NaN(neurons_area, change_point);

        % Loop over each neuron in the current area
        for n = 1:neurons_area
            % Loop over each trial
            for t = 1:change_point
                % Define the window of trials centered on trial t
                trial_start = max(1, t - half_window);
                trial_end = min(trials, t + half_window);
                trials_in_window = trial_start:trial_end;
                num_trials_in_window = length(trials_in_window);

                % Proceed only if we have at least two trials to correlate
                if num_trials_in_window > 1
                    % Extract the data for the current neuron and window
                    data_block = squeeze(area_activity(n, :, trials_in_window)); % Size: [spatial_bins x num_trials_in_window]

                    % Compute the correlation matrix for the trials in the current window
                    R = corr(data_block);

                    % Extract the upper triangle of the correlation matrix (excluding the diagonal)
                    upper_triangle = triu(R, 1);
                    upper_vals = upper_triangle(upper_triangle ~= 0);

                    % Compute the average correlation
                    avg_corrs(n, t) = mean(upper_vals);
                else
                    % If only one trial is available, set the average correlation to NaN
                    avg_corrs(n, t) = NaN;
                end
            end
        end

        % Store results for the current area and animal
        avg_corrs_control2_area{a}{ianimal} = avg_corrs;
    end

end
% Assign the final matrices to variables
avg_corrs_dms_control2 = avg_corrs_control2_area{1};
avg_corrs_dls_control2 = avg_corrs_control2_area{2};
avg_corrs_acc_control2 = avg_corrs_control2_area{3};

all_activity_control2 = {control2_data(:).firing_rates_per_bin};
mean_activity_control2_spatial_neurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_control2, 'UniformOutput', false);
mean_activity_control2_spatial_neurons = cat(1, mean_activity_control2_spatial_neurons{:});

mean_activity_control2 = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_control2, 'UniformOutput', false);
mean_activity_control2_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_control2, 'UniformOutput', false);
mean_activity_control2_spatial = cat(1, mean_activity_control2_spatial{:});

mean_activity_control2_spatial_dmsneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_dms_control2, 'UniformOutput', false);
mean_activity_control2_spatial_dmsneurons = cat(1, mean_activity_control2_spatial_dmsneurons{:});

mean_activity_control2_dms = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_dms_control2, 'UniformOutput', false);
mean_activity_control2_dms_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_control2_dms, 'UniformOutput', false);
mean_activity_control2_dms_spatial = cat(1, mean_activity_control2_dms_spatial{:});

mean_activity_control2_spatial_dlsneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_dls_control2, 'UniformOutput', false);
mean_activity_control2_spatial_dlsneurons = cat(1, mean_activity_control2_spatial_dlsneurons{:});

mean_activity_control2_dls = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_dls_control2, 'UniformOutput', false);
mean_activity_control2_dls_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_control2_dls, 'UniformOutput', false);
mean_activity_control2_dls_spatial = cat(1, mean_activity_control2_dls_spatial{:});

mean_activity_control2_spatial_accneurons = cellfun(@(x) mean(x, 3, 'omitmissing'), all_activity_acc_control2, 'UniformOutput', false);
mean_activity_control2_spatial_accneurons = cat(1, mean_activity_control2_spatial_accneurons{:});

mean_activity_control2_acc = cellfun(@(x) squeeze(mean(x, 1, 'omitmissing')), all_activity_acc_control2, 'UniformOutput', false);
mean_activity_control2_acc_spatial = cellfun(@(x) squeeze(mean(x, 2, "omitmissing"))', mean_activity_control2_acc, 'UniformOutput', false);
mean_activity_control2_acc_spatial = cat(1, mean_activity_control2_acc_spatial{:});


%% Analysis (new, only task)
n_animals_task = numel(task_data);
n_animals_control = numel(control_data);
n_animals_control2 = numel(control2_data);

% ----- TASK DATA ANALYSIS -----
for ianimal = 1:n_animals_task
    %%% Extract flags and data for this animal
    is_dms = task_data(ianimal).is_dms;
    is_dls = task_data(ianimal).is_dls;
    is_acc = task_data(ianimal).is_acc;
    
    current_activity = task_data(ianimal).spatial_binned_fr_all; % neurons x bins x trials
    [neurons, bins, trials] = size(current_activity);
    
    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end
    current_activity = current_activity(:, :, 1:change_point);
    
    % Separate by area
    current_activity_dms = current_activity(is_dms, :, :);
    current_activity_dls = current_activity(is_dls, :, :);
    current_activity_acc = current_activity(is_acc, :, :);
    
    %----- FULL-POPULATION DECODING (TASK) -----
    doShuffle = false;  % set to true for shuffle control
    % If more neurons than target, do subsampling; otherwise use all neurons
    if neurons > 0  % (your full-population decoding code remains here)
        if neurons > 50  % example threshold for subsampling if needed
            % (Your original subsampling code could go here; otherwise, simply use all neurons)
            predicted_bins_all = NaN(change_point, bins, 10);
            for iter = 1:10
                sample_idx = randsample(neurons, neurons);
                current_activity_sub = current_activity(sample_idx, :, :);
                predicted_bins_sub = NaN(change_point, bins);
                for testTrial = 1:change_point
                    fprintf('Full-pop decoding trial %d/%d (Animal %d, iter %d)\n', testTrial, change_point, ianimal, iter)
                    trainTrials = setdiff(1:change_point, testTrial);
                    trainData = current_activity_sub(:, :, trainTrials);
                    meanFR = mean(trainData, 3, 'omitnan');
                    if doShuffle
                        for ineuron = 1:size(meanFR,1)
                            meanFR(ineuron, :) = meanFR(ineuron, randperm(bins));
                        end
                    end
                    testData = current_activity_sub(:, :, testTrial);
                    for iBin = 1:bins
                        observedCounts = testData(:, iBin);
                        logLikelihood = zeros(bins, 1);
                        for c = 1:bins
                            lambda_c = meanFR(:, c);
                            lambda_c(lambda_c<=0) = 1e-6;
                            ll = observedCounts .* log(lambda_c) - lambda_c;
                            logLikelihood(c) = sum(ll, 'omitnan');
                        end
                        maxVal = max(logLikelihood);
                        maxInds = find(logLikelihood == maxVal);
                        if numel(maxInds) > 1
                            bestBin = maxInds(randi(numel(maxInds)));
                        else
                            bestBin = maxInds;
                        end
                        predicted_bins_sub(testTrial, iBin) = bestBin;
                    end
                end
                predicted_bins_all(:,:,iter) = predicted_bins_sub;
            end
            predicted_bins = mode(predicted_bins_all, 3);
        else
            predicted_bins = NaN(change_point, bins);
            for testTrial = 1:change_point
                fprintf('Full-pop decoding trial %d/%d (Animal %d)\n', testTrial, change_point, ianimal)
                trainTrials = setdiff(1:change_point, testTrial);
                trainData = current_activity(:, :, trainTrials);
                meanFR = mean(trainData, 3, 'omitnan');
                if doShuffle
                    for ineuron = 1:size(meanFR,1)
                        meanFR(ineuron, :) = meanFR(ineuron, randperm(bins));
                    end
                end
                testData = current_activity(:, :, testTrial);
                for iBin = 1:bins
                    observedCounts = testData(:, iBin);
                    logLikelihood = zeros(bins, 1);
                    for c = 1:bins
                        lambda_c = meanFR(:, c);
                        lambda_c(lambda_c<=0) = 1e-6;
                        ll = observedCounts .* log(lambda_c) - lambda_c;
                        logLikelihood(c) = sum(ll, 'omitnan');
                    end
                    maxVal = max(logLikelihood);
                    maxInds = find(logLikelihood == maxVal);
                    if numel(maxInds) > 1
                        bestBin = maxInds(randi(numel(maxInds)));
                    else
                        bestBin = maxInds;
                    end
                    predicted_bins(testTrial, iBin) = bestBin;
                end
            end
        end
    end
    actual_bins = repmat(1:bins, [change_point, 1]);
    errors_task{ianimal} = predicted_bins - actual_bins;
    predicted_bins_task{ianimal} = predicted_bins;
    actual_bins_task{ianimal} = actual_bins;
    
    %----- AREA-SPECIFIC DECODING (TASK) -----
    % For each area, use all available neurons if count > 5, otherwise skip.
    areas = {'dms', 'dls', 'acc'};
    activities = {current_activity_dms, current_activity_dls, current_activity_acc};
    for a = 1:length(areas)
        area_activity = activities{a};
        [neurons_area, ~, ~] = size(area_activity);
        if neurons_area > 5
            predicted_bins_area = NaN(change_point, bins);
            for testTrial = 1:change_point
                fprintf('Decoding %s trial %d/%d (Animal %d)\n', upper(areas{a}), testTrial, change_point, ianimal);
                trainTrials = setdiff(1:change_point, testTrial);
                trainData = area_activity(:, :, trainTrials);
                meanFR = mean(trainData, 3, 'omitnan');
                if doShuffle
                    for ineuron = 1:size(meanFR,1)
                        meanFR(ineuron, :) = meanFR(ineuron, randperm(bins));
                    end
                end
                testData = area_activity(:, :, testTrial);
                for iBin = 1:bins
                    observedCounts = testData(:, iBin);
                    logLikelihood = zeros(bins,1);
                    for c = 1:bins
                        lambda_c = meanFR(:, c);
                        lambda_c(lambda_c <= 0) = 1e-6;
                        ll = observedCounts .* log(lambda_c) - lambda_c;
                        logLikelihood(c) = sum(ll, 'omitnan');
                    end
                    maxVal = max(logLikelihood);
                    maxInds = find(logLikelihood==maxVal);
                    if numel(maxInds)>1
                        bestBin = maxInds(randi(numel(maxInds)));
                    else
                        bestBin = maxInds;
                    end
                    predicted_bins_area(testTrial, iBin) = bestBin;
                end
            end
            actual_bins_area = repmat(1:bins, [change_point, 1]);
            errors_area_task{a}{ianimal} = predicted_bins_area - actual_bins_area;
            predicted_bins_area_task{a}{ianimal} = predicted_bins_area;
            actual_bins_area_task{a}{ianimal} = actual_bins_area;
        else
            fprintf('Skipping decoding for %s in animal %d due to insufficient neurons (%d)\n', upper(areas{a}), ianimal, neurons_area);
            errors_area_task{a}{ianimal} = [];
            predicted_bins_area_task{a}{ianimal} = [];
            actual_bins_area_task{a}{ianimal} = [];
        end
    end
    
    %----- Trial-to-Trial Correlation (Per-Neuron) -----
    window_size = 5;
    half_window = floor(window_size / 2);
    avg_corrs = NaN(neurons, change_point);
    for n = 1:neurons
        for t = 1:change_point
            trial_start = max(1, t - half_window);
            trial_end = min(change_point, t + half_window);
            trials_in_window = trial_start:trial_end;
            if numel(trials_in_window) > 1
                data_block = squeeze(current_activity(n, :, trials_in_window));
                R = corr(data_block);
                upper_triangle = triu(R, 1);
                upper_vals = upper_triangle(upper_triangle ~= 0);
                avg_corrs(n, t) = mean(upper_vals);
            else
                avg_corrs(n, t) = NaN;
            end
        end
    end
    avg_corrs_task{ianimal} = avg_corrs;
    
    %----- Trial-to-Trial Correlation (Per Area) -----
    areas = {'dms', 'dls', 'acc'};
    activities = {current_activity_dms, current_activity_dls, current_activity_acc};
    for a = 1:length(areas)
        area_activity = activities{a};
        [neurons_area, ~, ~] = size(area_activity);
        avg_corrs_area = NaN(neurons_area, change_point);
        for n = 1:neurons_area
            for t = 1:change_point
                trial_start = max(1, t - half_window);
                trial_end = min(change_point, t + half_window);
                trials_in_window = trial_start:trial_end;
                if numel(trials_in_window) > 1
                    data_block = squeeze(area_activity(n, :, trials_in_window));
                    R = corr(data_block);
                    upper_triangle = triu(R, 1);
                    upper_vals = upper_triangle(upper_triangle ~= 0);
                    avg_corrs_area(n, t) = mean(upper_vals);
                else
                    avg_corrs_area(n, t) = NaN;
                end
            end
        end
        avg_corrs_task_area{a}{ianimal} = avg_corrs_area;
    end
end

%% Spatial FR

figure
t = tiledlayout(4, 3, "TileSpacing", "compact");
nexttile
shadedErrorBar(1:bins, mean(mean_activity_task_spatial, "omitmissing"), sem(mean_activity_task_spatial))
title('task')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control_spatial, "omitmissing"), sem(mean_activity_control_spatial))
title('control')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control2_spatial, "omitmissing"), sem(mean_activity_control2_spatial))
title('control2')
xlabel(t, 'position')
ylabel(t, 'firing rate')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_task_dms_spatial, "omitmissing"), sem(mean_activity_task_dms_spatial))
title('DMS task')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control_dms_spatial, "omitmissing"), sem(mean_activity_control_dms_spatial))
title('DMS control')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control2_dms_spatial, "omitmissing"), sem(mean_activity_control2_dms_spatial))
title('DMS control2')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_task_dls_spatial, "omitmissing"), sem(mean_activity_task_dls_spatial))
title('DLS task')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control_dls_spatial, "omitmissing"), sem(mean_activity_control_dls_spatial))
title('DLS control')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control2_dls_spatial, "omitmissing"), sem(mean_activity_control2_dls_spatial))
title('DLS control2')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_task_acc_spatial, "omitmissing"), sem(mean_activity_task_acc_spatial))
title('ACC task')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control_acc_spatial, "omitmissing"), sem(mean_activity_control_acc_spatial))
title('ACC control')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control2_acc_spatial, "omitmissing"), sem(mean_activity_control2_acc_spatial))
title('ACC control2')


figure
t = tiledlayout(4, 3, "TileSpacing", "compact");
nexttile
shadedErrorBar(1:bins, mean(mean_activity_task_spatial_neurons, "omitmissing"), sem(mean_activity_task_spatial_neurons))
title('task')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control_spatial_neurons, "omitmissing"), sem(mean_activity_control_spatial_neurons))
title('control')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control2_spatial_neurons, "omitmissing"), sem(mean_activity_control2_spatial_neurons))
title('control2')
xlabel(t, 'position')
ylabel(t, 'firing rate')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_task_spatial_dmsneurons, "omitmissing"), sem(mean_activity_task_spatial_dmsneurons))
title('DMS task')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control_spatial_dmsneurons, "omitmissing"), sem(mean_activity_control_spatial_dmsneurons))
title('DMS control')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control2_spatial_dmsneurons, "omitmissing"), sem(mean_activity_control2_spatial_dmsneurons))
title('DMS control2')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_task_spatial_dlsneurons, "omitmissing"), sem(mean_activity_task_spatial_dlsneurons))
title('DLS task')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control_spatial_dlsneurons, "omitmissing"), sem(mean_activity_control_spatial_dlsneurons))
title('DLS control')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control2_spatial_dlsneurons, "omitmissing"), sem(mean_activity_control2_spatial_dlsneurons))
title('DLS control2')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_task_spatial_accneurons, "omitmissing"), sem(mean_activity_task_spatial_accneurons))
title('ACC task')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control_spatial_accneurons, "omitmissing"), sem(mean_activity_control_spatial_accneurons))
title('ACC control')
nexttile
shadedErrorBar(1:bins, mean(mean_activity_control2_spatial_accneurons, "omitmissing"), sem(mean_activity_control2_spatial_accneurons))
title('ACC control2')
axis tight
linkaxes


%% Trials FR

mean_activity_task_trials_neurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_task, 'UniformOutput', false);
mean_activity_task_trials_neurons = cat(1, mean_activity_task_trials_neurons{:});

mean_activity_task_trials_dmsneurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_dms_task, 'UniformOutput', false);
mean_activity_task_trials_dmsneurons = cat(1, mean_activity_task_trials_dmsneurons{:});

mean_activity_task_trials_dlsneurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_dls_task, 'UniformOutput', false);
mean_activity_task_trials_dlsneurons = cat(1, mean_activity_task_trials_dlsneurons{:});

mean_activity_task_trials_accneurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_acc_task, 'UniformOutput', false);
mean_activity_task_trials_accneurons = cat(1, mean_activity_task_trials_accneurons{:});

mean_activity_control_trials_neurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_control, 'UniformOutput', false);
mean_activity_control_trials_neurons = cat(1, mean_activity_control_trials_neurons{:});

mean_activity_control_trials_dmsneurons = cellfun(@(x) mean(x(:, :, 1:50), 2, 'omitmissing'), all_activity_dms_control, 'UniformOutput', false);
mean_activity_control_trials_dmsneurons = squeeze(cat(1, mean_activity_control_trials_dmsneurons{:}));

mean_activity_control_trials_dlsneurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_dls_control, 'UniformOutput', false);
mean_activity_control_trials_dlsneurons = cat(1, mean_activity_control_trials_dlsneurons{:});

mean_activity_control_trials_accneurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_acc_control, 'UniformOutput', false);
mean_activity_control_trials_accneurons = cat(1, mean_activity_control_trials_accneurons{:});

mean_activity_control2_trials_neurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_control2, 'UniformOutput', false);
mean_activity_control2_trials_neurons = cat(1, mean_activity_control2_trials_neurons{:});

mean_activity_control2_trials_dmsneurons = cellfun(@(x) mean(x(:, :, 1:50), 2, 'omitmissing'), all_activity_dms_control2, 'UniformOutput', false);
mean_activity_control2_trials_dmsneurons = squeeze(cat(1, mean_activity_control2_trials_dmsneurons{:}));

mean_activity_control2_trials_dlsneurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_dls_control2, 'UniformOutput', false);
mean_activity_control2_trials_dlsneurons = cat(1, mean_activity_control2_trials_dlsneurons{:});

mean_activity_control2_trials_accneurons = cellfun(@(x) squeeze(mean(x(:, :, 1:50), 2, 'omitmissing')), all_activity_acc_control2, 'UniformOutput', false);
mean_activity_control2_trials_accneurons = cat(1, mean_activity_control2_trials_accneurons{:});

figure
t = tiledlayout(4, 3, "TileSpacing", "compact");
nexttile
shadedErrorBar(1:50, mean(mean_activity_task_trials_neurons, "omitmissing"), sem(mean_activity_task_trials_neurons))
title('task')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_control_trials_neurons, "omitmissing"), sem(mean_activity_control_trials_neurons))
title('control')
nexttile
shadedErrorBar(1:50, mean(mean_activity_control2_trials_neurons, "omitmissing"), sem(mean_activity_control2_trials_neurons))
title('control2')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_task_trials_dmsneurons, "omitmissing"), sem(mean_activity_task_trials_dmsneurons))
title('DMS task')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_control_trials_dmsneurons, "omitmissing"), sem(mean_activity_control_trials_dmsneurons))
title('DMS control')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_control2_trials_dmsneurons, "omitmissing"), sem(mean_activity_control2_trials_dmsneurons))
title('DMS control2')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_task_trials_dlsneurons, "omitmissing"), sem(mean_activity_task_trials_dlsneurons))
title('DLS task')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_control_trials_dlsneurons, "omitmissing"), sem(mean_activity_control_trials_dlsneurons))
title('DLS control')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_control2_trials_dlsneurons, "omitmissing"), sem(mean_activity_control2_trials_dlsneurons))
title('DLS control2')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_task_trials_accneurons, "omitmissing"), sem(mean_activity_task_trials_accneurons))
title('ACC task')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_control_trials_accneurons, "omitmissing"), sem(mean_activity_control_trials_accneurons))
title('ACC control')
axis tight
nexttile
shadedErrorBar(1:50, mean(mean_activity_control2_trials_accneurons, "omitmissing"), sem(mean_activity_control2_trials_accneurons))
title('ACC control2')
axis tight
linkaxes
xlabel(t, 'trials')
ylabel(t, 'firing rate')

%% Population FF

pop_ff_task_all = cat(1, population_ff_task{:});
pop_ff_control_all = cat(1, population_ff_control{:});
pop_ff_control2_all = cat(1, population_ff_control2{:});

pop_ff_task_dms = cat(1, population_ff_task_dms{:});
pop_ff_control_dms = cat(1, population_ff_control_dms{:});
pop_ff_control2_dms = cat(1, population_ff_control2_dms{:});

pop_ff_task_dls = cat(1, population_ff_task_dls{:});
pop_ff_control_dls = cat(1, population_ff_control_dls{:});
pop_ff_control2_dls = cat(1, population_ff_control2_dls{:});

pop_ff_task_acc = cat(1, population_ff_task_acc{:});
pop_ff_control_acc = cat(1, population_ff_control_acc{:});
pop_ff_control2_acc = cat(1, population_ff_control2_acc{:});


figure
t = tiledlayout(4, 3, "TileSpacing", "compact");
nexttile
shadedErrorBar(1:bins, mean(pop_ff_task_all), sem(pop_ff_task_all))
title('task')
nexttile
shadedErrorBar(1:bins, mean(pop_ff_control_all), sem(pop_ff_control_all))
title('control')
nexttile
shadedErrorBar(1:bins, mean(pop_ff_control2_all), sem(pop_ff_control2_all))
title('control2')

nexttile
shadedErrorBar(1:bins, mean(pop_ff_task_dms), sem(pop_ff_task_dms))
title('DMS task')
nexttile
shadedErrorBar(1:bins, mean(pop_ff_control_dms), sem(pop_ff_control_dms))
title('DMS control')
nexttile
shadedErrorBar(1:bins, mean(pop_ff_control2_dms), sem(pop_ff_control2_dms))
title('DMS control2')

nexttile
shadedErrorBar(1:bins, mean(pop_ff_task_dls), sem(pop_ff_task_dls))
title('DLS task')
nexttile
shadedErrorBar(1:bins, mean(pop_ff_control_dls), sem(pop_ff_control_dls))
title('DLS control')
nexttile
shadedErrorBar(1:bins, mean(pop_ff_control2_dls), sem(pop_ff_control2_dls))
title('DLS control2')

nexttile
shadedErrorBar(1:bins, mean(pop_ff_task_acc), sem(pop_ff_task_acc))
title('ACC task')
nexttile
shadedErrorBar(1:bins, mean(pop_ff_control_acc), sem(pop_ff_control_acc))
title('ACC control')
nexttile
shadedErrorBar(1:bins, mean(pop_ff_control2_acc), sem(pop_ff_control2_acc))
title('ACC control2')
linkaxes
xlabel(t, 'position')
ylabel(t, 'population FF')

%% Generalised Variance

figure
t = tiledlayout(4, 3, "TileSpacing", "compact");
nexttile
shadedErrorBar(1:bins, mean(genvar_task, 'omitmissing'), sem(genvar_task))
title('task')
nexttile
shadedErrorBar(1:bins, mean(genvar_control, 'omitmissing'), sem(genvar_control))
title('control')
nexttile
shadedErrorBar(1:bins, mean(genvar_control2, 'omitmissing'), sem(genvar_control2))
title('control2')

nexttile
shadedErrorBar(1:bins, mean(genvar_task_dms, 'omitmissing'), sem(genvar_task_dms))
title('DMS task')
nexttile
shadedErrorBar(1:bins, mean(genvar_control_dms, 'omitmissing'), sem(genvar_control_dms))
title('DMS control')
nexttile
shadedErrorBar(1:bins, mean(genvar_control2_dms, 'omitmissing'), sem(genvar_control2_dms))
title('DMS control2')

nexttile
shadedErrorBar(1:bins, mean(genvar_task_dls, 'omitmissing'), sem(genvar_task_dls))
title('DLS task')
nexttile
shadedErrorBar(1:bins, mean(genvar_control_dls, 'omitmissing'), sem(genvar_control_dls))
title('DLS control')
nexttile
shadedErrorBar(1:bins, mean(genvar_control2_dls, 'omitmissing'), sem(genvar_control2_dls))
title('DLS control2')

nexttile
shadedErrorBar(1:bins, mean(genvar_task_acc, 'omitmissing'), sem(genvar_task_acc))
title('ACC task')
nexttile
shadedErrorBar(1:bins, mean(genvar_control_acc, 'omitmissing'), sem(genvar_control_acc))
title('ACC control')
nexttile
shadedErrorBar(1:bins, mean(genvar_control2_acc, 'omitmissing'), sem(genvar_control2_acc))
title('ACC control2')
linkaxes
xlabel(t, 'trials')
ylabel(t, 'generalised variance')

%% Decoding
figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_task
    nexttile
    scatter(predicted_bins_task{ianimal}(:), actual_bins_task{ianimal}(:))
    axis tight
    axis square
end
xlabel(t, 'predicted bin')
ylabel(t, 'true bin')
title(t, 'task')

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_control
    nexttile
    scatter(predicted_bins_control{ianimal}(:), actual_bins_control{ianimal}(:))
    axis tight
    axis square
end
xlabel(t, 'predicted bin')
ylabel(t, 'true bin')
title(t, 'control')

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_control2
    nexttile
    scatter(predicted_bins_control2{ianimal}(:), actual_bins_control2{ianimal}(:))
    axis tight
    axis square
end
xlabel(t, 'predicted bin')
ylabel(t, 'true bin')
title(t, 'control2')

%% Decoding contour

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_task
    nexttile

    gridx1 = 0:.1:bins;
    gridx2 = gridx1;
    [x1,x2] = meshgrid(gridx1, gridx2);
    x1 = x1(:);
    x2 = x2(:);
    xi = [x1 x2];
    x = [predicted_bins_task{ianimal}(:), actual_bins_task{ianimal}(:)];
    ksdensity(x, xi, 'PlotFcn', 'contour');
    contourObj = findobj(gca, 'Type', 'Contour');
    set(contourObj, 'LineWidth', 1.2);

    axis tight
    axis square
end
xlabel(t, 'predicted bin')
ylabel(t, 'true bin')
title(t, 'task')

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_control
    nexttile

    gridx1 = 0:.1:bins;
    gridx2 = gridx1;
    [x1,x2] = meshgrid(gridx1, gridx2);
    x1 = x1(:);
    x2 = x2(:);
    xi = [x1 x2];
    x = [predicted_bins_control{ianimal}(:), actual_bins_control{ianimal}(:)];
    ksdensity(x, xi, 'PlotFcn', 'contour');
    contourObj = findobj(gca, 'Type', 'Contour');
    set(contourObj, 'LineWidth', 1.2);

    axis tight
    axis square
end
xlabel(t, 'predicted bin')
ylabel(t, 'true bin')
title(t, 'control')

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_control2
    nexttile

    gridx1 = 0:.1:bins;
    gridx2 = gridx1;
    [x1,x2] = meshgrid(gridx1, gridx2);
    x1 = x1(:);
    x2 = x2(:);
    xi = [x1 x2];
    x = [predicted_bins_control2{ianimal}(:), actual_bins_control2{ianimal}(:)];
    ksdensity(x, xi, 'PlotFcn', 'contour');
    contourObj = findobj(gca, 'Type', 'Contour');
    set(contourObj, 'LineWidth', 1.2);

    axis tight
    axis square
end
xlabel(t, 'predicted bin')
ylabel(t, 'true bin')
title(t, 'control2')


%% Decoding errors
trial_average_errors_task = cellfun(@(x) mean(abs(x), 2, 'omitmissing')', errors_task, 'UniformOutput', false);
trial_average_errors_task = cat(1, trial_average_errors_task{:});

trial_average_errors_control = cellfun(@(x) mean(abs(x), 2, 'omitmissing')', errors_control, 'UniformOutput', false);
trial_average_errors_control = cat(1, trial_average_errors_control{:});

trial_average_errors_control2 = cellfun(@(x) mean(abs(x), 2, 'omitmissing')', errors_control2, 'UniformOutput', false);
trial_average_errors_control2 = cat(1, trial_average_errors_control2{:});

figure
hold on
h1 = shadedErrorBar(1:50, mean(trial_average_errors_task), sem(trial_average_errors_task), 'lineprops', {'Color', 'r'});
h2 = shadedErrorBar(1:50, mean(trial_average_errors_control), sem(trial_average_errors_control), 'lineprops', {'Color', 'b'});
h3 = shadedErrorBar(1:50, mean(trial_average_errors_control2), sem(trial_average_errors_control2), 'lineprops', {'Color', 'k'});
legend([h1.mainLine, h2.mainLine, h3.mainLine], {'Task', 'Control1', 'Control2'})
xlabel('trials')
ylabel('mean abs decoding error')


spatial_average_errors_task = cellfun(@(x) mean(abs(x), 1, 'omitmissing'), errors_task, 'UniformOutput', false);
spatial_average_errors_task = cat(1, spatial_average_errors_task{:});

spatial_average_errors_control = cellfun(@(x) mean(abs(x), 1, 'omitmissing'), errors_control, 'UniformOutput', false);
spatial_average_errors_control = cat(1, spatial_average_errors_control{:});

spatial_average_errors_control2 = cellfun(@(x) mean(abs(x), 1, 'omitmissing'), errors_control2, 'UniformOutput', false);
spatial_average_errors_control2 = cat(1, spatial_average_errors_control2{:});

figure
hold on
h1 = shadedErrorBar(1:50, mean(spatial_average_errors_task), sem(spatial_average_errors_task), 'lineprops', {'Color', 'r'});
h2 = shadedErrorBar(1:50, mean(spatial_average_errors_control), sem(spatial_average_errors_control), 'lineprops', {'Color', 'b'});
h3 = shadedErrorBar(1:50, mean(spatial_average_errors_control2), sem(spatial_average_errors_control2), 'lineprops', {'Color', 'k'});
legend([h1.mainLine, h2.mainLine, h3.mainLine], {'Task', 'Control1', 'Control2'})
xlabel('bins')
ylabel('mean abs decoding error')


%% Plotting Area-Specific Decoding Performance (Task Data)

areas = {'DMS', 'DLS', 'ACC'};
nAreas = length(areas);
n_animals = n_animals_task;  % number of task animals

% Scatter Plots for Each Area
for a = 1:nAreas
    figure;
    t = tiledlayout('flow');
    for ianimal = 1:n_animals
        % Check if area-specific decoding data exists for this animal
        if ~isempty(predicted_bins_area_task{a}{ianimal})
            % Determine the number of neurons for this animal and area
            switch areas{a}
                case 'DMS'
                    n_neurons = sum(task_data(ianimal).is_dms);
                case 'DLS'
                    n_neurons = sum(task_data(ianimal).is_dls);
                case 'ACC'
                    n_neurons = sum(task_data(ianimal).is_acc);
            end
            
            nexttile;
            scatter(predicted_bins_area_task{a}{ianimal}(:), actual_bins_area_task{a}{ianimal}(:), 10, 'filled');
            axis tight;
            axis square;
            title(sprintf('Animal %d (n = %d)', ianimal, n_neurons));
        end
    end
    xlabel(t, 'Predicted Bin');
    ylabel(t, 'True Bin');
    sgtitle(sprintf('Decoding Performance (Scatter) for %s (Task Data)', areas{a}));
end

% % Contour Plots for Each Area
% for a = 1:nAreas
%     figure;
%     t = tiledlayout('flow');
%     for ianimal = 1:n_animals
%         if ~isempty(predicted_bins_area_task{a}{ianimal})
%             nexttile;
%             % Create a grid for density estimation
%             gridx = 0:0.1:bins;
%             [X1,X2] = meshgrid(gridx, gridx);
%             xi = [X1(:), X2(:)];
%             % Data points: each row = [predicted, true]
%             x = [predicted_bins_area_task{a}{ianimal}(:), actual_bins_area_task{a}{ianimal}(:)];
%             % Compute and plot the kernel density estimate as contours
%             ksdensity(x, xi, 'PlotFcn', 'contour');
%             contourObj = findobj(gca, 'Type', 'Contour');
%             set(contourObj, 'LineWidth', 1.2);
%             axis tight;
%             axis square;
%             title(sprintf('Animal %d', ianimal));
%         end
%     end
%     xlabel(t, 'Predicted Bin');
%     ylabel(t, 'True Bin');
%     sgtitle(sprintf('Decoding Performance (Contour) for %s (Task Data)', areas{a}));
% end

avg_error_area = nan(n_animals_task, nAreas);

% Loop over areas and animals to compute the mean absolute decoding error
for a = 1:nAreas
    for ianimal = 1:n_animals_task
        if ~isempty(errors_area_task{a}{ianimal})
            err = errors_area_task{a}{ianimal};  % decoding error matrix: trials x bins
            % Compute average absolute error over all trials and bins for this animal and area
            avg_error_area(ianimal, a) = mean(abs(err(:)));
        end
    end
end

% Remove any rows (animals) that have NaNs in all areas (if any)
valid_animals = any(~isnan(avg_error_area), 2);
avg_error_area = avg_error_area(valid_animals, :);

% Plot average decoding error using your custom my_errorbar_plot function
figure;
my_errorbar_plot(avg_error_area, [color_dms; color_dls; color_acc]);
set(gca, 'XTick', 1:nAreas, 'XTickLabel', areas, 'FontSize', 12);
xlabel('Area');
ylabel('Mean Absolute Decoding Error');
title('Area-Specific Average Decoding Performance (Task Data)');
%% 
trial_errors = cell(nAreas,1);

for a = 1:nAreas
    animal_errors = {};  % will hold one vector per animal for the current area
    for ianimal = 1:n_animals_task
        if ~isempty(errors_area_task{a}{ianimal})
            % errors_area_task is a matrix (trials x bins) for this animal and area
            % Average across bins to get one error value per trial
            trial_err = mean(abs(errors_area_task{a}{ianimal}), 2);  % vector: [nTrials x 1]
            animal_errors{end+1} = trial_err;
        end
    end
    if ~isempty(animal_errors)
        % Determine the minimum number of trials available across animals for this area
        min_trials = min(cellfun(@length, animal_errors));
        % Truncate each animal's trial error vector to min_trials and store as a row
        errors_mat = cell2mat(cellfun(@(x) x(1:min_trials)', animal_errors, 'UniformOutput', false));
        % errors_mat now is [nAnimals x min_trials]
        trial_errors{a} = errors_mat;
    else
        trial_errors{a} = [];
    end
end

% Plot all Areas on a Single Figure
figure;
hold on;
colors = [color_dms;    % DMS: deep blue
          color_dls;  % DLS: forest green
          color_acc]; % ACC: crimson red


my_simple_errorbar_plot(trial_errors, colors);

xlabel('Trial Number');
ylabel('Mean Absolute Decoding Error');
title('Trial-By-Trial Decoding Error (Pooled across Animals) per Area');
legend(hPlots, areas, 'Location', 'Best');
hold off;
%% Trial-to-trial correlation

neuronal_stability_task = cat(1, avg_corrs_task{:});
neuronal_stability_control = cat(1, avg_corrs_control{:});
neuronal_stability_control2 = cat(1, avg_corrs_control2{:});

figure
hold on
h1 = shadedErrorBar(1:50, mean(neuronal_stability_task, 'omitmissing'), sem(neuronal_stability_task), 'lineprops', {'Color', 'r'});
h2 = shadedErrorBar(1:50, mean(neuronal_stability_control, 'omitmissing'), sem(neuronal_stability_control), 'lineprops', {'Color', 'b'});
h3 = shadedErrorBar(1:50, mean(neuronal_stability_control2, 'omitmissing'), sem(neuronal_stability_control2), 'lineprops', {'Color', 'k'});
legend([h1.mainLine, h2.mainLine, h3.mainLine], {'Task', 'Control1', 'Control2'})
xlabel('trials')
ylabel('trial-to-trial correlation')


neuronal_stability_task_dms = cat(1, avg_corrs_dms_task{:});
neuronal_stability_task_dls = cat(1, avg_corrs_dls_task{:});
neuronal_stability_task_acc = cat(1, avg_corrs_acc_task{:});

neuronal_stability_control_dms = cat(1, avg_corrs_dms_control{:});
neuronal_stability_control_dls = cat(1, avg_corrs_dls_control{:});
neuronal_stability_control_acc = cat(1, avg_corrs_acc_control{:});

neuronal_stability_control2_dms = cat(1, avg_corrs_dms_control2{:});
neuronal_stability_control2_dls = cat(1, avg_corrs_dls_control2{:});
neuronal_stability_control2_acc = cat(1, avg_corrs_acc_control2{:});

figure
t = tiledlayout(1, 3, "TileSpacing", "compact");
nexttile
hold on
h1 = shadedErrorBar(1:50, mean(neuronal_stability_task_dms, 'omitmissing'), sem(neuronal_stability_task_dms), 'lineprops', {'Color', color_dms});
h2 = shadedErrorBar(1:50, mean(neuronal_stability_task_dls, 'omitmissing'), sem(neuronal_stability_task_dls), 'lineprops', {'Color', color_dls});
h3 = shadedErrorBar(1:50, mean(neuronal_stability_task_acc, 'omitmissing'), sem(neuronal_stability_task_acc), 'lineprops', {'Color', color_acc});
legend([h1.mainLine, h2.mainLine, h3.mainLine], {'DMS', 'DLS', 'ACC'})
title('Task')
nexttile
hold on
h1 = shadedErrorBar(1:50, mean(neuronal_stability_control_dms, 'omitmissing'), sem(neuronal_stability_control_dms), 'lineprops', {'Color', color_dms});
h2 = shadedErrorBar(1:50, mean(neuronal_stability_control_dls, 'omitmissing'), sem(neuronal_stability_control_dls), 'lineprops', {'Color', color_dls});
h3 = shadedErrorBar(1:50, mean(neuronal_stability_control_acc, 'omitmissing'), sem(neuronal_stability_control_acc), 'lineprops', {'Color', color_acc});
legend([h1.mainLine, h2.mainLine, h3.mainLine], {'DMS', 'DLS', 'ACC'})
title('Control')
nexttile
hold on
h1 = shadedErrorBar(1:50, mean(neuronal_stability_control2_dms, 'omitmissing'), sem(neuronal_stability_control2_dms), 'lineprops', {'Color', color_dms});
h2 = shadedErrorBar(1:50, mean(neuronal_stability_control2_dls, 'omitmissing'), sem(neuronal_stability_control2_dls), 'lineprops', {'Color', color_dls});
h3 = shadedErrorBar(1:50, mean(neuronal_stability_control2_acc, 'omitmissing'), sem(neuronal_stability_control2_acc), 'lineprops', {'Color', color_acc});
legend([h1.mainLine, h2.mainLine, h3.mainLine], {'DMS', 'DLS', 'ACC'})
title('Control2')
xlabel(t, 'trials')
ylabel(t, 'trial-to-trial correlation')
linkaxes

%% Align Neural Metrics by Learning Point (Task Data) Using All Trials

nAnimals = numel(task_data);
aligned_decoding_error = nan(nAnimals, 3);
aligned_genvar           = nan(nAnimals, 3);
aligned_stability       = nan(nAnimals, 3);

learning_points = learning_point(has_learning_point);

for iAnimal = 1:nAnimals
    lp = learning_points{iAnimal};
    if isempty(lp)
        continue;
    end
    nTrialsAnimal = size(errors_task{iAnimal}, 1);

    % Decoding error: average absolute error across bins.
    error_animal = mean(abs(errors_task{iAnimal}), 2, 'omitnan');  % [nTrialsAnimal x 1]

    % Generalised variance: assume genvar_task is a matrix with each row corresponding to an animal.
    genvar_animal = genvar_task{iAnimal};

    % Neuronal stability: average trial-to-trial correlation across neurons.
    stability_animal = mean(avg_corrs_task{iAnimal}, 1, 'omitnan');  % [1 x nTrialsAnimal]

    % Define epochs relative to the learning point.
    idx_epoch1 = 1:min(3, nTrialsAnimal);             % First 3 trials.
    % start_pre = max(lp - 10, 1);
    start_pre = 4;
    % idx_epoch2 = start_pre:(lp - 1);
    idx_epoch2 = start_pre:(start_pre + 10 -1);
    idx_epoch3 = (lp + 1):min(lp + 10, nTrialsAnimal);   % 10 trials following lp.

    aligned_decoding_error(iAnimal, 1) = mean(error_animal(idx_epoch1), 'omitnan');
    aligned_decoding_error(iAnimal, 2) = mean(error_animal(idx_epoch2), 'omitnan');
    aligned_decoding_error(iAnimal, 3) = mean(error_animal(idx_epoch3), 'omitnan');

    aligned_genvar(iAnimal, 1) = mean(genvar_animal(idx_epoch1), 'omitnan');
    aligned_genvar(iAnimal, 2) = mean(genvar_animal(idx_epoch2), 'omitnan');
    aligned_genvar(iAnimal, 3) = mean(genvar_animal(idx_epoch3), 'omitnan');

    aligned_stability(iAnimal, 1) = mean(stability_animal(idx_epoch1), 'omitnan');
    aligned_stability(iAnimal, 2) = mean(stability_animal(idx_epoch2), 'omitnan');
    aligned_stability(iAnimal, 3) = mean(stability_animal(idx_epoch3), 'omitnan');
end

mean_error   = nanmean(aligned_decoding_error, 1);
sem_error    = nanstd(aligned_decoding_error, [], 1) ./ sqrt(sum(~isnan(aligned_decoding_error), 1));
mean_genvar  = nanmean(aligned_genvar, 1);
sem_genvar   = nanstd(aligned_genvar, [], 1) ./ sqrt(sum(~isnan(aligned_genvar), 1));
mean_stability = nanmean(aligned_stability, 1);
sem_stability  = nanstd(aligned_stability, [], 1) ./ sqrt(sum(~isnan(aligned_stability), 1));

epochs = {'First 3 Trials', 'Pre-learning (10 trials)', 'Post-learning (10 trials)'};

figure
my_errorbar_plot(aligned_decoding_error)
ylabel('Mean Abs Decoding Error');
title('Decoding Error');
set(gca, 'XTickLabel', epochs, 'FontSize', 12);

figure
my_errorbar_plot(aligned_genvar)
ylabel('Generalised Variance');
title('Generalised Variance');
set(gca, 'XTickLabel', epochs, 'FontSize', 12);

figure
my_errorbar_plot(aligned_stability)
ylabel('Trial-to-Trial Correlation');
title('Neuronal Stability');
set(gca, 'XTickLabel', epochs, 'FontSize', 12);


figure
subplot(2, 2, 1)
scatter(cellfun(@mean, trial_average_errors_task), [learning_points{:}])
lsline
subplot(2, 2, 2)
scatter(cellfun(@(x) mean(x, 'all', 'omitmissing'), avg_corrs_task), [learning_points{:}])
lsline

%% Decoding vs Behaviour (task)
zscored_lick_errors_animals = {task_data(:).zscored_lick_errors};

mov_window_size = 5;
figure
t = tiledlayout(n_animals_task, 3, 'TileSpacing', 'compact');
for ianimal = 1:n_animals_task
    ntrials_animal = size(trial_average_errors_task{ianimal}, 2);
    nexttile
    shadedErrorBar(1:ntrials_animal, movmean(trial_average_errors_task{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing'), movstd(trial_average_errors_task{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing')/sqrt(mov_window_size))
    title(num2str(ianimal))
    xline(learning_points{ianimal})
    xlabel('trials')
    ylabel('abs decoding error')
    axis tight
    ylim([5 20])

    nexttile
    shadedErrorBar(1:ntrials_animal, movmean(zscored_lick_errors_animals{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing'), movstd(zscored_lick_errors_animals{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing')/sqrt(mov_window_size))
    title(num2str(ianimal))
    xline(learning_points{ianimal})
    xlabel('trials')
    ylabel('lick error')
    axis tight
    ylim([-5 5])

    nexttile
    scatter(trial_average_errors_task{ianimal}(1:ntrials_animal), zscored_lick_errors_animals{ianimal}(1:ntrials_animal))
    title(num2str(ianimal))
    lsline
    [rho, pval] = corr(trial_average_errors_task{ianimal}(1:ntrials_animal)', zscored_lick_errors_animals{ianimal}(1:ntrials_animal)', "Rows","complete");
    legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
    xlabel('abs decoding error')
    ylabel('lick error')
    axis tight
end

delta_decoding = cellfun(@(x) (x(4) - x(1))/x(1), trial_average_errors_task, 'UniformOutput', true);

figure
scatter(delta_decoding', cell2mat(learning_points)')
lsline
[rho, pval] = corr(delta_decoding', cell2mat(learning_points)', "Rows","complete");
legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
xlabel('delta decoding error (trial 4 - trial 1)')
ylabel('learning point')


%% Stability vs Behaviour (task)
zscored_lick_errors_animals = {task_data(:).zscored_lick_errors};

mov_window_size = 5;
figure
t = tiledlayout(n_animals_task, 3, 'TileSpacing', 'compact');
for ianimal = 1:n_animals_task
    ntrials_animal = size(avg_corrs_task{ianimal}, 2);
    nexttile
    shadedErrorBar(1:ntrials_animal, movmean(squeeze(mean(avg_corrs_task{ianimal}(:, 1:ntrials_animal), 'omitmissing')), mov_window_size, 'omitmissing'), movstd(squeeze(mean(avg_corrs_task{ianimal}(:, 1:ntrials_animal), 'omitmissing')), mov_window_size, 'omitmissing')/sqrt(mov_window_size))
    title(num2str(ianimal))
    xline(learning_points{ianimal})
    xlabel('trials')
    ylabel('stability')
    axis tight

    nexttile
    shadedErrorBar(1:ntrials_animal, movmean(zscored_lick_errors_animals{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing'), movstd(zscored_lick_errors_animals{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing')/sqrt(mov_window_size))
    title(num2str(ianimal))
    xline(learning_points{ianimal})
    xlabel('trials')
    ylabel('lick error')
    axis tight
    ylim([-5 5])

    nexttile
    scatter(squeeze(mean(avg_corrs_task{ianimal}(:, 1:ntrials_animal), 'omitmissing')), zscored_lick_errors_animals{ianimal}(1:ntrials_animal))
    title(num2str(ianimal))
    lsline
    [rho, pval] = corr(squeeze(mean(avg_corrs_task{ianimal}(:, 1:ntrials_animal), 'omitmissing'))', zscored_lick_errors_animals{ianimal}(1:ntrials_animal)', "Rows","complete");
    legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
    xlabel('stability')
    ylabel('lick error')
    axis tight
end

delta_stability = cellfun(@(x) (squeeze(mean(x(:, 4), 'omitmissing')) - squeeze(mean(x(:, 1), 'omitmissing')))/squeeze(mean(x(:, 1), 'omitmissing')), avg_corrs_task, 'UniformOutput', true);

figure
scatter(delta_stability', cell2mat(learning_points)')
lsline
[rho, pval] = corr(delta_stability', cell2mat(learning_points)', "Rows","complete");
legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
xlabel('delta stability (trial 4 - trial 1)')
ylabel('learning point')

%% Genvar vs Behaviour (task)
zscored_lick_errors_animals = {task_data(:).zscored_lick_errors};

mov_window_size = 5;
figure
t = tiledlayout(n_animals_task, 3, 'TileSpacing', 'compact');
for ianimal = 1:n_animals_task
    ntrials_animal = size(genvar_task{ianimal}, 2);
    nexttile
    shadedErrorBar(1:ntrials_animal, movmean(genvar_task{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing'), movstd(genvar_task{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing')/sqrt(mov_window_size))
    title(num2str(ianimal))
    xline(learning_points{ianimal})
    xlabel('trials')
    ylabel('generalised variance')
    axis tight

    nexttile
    shadedErrorBar(1:ntrials_animal, movmean(zscored_lick_errors_animals{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing'), movstd(zscored_lick_errors_animals{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing')/sqrt(mov_window_size))
    title(num2str(ianimal))
    xline(learning_points{ianimal})
    xlabel('trials')
    ylabel('lick error')
    axis tight
    ylim([-5 5])

    nexttile
    scatter(genvar_task{ianimal}(1:ntrials_animal), zscored_lick_errors_animals{ianimal}(1:ntrials_animal))
    title(num2str(ianimal))
    lsline
    [rho, pval] = corr(genvar_task{ianimal}(1:ntrials_animal)', zscored_lick_errors_animals{ianimal}(1:ntrials_animal)', "Rows","complete");
    legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
    xlabel('gen var')
    ylabel('lick error')
    axis tight
end

delta_genvar = cellfun(@(x) (x(4) - x(1))/x(1), genvar_task, 'UniformOutput', true);

figure
scatter(delta_genvar', cell2mat(learning_points)')
lsline
[rho, pval] = corr(delta_genvar', cell2mat(learning_points)', "Rows","complete");
legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
xlabel('delta genvar (trial 4 - trial 1)')
ylabel('learning point')

%% Stability of behaviour (task)

avg_lick_corrs = cell(1, n_animals_task);
avg_occupancy_corrs = cell(1, n_animals_task);

for ianimal = 1:n_animals_task
    % n_trials = preprocessed_data(ianimal).n_trials;

    trials = task_data(ianimal).n_trials;
    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end

    lick_data = task_data(ianimal).spatial_binned_data.licks(1:change_point, :);
    occupancy_data = task_data(ianimal).spatial_binned_data.durations(1:change_point, :);

    window_size = 5;
    half_window = floor(window_size / 2);
    trials = change_point;

    avg_lick_corrs_animal = nan(1, trials);
    avg_occupancy_corrs_animal = nan(1, trials);

    for t = 1:trials
        % Define the window of trials centered on trial t
        trial_start = max(1, t - half_window);
        trial_end = min(trials, t + half_window);
        trials_in_window = trial_start:trial_end;
        num_trials_in_window = length(trials_in_window);

        % Proceed only if we have at least two trials to correlate
        if num_trials_in_window > 1
            % Extract the data for the current neuron and window
            lick_data_block = squeeze(lick_data(trials_in_window, :)');
            occ_data_block = squeeze(occupancy_data(trials_in_window, :)');

            % Compute the correlation matrices
            R_lick = corrcoef(lick_data_block);
            R_occ = corrcoef(occ_data_block);

            % Extract the upper triangle values
            upper_triangle_lick = triu(R_lick, 1);
            upper_vals_lick = upper_triangle_lick(upper_triangle_lick ~= 0);

            upper_triangle_occ = triu(R_occ, 1);
            upper_vals_occ = upper_triangle_occ(upper_triangle_occ ~= 0);

            % Compute the average correlation
            avg_lick_corrs_animal(t) = mean(upper_vals_lick);
            avg_occupancy_corrs_animal(t) = mean(upper_vals_occ);
        end
    end

    avg_lick_corrs{ianimal} = avg_lick_corrs_animal;
    avg_occupancy_corrs{ianimal} = avg_occupancy_corrs_animal;
end

% Concatenate data across animals
max_trials = max(cellfun(@length, avg_lick_corrs));
avg_lick_corrs_matrix = nan(n_animals, max_trials);
avg_occupancy_corrs_matrix = nan(n_animals, max_trials);

for ianimal = 1:n_animals_task
    n_trials = length(avg_lick_corrs{ianimal});
    avg_lick_corrs_matrix(ianimal, 1:n_trials) = avg_lick_corrs{ianimal};
    avg_occupancy_corrs_matrix(ianimal, 1:n_trials) = avg_occupancy_corrs{ianimal};
end


mov_window_size = 5;
figure
t = tiledlayout(n_animals_task, 3, 'TileSpacing', 'compact');
for ianimal = 1:n_animals_task
    ntrials_animal = size(avg_lick_corrs{ianimal}, 2);
    nexttile
    shadedErrorBar(1:ntrials_animal, movmean(avg_lick_corrs{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing'), movstd(avg_lick_corrs{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing')/sqrt(mov_window_size))
    title(num2str(ianimal))
    xline(learning_points{ianimal})
    xlabel('trials')
    ylabel('lick correlation')
    axis tight

    nexttile
    shadedErrorBar(1:ntrials_animal, movmean(zscored_lick_errors_animals{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing'), movstd(zscored_lick_errors_animals{ianimal}(1:ntrials_animal), mov_window_size, 'omitmissing')/sqrt(mov_window_size))
    title(num2str(ianimal))
    xline(learning_points{ianimal})
    xlabel('trials')
    ylabel('lick error')
    axis tight

    nexttile
    scatter(avg_lick_corrs{ianimal}(1:ntrials_animal), zscored_lick_errors_animals{ianimal}(1:ntrials_animal))
    title(num2str(ianimal))
    lsline
    [rho, pval] = corr(avg_lick_corrs{ianimal}(1:ntrials_animal)', zscored_lick_errors_animals{ianimal}(1:ntrials_animal)', "Rows","complete");
    legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
    xlabel('lick_correlation')
    ylabel('lick error')
    axis tight
end

delta_lickcorr = cellfun(@(x) (x(4) - x(1))/x(1), avg_lick_corrs, 'UniformOutput', true);

figure
scatter(delta_lickcorr', cell2mat(learning_points)')
lsline
[rho, pval] = corr(delta_lickcorr', cell2mat(learning_points)', "Rows","complete");
legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
xlabel('delta lickcorr (trial 4 - trial 1)')
ylabel('learning point')


%%
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals_task
    ntrials_animal = size(avg_lick_corrs{ianimal}, 2);

    nexttile
    scatter(avg_lick_corrs{ianimal}(1:ntrials_animal), squeeze(mean(avg_corrs_task{ianimal}(:, 1:ntrials_animal), 'omitmissing')))
    title(num2str(ianimal))
    lsline
    [rho, pval] = corr(avg_lick_corrs{ianimal}(1:ntrials_animal)', squeeze(mean(avg_corrs_task{ianimal}(:, 1:ntrials_animal), 'omitmissing'))', "Rows","complete");
    legend(sprintf('rho = %.2f - pval = %.3f', rho, pval))
    xlabel('lick correlation')
    ylabel('neuronal stability')
    axis tight

end

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_task
    nexttile
    plot(squeeze(mean(avg_corrs_task{ianimal}, 'omitmissing')))
    hold on
    yyaxis('right')
    plot(avg_lick_corrs{ianimal})
    xlim([1, 20])
end

%% Tuning (skewness)


window_size = 5;                   % total window width
half_window = floor(window_size/2);
p_value_threshold = 0.01;          % for shuffle-based significance
nShuffles = 20;                   % number of shuffles

% Preallocate cells to store final results for each animal
skewness_values_sliding   = cell(n_animals_task, 1);  % [neurons x nTrials]
is_tuned_skewness_sliding = cell(n_animals_task, 1);  % logical [neurons x nTrials]

shuffle_scores_sliding    = cell(n_animals_task, 1);  % [neurons x nTrials]
p_values_sliding          = cell(n_animals_task, 1);  % [neurons x nTrials]
is_tuned_shuffled_sliding = cell(n_animals_task, 1);  % logical [neurons x nTrials]

skew_threshold = 1;  % example threshold for skewness-based approach


for ianimal = 1:n_animals_task

    % Extract the firing rate matrix: [neurons x bins x trials]
    current_activity = task_data(ianimal).spatial_binned_fr_all;
    [neurons, bins, nTrials] = size(current_activity);

    % Prepare arrays to store results for each neuron at each trial
    skewness_vals_animal = NaN(neurons, nTrials);
    is_tuned_skew_animal = false(neurons, nTrials);

    for t = 1:nTrials
        % Define the window centred at trial t
        trial_start = max(1, t - half_window);
        trial_end   = min(nTrials, t + half_window);
        window_indices = trial_start:trial_end;

        for n = 1:neurons
            % Extract data for these trials: [bins x number_of_trials_in_window]
            data_block = squeeze(current_activity(n, :, window_indices));

            % Compute mean FR across the window for each bin
            meanFR_window = mean(data_block, 2, 'omitmissing');  % [bins x 1]

            % Compute skewness across bins
            skew_val = skewness(meanFR_window, 0, 1);
            skewness_vals_animal(n, t) = skew_val;

            % Check if skewness exceeds threshold
            if skew_val > skew_threshold
                is_tuned_skew_animal(n, t) = true;
            end
        end
    end

    % Assign outputs for this animal
    skewness_values_sliding{ianimal}   = skewness_vals_animal;
    is_tuned_skewness_sliding{ianimal} = is_tuned_skew_animal;
end

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_task

    current_activity = task_data(ianimal).spatial_binned_fr_all;
    [neurons, bins, trials] = size(current_activity);

    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end

    nexttile
    imagesc(skewness_values_sliding{ianimal}(:, 1:change_point))
end


%%

for ianimal = 1:n_animals_task


    current_activity = task_data(ianimal).spatial_binned_fr_all;
    [~, ~, trials] = size(current_activity);

    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end

    current_activity = current_activity(:, :, 1:change_point);

    [neurons, bins, nTrials] = size(current_activity);

    shuffle_scores_animal    = NaN(neurons, nTrials);
    p_vals_animal            = NaN(neurons, nTrials);
    tuned_shuffled_animal    = false(neurons, nTrials);

    for t = 1:nTrials
        % Define the window around trial t
        trial_start = max(1, t - half_window);
        trial_end   = min(nTrials, t + half_window);
        window_indices = trial_start:trial_end;
        current_window_size = length(window_indices);

        for n = 1:neurons
            % Data block for this neuron in this trial window
            data_block = squeeze(current_activity(n, :, window_indices));  % [bins x current_window_size]

            % Mean FR across these trials
            meanFR_window = mean(data_block, 2, 'omitmissing');  % [bins x 1]

            % Tuning score = difference between peak and trough
            actual_score = max(meanFR_window) - min(meanFR_window);
            shuffle_scores_animal(n, t) = actual_score;

            % Build a shuffle distribution by randomising the order of trials in 'data_block'
            shuffle_dist = NaN(nShuffles, 1);
            for s = 1:nShuffles
                shuf_idx = randperm(current_window_size);
                shuffled_data = data_block(:, shuf_idx);
                meanFR_shuffled = mean(shuffled_data, 2, 'omitmissing');

                shuffle_dist(s) = max(meanFR_shuffled) - min(meanFR_shuffled);
            end

            % p-value: fraction of shuffled scores >= the real one
            p_val = mean(shuffle_dist >= actual_score);
            p_vals_animal(n, t) = p_val;

            % Check significance
            if p_val < p_value_threshold
                tuned_shuffled_animal(n, t) = true;
            end
        end
    end

    % Assign outputs for this animal
    shuffle_scores_sliding{ianimal}    = shuffle_scores_animal;
    p_values_sliding{ianimal}         = p_vals_animal;
    is_tuned_shuffled_sliding{ianimal} = tuned_shuffled_animal;
end


figure
t = tiledlayout('flow');
for ianimal = 1:n_animals_task

    nexttile
    imagesc(-log(p_values_sliding{ianimal}))
end

%%

window_size = 10;                   % total window width
half_window = floor(window_size/2);

figure
l = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding','compact');
ianimal = 3;
current_activity = task_data(ianimal).spatial_binned_fr_all;
[~, ~, trials] = size(current_activity);

try
    change_point = min([task_data(ianimal).change_point_mean, trials]);
catch
    change_point = trials;
end

current_activity = current_activity(:, :, 1:change_point);

[neurons, bins, nTrials] = size(current_activity);
for t = 1:nTrials
    nexttile

    % Define the window around trial t
    trial_start = max(1, t - half_window);
    trial_end   = min(nTrials, t + half_window);
    window_indices = trial_start:trial_end;
    current_window_size = length(window_indices);

    for n = 13
        % Data block for this neuron in this trial window
        data_block = squeeze(current_activity(n, :, window_indices));  % [bins x current_window_size]

        % Mean FR across these trials
        meanFR_window = mean(data_block, 2, 'omitmissing');  % [bins x 1]
        semFR_window = sem(data_block, 2);

        shadedErrorBar(1:bins, meanFR_window, semFR_window)
    end
    title(num2str(t))
end
linkaxes

%% Coding dimension (lick error)

% Load or define current_activity (neurons x spatial bins x trials) and zscored_lick_errors (1 x trials)
for ianimal = 1:n_animals_task
    current_activity = task_data(ianimal).z_spatial_binned_fr_all;
    [~, ~, trials] = size(current_activity);

    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end

    current_activity = current_activity(:, :, 1:change_point);

    [neurons, bins, trials] = size(current_activity);

    zscored_lick_errors = task_data(ianimal).zscored_lick_errors(1:change_point);

    % Threshold to define high vs. low error trials (median split)
    high_error_trials = find(zscored_lick_errors > -1);
    low_error_trials = find(zscored_lick_errors <= -2);

    % --- 1. Compute Simple Mean-Subtraction Coding Direction ---
    % Reshape to neurons x trials (averaging across spatial bins)
    activity_avg = squeeze(mean(current_activity, 2, 'omitmissing'));

    % Compute mean response for high and low error trials
    mu_high = mean(activity_avg(:, high_error_trials), 2, 'omitmissing');
    mu_low  = mean(activity_avg(:, low_error_trials), 2, 'omitmissing');

    % Compute coding direction (difference vector)
    CD_simple = mu_high - mu_low;

    % Optionally, normalise the coding direction (unit vector)
    CD_simple = CD_simple / norm(CD_simple);

    % Project each trial’s activity onto the coding direction
    projections_simple = CD_simple' * activity_avg;  % 1 x trials

    figure;

    subplot(2,2,1);
    scatter(1:trials, projections_simple, 40, zscored_lick_errors, 'filled');
    xlabel('Trial');
    ylabel('Projection onto CD (au)');
    title('Simple Mean-Subtraction Projection');
    colorbar;
    set(gca, 'ytick', []);
    axis tight
    if ~isempty(learning_point{ianimal})
        xline(learning_point{ianimal})
    end

    % Projections across bins
    projections_bins_simple = zeros(bins, trials);

    for b = 1:bins
        % Extract activity for spatial bin b (neurons x trials)
        activity_bin = squeeze(current_activity(:, b, :));
        % Compute projection: dot product of CD_simple with activity at bin b.
        projections_bins_simple(b, :) = CD_simple' * activity_bin;
    end

    % Statistical testing at each bin: Compare high vs. low error groups
    p_values = zeros(bins, 1);
    for b = 1:bins
        data_high = projections_bins_simple(b, high_error_trials);
        data_low  = projections_bins_simple(b, low_error_trials);
        % Use ttest2 for an unpaired t-test between groups
        [~, p] = ttest2(data_high, data_low);
        p_values(b) = p;
    end

    num_tests = bins;  % number of spatial bins/tests
    p_values_corrected = p_values * num_tests;
    p_values_corrected(p_values_corrected > 1) = 1; % Ensure p-values do not exceed 1

    % Annotate significance using sigstar (available on the MATLAB File Exchange)
    % Prepare cell array for significant bin pairs and corresponding p-values.
    sig_pairs = {};
    sig_pvals = [];

    % Loop through each bin and check if the p-value is below threshold (e.g. 0.05)
    for b = 1:bins
        if p_values_corrected(b) < 0.05
            % The x-axis coordinates for the two groups in bin b
            % are (b - offset) for the high error group and (b + offset) for the low error group.
            sig_pairs{end+1} = [b, b];
            sig_pvals(end+1) = p_values_corrected(b);
        end
    end


    % Plotting the Projections Across Spatial Bins
    % For clarity, we can average the projection across high and low error trials separately.
    mean_proj_high_simple = mean(projections_bins_simple(:, high_error_trials), 2);
    mean_proj_low_simple  = mean(projections_bins_simple(:, low_error_trials), 2);

    sem_proj_high_simple = sem(projections_bins_simple(:, high_error_trials), 2);
    sem_proj_low_simple  = sem(projections_bins_simple(:, low_error_trials), 2);

    subplot(2,2,2);
    hold on
    h = shadedErrorBar(1:bins, mean_proj_high_simple, sem_proj_high_simple, 'lineprops', {'Color', 'r'});
    g = shadedErrorBar(1:bins, mean_proj_low_simple, sem_proj_low_simple, 'lineprops', {'Color', 'b'});
    xlabel('spatial bin');
    ylabel('CD projection (au)');

    title('Mean Projections across Spatial Bins');
    yticks([])
    if ~isempty(sig_pairs)
        % Example usage: sigstar({[1,3], [2,4]}, [0.01 0.04]);
        sigstar(sig_pairs, sig_pvals);
    end
    legend([h.mainLine, g.mainLine], {'High Error', 'Low Error'});

    % Alternatively, you can visualise the projection for each trial as a heatmap.
    subplot(2,2,3);
    imagesc(projections_bins_simple');
    ylabel('trial');
    xlabel('spatial bin');
    title('CD projection');
    if ~isempty(learning_point{ianimal})
        yline(learning_point{ianimal})
    end
    sgtitle(sprintf('animal %d', ianimal))

    subplot(2,2,4)
    shadedErrorBar(1:trials, movmean(zscored_lick_errors, 5, 'omitmissing'), movstd(zscored_lick_errors, 5, 'omitmissing')/sqrt(5));
    if ~isempty(learning_point{ianimal})
        xline(learning_point{ianimal})
    end
    xlabel('trial');
    ylabel('lick error');
    axis tight
end

%% Coding dimension (early vs late trials)
% Loop over animals
for ianimal = 1:n_animals_task
    % Load current activity (neurons x spatial bins x trials) and trial count
    current_activity = task_data(ianimal).spatial_binned_fr_all;
    [~, ~, trials] = size(current_activity);

    % Determine change point (if applicable)
    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end
    current_activity = current_activity(:, :, 1:change_point);
    [neurons, bins, trials] = size(current_activity);

    % Define early vs. late trials
    early_trial_n = 5;   % Set to 3 or 5 as desired
    early_trials = 1:early_trial_n;
    late_trials = (early_trial_n+1):trials;

    % --- 1. Compute Simple Mean-Subtraction Coding Direction ---
    % Average activity across spatial bins to obtain neurons x trials
    activity_avg = squeeze(mean(current_activity, 2, 'omitmissing'));

    % Compute the mean response for early and late trials
    mu_early = mean(activity_avg(:, early_trials), 2, 'omitmissing');
    mu_late  = mean(activity_avg(:, late_trials), 2, 'omitmissing');

    % Define the coding direction as the difference vector (early minus late)
    CD_simple = mu_early - mu_late;
    CD_simple = CD_simple / norm(CD_simple);

    % Project each trial's activity onto the coding direction
    projections_simple = CD_simple' * activity_avg;  % 1 x trials

    % Plot trial-by-trial projections
    figure;
    subplot(2,2,1);
    scatter(1:trials, projections_simple, 40, 'filled');
    xlabel('Trial');
    ylabel('Projection onto CD (au)');
    title('Simple Mean-Subtraction Projection');
    axis tight;
    if ~isempty(learning_point{ianimal})
        xline(learning_point{ianimal});
    end

    % --- 2. Projections across spatial bins ---
    % Compute projections for each spatial bin (neurons x trials)
    projections_bins_simple = zeros(bins, trials);
    for b = 1:bins
        activity_bin = squeeze(current_activity(:, b, :));  % neurons x trials for bin b
        projections_bins_simple(b, :) = CD_simple' * activity_bin;
    end

    % Statistical testing at each bin: compare early vs late trials using ttest2
    p_values = zeros(bins, 1);
    for b = 1:bins
        data_early = projections_bins_simple(b, early_trials);
        data_late  = projections_bins_simple(b, late_trials);
        [~, p] = ttest2(data_early, data_late);
        p_values(b) = p;
    end

    % Bonferroni correction for multiple comparisons
    num_tests = bins;
    p_values_corrected = p_values * num_tests;
    p_values_corrected(p_values_corrected > 1) = 1;  % cap p-values at 1

    % Prepare sigstar annotations (one pair per bin, using the bin number for both groups)
    sig_pairs = {};
    sig_pvals = [];
    for b = 1:bins
        if p_values_corrected(b) < 0.05
            sig_pairs{end+1} = [b, b];
            sig_pvals(end+1) = p_values_corrected(b);
        end
    end

    % --- 3. Visualisation ---
    % Compute mean projections and standard error across spatial bins for early and late groups
    mean_proj_early = mean(projections_bins_simple(:, early_trials), 2);
    mean_proj_late  = mean(projections_bins_simple(:, late_trials), 2);
    sem_proj_early = sem(projections_bins_simple(:, early_trials), 2);
    sem_proj_late  = sem(projections_bins_simple(:, late_trials), 2);

    % Plot the mean projections with shaded error bars for early (red) and late (blue) trials
    subplot(2,2,2);
    hold on;
    h = shadedErrorBar(1:bins, mean_proj_early, sem_proj_early, 'lineprops', {'Color', 'r'});
    g = shadedErrorBar(1:bins, mean_proj_late, sem_proj_late, 'lineprops', {'Color', 'b'});
    xlabel('Spatial Bin');
    ylabel('CD projection (au)');
    title('Mean Projections across Spatial Bins');
    yticks([]);
    if ~isempty(sig_pairs)
        sigstar(sig_pairs, sig_pvals);
    end
    legend([h.mainLine, g.mainLine], {'Early', 'Late'});

    % Visualise the trial-by-trial projections across spatial bins as a heatmap
    subplot(2,2,3);
    imagesc(projections_bins_simple');
    ylabel('Trial');
    xlabel('Spatial Bin');
    title('CD projection (heatmap)');
    if ~isempty(learning_point{ianimal})
        yline(learning_point{ianimal});
    end

    sgtitle(sprintf('Animal %d (Early vs Late Trials)', ianimal));
end

%% Per-Area Coding Dimension Plots (Defined by Lick Error)
% Loop over animals
for ianimal = 1:n_animals_task
    % Load the z-scored spatial binned firing rates and lick errors
    current_activity = task_data(ianimal).z_spatial_binned_fr_all;  % neurons x bins x trials
    [neurons, bins, trials] = size(current_activity);
    
    % Limit to change_point trials (if defined)
    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end
    current_activity = current_activity(:, :, 1:change_point);
    zscored_lick_errors = task_data(ianimal).zscored_lick_errors(1:change_point);
    
    % Define trial groups based on lick error thresholds
    high_error_trials = find(zscored_lick_errors > -1);
    low_error_trials  = find(zscored_lick_errors <= -2);
    
    % Get area flags and names
    areas = {'DMS', 'DLS', 'ACC'};
    area_flags = { task_data(ianimal).is_dms, ...
                   task_data(ianimal).is_dls, ...
                   task_data(ianimal).is_acc };
    
    % Create a new figure with a tiled layout (2 rows x 3 columns)
    figure;
    t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Animal %d: Coding Dimension by Area (Lick Error)', ianimal));
    
    % Loop over areas (columns)
    for a = 1:length(areas)
        % Extract activity for the current area
        area_activity = current_activity(area_flags{a}, :, :);  % [nAreaNeurons x bins x trials]
        if isempty(area_activity)
            continue;
        end
        
        % Average across spatial bins to get neurons x trials
        activity_avg = squeeze(mean(area_activity, 2, 'omitmissing'));  % [nAreaNeurons x trials]
        
        % Compute mean responses for high and low error trials
        mu_high = mean(activity_avg(:, high_error_trials), 2, 'omitmissing');
        mu_low  = mean(activity_avg(:, low_error_trials), 2, 'omitmissing');
        
        % Compute the simple mean-subtraction coding direction
        CD = mu_high - mu_low;
        % Normalize the coding direction (unit vector)
        CD = CD / norm(CD);
        
        % Project each trial's activity onto the coding direction (trial progression)
        projections = CD' * activity_avg;  % 1 x trials

        % --- Compute d′ for the projections ---
        % Use the trial projections for high and low error groups
        proj_high = projections(high_error_trials);
        proj_low  = projections(low_error_trials);
        mu_proj_high = mean(proj_high);
        mu_proj_low  = mean(proj_low);
        sigma_proj_high = std(proj_high);
        sigma_proj_low  = std(proj_low);
        d_prime(ianimal, a) = (mu_proj_high - mu_proj_low) / sqrt(0.5*(sigma_proj_high^2 + sigma_proj_low^2));
        
        
        % ------ Top Row: Trial Progression ------
        nexttile(t, a);
        % Scatter plot: projection vs. trial, colored by lick error
        scatter(1:change_point, projections, 40, zscored_lick_errors, 'filled');
        hold on;
        % Plot moving average for better trend visualization
        plot(movmean(projections, 5, 'omitnan'), 'LineWidth', 1.5);
        if ~isempty(learning_point{ianimal})
            xline(learning_point{ianimal}, '--k', 'LineWidth', 1.5);
        end
        xlabel('Trial');
        ylabel('Projection (au)');
        title(sprintf('%s: Trial Progression, d'' = %.2f', areas{a}, d_prime(ianimal, a)));
        colorbar;
        axis tight;
        set(gca, 'ytick', []);
        
        % ------ Bottom Row: Spatial Projections ------
        % Compute projections per spatial bin:
        projections_bins = zeros(bins, change_point);
        for b = 1:bins
            activity_bin = squeeze(area_activity(:, b, :));  % [nAreaNeurons x trials]
            projections_bins(b, :) = CD' * activity_bin;
        end
        
        % Statistical testing: ttest2 between high and low error trials at each bin
        p_values = zeros(bins, 1);
        for b = 1:bins
            data_high = projections_bins(b, high_error_trials);
            data_low  = projections_bins(b, low_error_trials);
            [~, p] = ttest2(data_high, data_low);
            p_values(b) = p;
        end
        
        % Bonferroni correction
        num_tests = bins;
        p_corr = p_values * num_tests;
        p_corr(p_corr > 1) = 1;
        
        % Identify bins with significant differences for sigstar annotation
        sig_pairs = {};
        sig_pvals = [];
        for b = 1:bins
            if p_corr(b) < 0.05
                sig_pairs{end+1} = [b, b];
                sig_pvals(end+1) = p_corr(b);
            end
        end
        
        % Average projections across spatial bins for each group
        mean_proj_high = mean(projections_bins(:, high_error_trials), 2);
        mean_proj_low  = mean(projections_bins(:, low_error_trials), 2);
        
        % Compute standard errors
        sem_proj_high = sem(projections_bins(:, high_error_trials), 2);
        sem_proj_low  = sem(projections_bins(:, low_error_trials), 2);
        
        nexttile(t, a+3);
        hold on;
        hLine = shadedErrorBar(1:bins, mean_proj_high, sem_proj_high, 'lineprops', {'Color', 'r'});
        lLine = shadedErrorBar(1:bins, mean_proj_low, sem_proj_low, 'lineprops', {'Color', 'b'});
        xlabel('Spatial Bin');
        ylabel('Projection (au)');
        title(sprintf('%s: Spatial Pattern', areas{a}));
        set(gca, 'ytick', []);
        if ~isempty(sig_pairs)
            sigstar(sig_pairs, sig_pvals);
        end
        legend([hLine.mainLine, lLine.mainLine], {'High Error', 'Low Error'});
        axis tight;
    end
    
    % Optionally, add an extra tile showing the lick error progression
    nexttile(t, 2*length(areas) + 1);
    shadedErrorBar(1:change_point, movmean(zscored_lick_errors, 5, 'omitnan'), ...
                   movstd(zscored_lick_errors, 5, 'omitnan')/sqrt(5));
    if ~isempty(learning_point{ianimal})
        xline(learning_point{ianimal}, '--k', 'LineWidth', 1.5);
    end
    xlabel('Trial');
    ylabel('Lick Error');
    title('Lick Error Progression');
    axis tight;
end

figure
t = tiledlayout('flow');
for iarea = 1:numel(areas)
    nexttile
    scatter(d_prime(:, iarea), [learning_point{:}], 75, 'filled', 'markeredgecolor', 'w')
    lsline
    title(areas{iarea})
end

%% Similarity Analysis by Area (Defined by Lick Error using Learning-Point Prototype)
% This code assumes that for each animal we have:
% - task_data(ianimal).z_spatial_binned_fr_all: neurons x bins x trials
% - task_data(ianimal).zscored_lick_errors: 1 x trials
% - task_data(ianimal).is_dms, is_dls, is_acc: logical indices for neurons in each area
% - task_data(ianimal).change_point_mean: maximum trial index to consider (if defined)
% - learning_point{ianimal}: a trial index marking the learning point (if defined)
%
% For each animal and each area (if >4 neurons), we:
% 1. Extract the area-specific activity.
% 2. Reshape the data (each trial yields a trajectory across spatial bins).
% 3. Perform PCA on the concatenated data to obtain 3 principal components.
% 4. Reconstruct a [bins x trials x 3] representation.
% 5. Define the prototype trajectory as the mean across the 10 trials immediately following the learning point.
% 6. Compute cosine similarity at each spatial bin between each trial and the prototype,
%    aggregating the similarity (using median across bins) for each trial.
% 7. Group trials into three groups: first 3, trials 4-13, and “experienced” (the 10 following learning).
% 8. Plot (a) a bar plot of the group-averaged similarity (n = animals) per area with sigstar comparisons, and
%    (b) a scatter plot of similarity versus z-scored lick error with the correlation (r) and p value in the title.

% Setup
areas = {'DMS', 'DLS', 'ACC'};  % Define area names (assumed consistent)
nAnimals = n_animals_task;  
nAreas   = numel(areas);

% Preallocate storage for trial-wise similarity and group averages.
similarity_all = cell(nAnimals, nAreas); % each cell: vector (1 x trials) for similarity
lick_error_all = cell(nAnimals, nAreas);   % corresponding lick error per trial

% For group averages: group_sim(ianimal, area, group)
% Groups: 1 = first 3 trials, 2 = trials 4-13, 3 = experienced (first 10 after learning)
group_sim = nan(nAnimals, nAreas, 3);

% Loop over animals
for ianimal = 1:nAnimals
    % Load animal data
    current_activity = task_data(ianimal).spatial_binned_fr_all;  % neurons x bins x trials
    [~, bins, trials] = size(current_activity);
    
    % Limit to change_point trials if defined; otherwise, use all trials.
    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end
    current_activity = current_activity(:, :, 1:change_point);
    zscored_lick_errors = task_data(ianimal).zscored_lick_errors(1:change_point);
    % Remove extreme values (if any)
    zscored_lick_errors(zscored_lick_errors > 5) = nan;
    
    % Define trial groups
    group1_trials = 1:min(3, change_point);
    group2_trials = 4:min(13, change_point);
    % Define experienced trials: first 10 trials following the learning point
    if isempty(learning_point{ianimal}) || (learning_point{ianimal} + 10 > change_point)
        % Skip this animal if a valid learning point is not defined or there are not enough trials.
        continue;
    else
        experienced_trials = (learning_point{ianimal}+1):(learning_point{ianimal}+10);
    end
    
    % Area flags for this animal (logical indices per neuron)
    area_flags = { task_data(ianimal).is_dms, ...
                   task_data(ianimal).is_dls, ...
                   task_data(ianimal).is_acc };
    
    % Loop over areas
    for a = 1:nAreas
        % Extract area-specific activity: [nAreaNeurons x bins x trials]
        area_activity = current_activity(area_flags{a}, :, :);
        % Skip if no units or not enough units (<8)
        if isempty(area_activity) || (size(area_activity,1) < 5)
            continue;
        end
        nAreaNeurons = size(area_activity, 1);
        
        % --- PCA: Obtain 3 principal components ---
        % Concatenate each trial's data (each trial: [bins x nAreaNeurons])
        data_all = [];
        for t = 1:change_point
            trial_data = squeeze(area_activity(:, :, t))';  % [bins x nAreaNeurons]
            data_all = [data_all; trial_data];  % concatenate along bins dimension
        end
        num_components = 3;
        % Run PCA on concatenated data with 3 components.
        [coeff, score, ~, ~, ~, mu] = pca(data_all, 'NumComponents', num_components);
        % Reconstruct trial-by-trial reduced trajectories: [bins x trials x num_components]
        reduced_data = nan(bins, change_point, num_components);
        for t = 1:change_point
            idx_start = (t-1)*bins + 1;
            idx_end   = t*bins;
            reduced_data(:, t, :) = score(idx_start:idx_end, :);
        end
        
        % --- Define Prototype Trajectory ---
        % Use only the 10 trials following the learning point.
        prototype_trials = experienced_trials;
        % Compute the prototype trajectory as the mean across these trials (for each bin).
        % Result: [bins x 3]
        prototype_traj = squeeze(mean(reduced_data(:, prototype_trials, :), 2, 'omitnan'));
        
        % --- Compute Cosine Similarity per Trial ---
        % For each trial, compute the cosine similarity between its reduced trajectory and the prototype,
        % at each spatial bin and then take the median similarity across bins.
        trial_similarity = nan(change_point, 1);
        for t = 1:change_point
            trial_traj = squeeze(reduced_data(:, t, :));  % [bins x num_components]
            valid_bins = all(~isnan(trial_traj),2) & all(~isnan(prototype_traj),2);
            if sum(valid_bins)==0
                continue;
            end
            cos_sim_bins = nan(sum(valid_bins),1);
            valid_idx = find(valid_bins);
            for b = 1:length(valid_idx)
                bin_idx = valid_idx(b);
                vec_trial = trial_traj(bin_idx, :);
                vec_proto = prototype_traj(bin_idx, :);
                if norm(vec_trial)==0 || norm(vec_proto)==0
                    cos_sim_bins(b) = NaN;
                else
                    cos_sim_bins(b) = dot(vec_trial, vec_proto) / (norm(vec_trial)*norm(vec_proto));
                end
            end
            trial_similarity(t) = median(cos_sim_bins, 'omitnan');
        end
        
        % Store trial-wise similarity and corresponding lick errors (for scatter plots)
        similarity_all{ianimal, a} = trial_similarity;
        lick_error_all{ianimal, a} = zscored_lick_errors;
        
        % --- Compute Group Averages for Bar Plot ---
        group_sim(ianimal, a, 1) = mean(trial_similarity(group1_trials), 'omitnan');
        group_sim(ianimal, a, 2) = mean(trial_similarity(group2_trials), 'omitnan');
        group_sim(ianimal, a, 3) = mean(trial_similarity(experienced_trials), 'omitnan');
    end % end for area
end % end for animal

% Plot 1: Bar Plot of Average Similarity by Trial Group (Across Animals) per Area with sigstar
figure;
group_names = {'First 3', 'Trials 4-13', 'Experienced'};
nGroups = numel(group_names);
for a = 1:nAreas
    subplot(1, nAreas, a);
    % Extract data for current area (animals x groups)
    data = squeeze(group_sim(:, a, :));
    % Remove rows with all NaNs (animals skipped for this area)
    data = data(~all(isnan(data),2), :);
    if isempty(data)
        title(sprintf('%s: Insufficient data', areas{a}));
        continue;
    end
   
    my_simple_errorbar_plot(data)
    % errorbar(1:nGroups, means, sems, '.k','LineWidth',1.5);
    set(gca, 'XTick', 1:nGroups, 'XTickLabel', group_names);
    xlabel('Trial Group');
    ylabel('Cosine Similarity');
    title(sprintf('%s', areas{a}));
    
    % Compute pairwise comparisons using paired t-tests and annotate using sigstar.
    pairs = {};
    p_vals = [];
    % Compare Group 1 vs. Group 2
    if all(~isnan(data(:,1))) && all(~isnan(data(:,2)))
        [~, p12] = ttest(data(:,1), data(:,2));
        pairs{end+1} = [1,2];
        p_vals(end+1) = p12;
    end
    % Compare Group 1 vs. Group 3
    if all(~isnan(data(:,1))) && all(~isnan(data(:,3)))
        [~, p13] = ttest(data(:,1), data(:,3));
        pairs{end+1} = [1,3];
        p_vals(end+1) = p13;
    end
    % Compare Group 2 vs. Group 3
    if all(~isnan(data(:,2))) && all(~isnan(data(:,3)))
        [~, p23] = ttest(data(:,2), data(:,3));
        pairs{end+1} = [2,3];
        p_vals(end+1) = p23;
    end
    if ~isempty(pairs)
        sigstar(pairs, p_vals);
    end
    hold off;
end
linkaxes

% Plot 2: Scatter Plot of Similarity vs. Z-Scored Lick Error per Area with Correlation Stats
% Arrange the scatter plots in a tiled layout: one row per area, one column per animal.
figure;
t = tiledlayout(nAreas, nAnimals, 'TileSpacing', 'compact', 'Padding', 'compact');
for a = 1:nAreas
    for ianimal = 1:nAnimals
        nexttile((a-1)*nAnimals + ianimal);
        if isempty(similarity_all{ianimal, a})
            axis off;
            continue;
        end
        x = lick_error_all{ianimal, a};
        y = similarity_all{ianimal, a};
        scatter(x, y, 50, 'filled', 'MarkerEdgeColor', 'w');
        % lsline;
        
        % Compute robust regression using robustfit:
        [b, stats] = robustfit(x, y);  % b(1)=intercept, b(2)=slope
        yfit = b(1) + b(2)*x;
        
        % Compute R^2: 1 - (SS_res / SS_tot)
        SS_res = sum((y - yfit').^2, 'omitmissing');
        SS_tot = sum((y - mean(y)).^2, 'omitmissing');
        r2 = 1 - SS_res/SS_tot;
        
        hold on;
        plot(x, yfit, 'r-', 'LineWidth', 1.5);
        
        % Display the R^2 value in the title
        title(sprintf('Animal%d, %s: R^2 = %.2f', ianimal, areas{a}, r2), 'FontSize',8);
        
    end

end
xlabel(t, 'Z-scored Lick Error'); 
ylabel(t, 'Cosine Similarity');

%% Per-Area Coding Dimension Analysis Relative to Disengagement (change_point) in One Figure per Animal
% Define epoch window sizes
pre_window = 10;   % number of trials before the disengagement
post_window = 10;  % number of trials after (including) the disengagement

% Loop over animals
for ianimal = 1:n_animals_task
    % Load current activity (assumed z-scored) and get dimensions
    current_activity = task_data(ianimal).z_spatial_binned_fr_all;  % neurons x bins x trials
    [nNeurons, bins, total_trials] = size(current_activity);
    
    % Determine disengagement point (change_point)
    try
        disengagement_point = min([task_data(ianimal).change_point_mean, total_trials]);
    catch
        disengagement_point = total_trials;
    end
    current_activity = current_activity(:, :, 1:total_trials);
    
    % Define trial epochs relative to disengagement
    if disengagement_point > pre_window
        trials_pre = (disengagement_point - pre_window):(disengagement_point - 1);
    else
        trials_pre = 1:(disengagement_point - 1);
    end
    trials_post = disengagement_point:min(disengagement_point + post_window - 1, total_trials);
    
    % Get area flags and names
    areas = {'DMS', 'DLS', 'ACC'};
    area_flags = { task_data(ianimal).is_dms, ...
                   task_data(ianimal).is_dls, ...
                   task_data(ianimal).is_acc };
    
    % Create one figure per animal with a tiled layout (2 rows x 3 columns)
    figure;
    t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Animal %d: CD Projection Relative to Disengagement', ianimal));
    
    % Loop over each area (each column corresponds to one area)
    for a = 1:length(areas)
        % Extract activity for neurons in the current area
        area_activity = current_activity(area_flags{a}, :, :);  % [nAreaNeurons x bins x trials]
        if isempty(area_activity)
            % If no neurons in this area, skip to the next column
            continue;
        end
        
        % Average across spatial bins -> neurons x trials
        activity_avg = squeeze(mean(area_activity, 2, 'omitmissing'));
        
        % Compute mean responses for the two epochs (pre and post disengagement)
        mu_pre  = mean(activity_avg(:, trials_pre), 2, 'omitmissing');
        mu_post = mean(activity_avg(:, trials_post), 2, 'omitmissing');
        
        % Define the coding direction as the difference (post - pre) and normalize it
        CD = mu_post - mu_pre;
        CD = CD / norm(CD);
        
        % Project each trial's activity onto the coding direction
        projections = CD' * activity_avg;  % 1 x total_trials
        
        % ---- Top Tile: Trial-by-Trial Projection ----
        nexttile(t, a);
        scatter(1:total_trials, projections, 75, 'filled', 'MarkerFaceColor', colors(a, :), 'markerEdgeColor', 'w');
        
        hold on;
        plot(movmean(projections, 5, 'omitnan'), 'LineWidth', 1.5, 'Color', 'k');
        xline(disengagement_point, '--k', 'LineWidth', 1.5);
        xlabel('Trial');
        ylabel('Projection (au)');
        title(sprintf('%s: Trial Projections', areas{a}));
        axis tight;
        
    end
end