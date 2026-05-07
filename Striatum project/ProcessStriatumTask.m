%% Run preprocessing analysis
clearvars -except all_data
clc
% --- Constants ---
reward_zone_start_cm = 125; 
visual_zone_start_au = 80;
reward_zone_start_au = 100;
reward_zone_end_au = 135;
corridor_end_au = 200;
bin_size = 4; 
bin_edges = 0:bin_size:corridor_end_au;
bin_edges(end) = corridor_end_au + bin_size;
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres);
visual_zone_start_bins = visual_zone_start_au/bin_size;
reward_zone_start_bins = reward_zone_start_au / bin_size;
reward_zone_end_bins = reward_zone_end_au / bin_size;

% --- Load Data ---
if exist('preprocessed_data.mat', 'file')
    fprintf('Loading existing preprocessed data...\n');
    load('preprocessed_data.mat', 'preprocessed_data');
    n_animals = numel(preprocessed_data);
else
    if ~exist('all_data', 'var')
        load('all_data.mat');
    end
    fr_threshold = 0.02; % Hz
    n_animals = numel(all_data);
    
    % --- Step 1: Filter Low Firing Neurons (Robustly) ---
    fprintf('Filtering low-firing neurons...\n');
    for ianimal = 1:n_animals
        keep_neurons = all_data(ianimal).avg_fr_all >= fr_threshold;
        all_data(ianimal).final_spikes = all_data(ianimal).final_spikes(keep_neurons, :);
        all_data(ianimal).final_areas = all_data(ianimal).final_areas(keep_neurons);
        all_data(ianimal).avg_fr_all = all_data(ianimal).avg_fr_all(keep_neurons);
        % ROBUSTNESS: Check if neuron types exist before indexing
        if isfield(all_data(ianimal), 'final_neurontypes') && ~isempty(all_data(ianimal).final_neurontypes)
            all_data(ianimal).final_neurontypes = all_data(ianimal).final_neurontypes(keep_neurons, :);
        else
            % Create dummy NaNs if missing so downstream code doesn't break
            n_kept = sum(keep_neurons);
            all_data(ianimal).final_neurontypes = nan(n_kept, 1); 
        end
    end
    fprintf('Processing data for all animals...\n');
    
    % PREALLOCATION: Initialize struct array to avoid dynamic growing
    preprocessed_data(n_animals) = struct();
    for ianimal = 1:n_animals
        fprintf('Processing animal %d/%d (ID: %d)...\n', ianimal, n_animals, all_data(ianimal).mouseid);
        
        % 1. Cut data and align indices
        trialData = cut_data_per_trial(all_data, ianimal);
        all_data = reorganize_spikes_by_area(all_data, ianimal);
        n_npx_datapoints = length(all_data(ianimal).npx_time);
        npxStartIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialData.trialStartTimes_vr, 'nearest', 'extrap');
        npxEndIdx   = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialData.trialEndTimes_vr, 'nearest', 'extrap');
        
        % 2. Extract Trial Data
        [binned_spikes_trials, npx_times_trials] = extract_binned_spikes(all_data, ianimal, npxStartIdx, npxEndIdx);
        trial_metrics = compute_trial_metrics(trialData);
        
        mov_window_size = 5;
        change_point_mean = find_change_points(trialData.trialDurations_vr, trial_metrics, mov_window_size);
        n_trials = trialData.n_trials - 1;
        
        % 3. Separate Periods & Filter Bad Trials
        [darkData, corridorData] = separate_dark_and_corridor_periods(trialData, binned_spikes_trials, npx_times_trials);
        
        % Robust check for empty rewards
        trials_to_exclude = cellfun(@isempty, corridorData.trial_reward);
        goodTrials = ~trials_to_exclude;
        
        % Apply filtering manually to ensure structure consistency
        fNames = fieldnames(trialData);
        for i = 1:numel(fNames)
            if numel(trialData.(fNames{i})) == trialData.n_trials
                 if iscell(trialData.(fNames{i})) || isnumeric(trialData.(fNames{i}))
                    trialData.(fNames{i}) = trialData.(fNames{i})(goodTrials);
                 end
            end
        end
        
        npxStartIdx = npxStartIdx(goodTrials);
        npxEndIdx   = npxEndIdx(goodTrials);
        n_trials = sum(goodTrials);
        
        % 4. Lick Analysis
        trial_lick_positions = cellfun(@(x,y) x(logical(y)), corridorData.trial_position, corridorData.trial_licks, 'UniformOutput', false);
        
        trial_lick_errors = nan(1, n_trials);
        shuffled_lick_error_means = nan(1, n_trials);
        shuffled_lick_error_stds = nan(1, n_trials);
        
        for itrial = 1:n_trials
            try
                [trial_lick_errors(itrial), shuffled_lick_error_means(itrial), shuffled_lick_error_stds(itrial), ~] = ...
                    calculate_lick_precision(trial_lick_positions{itrial}, reward_zone_start_au);
            catch
                % Keep NaNs
            end
        end
        
        % Lick fraction calculation
        calc_frac = @(x) (sum((x > reward_zone_start_au - 20) & (x < reward_zone_start_au)) + 1) / (sum((x > 0) & (x < reward_zone_start_au)) + 1);
        trial_lick_fractions = cellfun(calc_frac, trial_lick_positions);
        
        % 5. Neural Data Re-binning
        is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
        is_dls = strcmp(all_data(ianimal).final_areas, 'DLS');
        is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');
        is_v1  = strcmp(all_data(ianimal).final_areas, 'V1'); % NEW: V1 logical
        
        % Helper for slicing spikes
        slice_spikes = @(spikes) arrayfun(@(s,e) spikes(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
        
        binned_spikes_trials_dms = slice_spikes(all_data(ianimal).final_spikes(is_dms, :));
        binned_spikes_trials_dls = slice_spikes(all_data(ianimal).final_spikes(is_dls, :));
        binned_spikes_trials_acc = slice_spikes(all_data(ianimal).final_spikes(is_acc, :));
        
        [trial_average_fr_dms, trial_sem_fr_dms] = compute_firing_rates(binned_spikes_trials_dms, trialData.trialDurations_vr);
        [trial_average_fr_dls, trial_sem_fr_dls] = compute_firing_rates(binned_spikes_trials_dls, trialData.trialDurations_vr);
        [trial_average_fr_acc, trial_sem_fr_acc] = compute_firing_rates(binned_spikes_trials_acc, trialData.trialDurations_vr);
        
        % NEW: V1 Firing Rates (safely wrap in case mouse has no V1)
        binned_spikes_trials_v1 = slice_spikes(all_data(ianimal).final_spikes(is_v1, :));
        if any(is_v1)
            [trial_average_fr_v1, trial_sem_fr_v1] = compute_firing_rates(binned_spikes_trials_v1, trialData.trialDurations_vr);
        else
            trial_average_fr_v1 = []; trial_sem_fr_v1 = [];
        end

        % Outlier Handling
        outlier_trials = isoutlier(trial_lick_errors, "percentiles", [0, 99]);
        trial_lick_errors(outlier_trials) = nan; 
        trial_lick_errors(1) = nan; % Explicitly remove first trial
        
        shuffled_lick_error_means(outlier_trials) = nan; shuffled_lick_error_means(1) = nan;
        zscored_lick_errors = (trial_lick_errors - shuffled_lick_error_means) ./ shuffled_lick_error_stds;
        
        % 6. Spatial Binning & Dark Data
        spatial_binned_data = spatial_binning(corridorData, bin_edges, num_bins);
        
        % Temporal Binning (Dark)
        n_units = size(all_data(ianimal).final_spikes, 1);
        temp_bin_edges = 1:100:5001;
        num_temp_bins = numel(temp_bin_edges) - 1;
        temp_binned_dark_spikes = nan(n_units, num_temp_bins, n_trials);
        for itrial = 1:n_trials
            [~, ~, bin_idx] = histcounts(1:length(darkData.binned_spikes{itrial}), temp_bin_edges);
            for ibin = 1:num_temp_bins
                idx_in_bin = (bin_idx == ibin);
                if any(idx_in_bin)
                    temp_binned_dark_spikes(:, ibin, itrial) = sum(darkData.binned_spikes{itrial}(:, idx_in_bin), 2);
                end
            end
        end
        temp_binned_dark_fr = temp_binned_dark_spikes / 0.1; % 100ms = 0.1s
        z_temp_binned_dark_fr = zscore(temp_binned_dark_fr, [], [2, 3]);
        
        % 7. Data Preparation for Analysis
        spatial_binned_fr_all = cat(3, spatial_binned_data.firing_rates{1:n_trials});
        if size(spatial_binned_fr_all, 3) > n_trials
            spatial_binned_fr_all = spatial_binned_fr_all(:, :, 1:n_trials);
        end
        z_spatial_binned_fr_all = nan_zscore(spatial_binned_fr_all, [2, 3]);
        
        % --- OPTIMIZED: Vectorized Correlations ---
        DMS_data = spatial_binned_fr_all(is_dms, :, :); 
        DLS_data = spatial_binned_fr_all(is_dls, :, :);
        ACC_data = spatial_binned_fr_all(is_acc, :, :);
        V1_data  = spatial_binned_fr_all(is_v1, :, :); % NEW
        
        n_neurons_DMS = sum(is_dms); n_neurons_DLS = sum(is_dls); 
        n_neurons_ACC = sum(is_acc); n_neurons_V1  = sum(is_v1);
        
        all_cross_area_correlations_DMSACC = nan(n_trials, n_neurons_DMS, n_neurons_ACC);
        all_cross_area_correlations_DMSDLS = nan(n_trials, n_neurons_DMS, n_neurons_DLS);
        all_cross_area_correlations_V1ACC  = nan(n_trials, n_neurons_V1, n_neurons_ACC); % NEW
        all_cross_area_correlations_V1DMS  = nan(n_trials, n_neurons_V1, n_neurons_DMS); % NEW
        
        for itrial = 1:n_trials
            DMS_trial = squeeze(DMS_data(:, :, itrial))'; 
            ACC_trial = squeeze(ACC_data(:, :, itrial))';
            DLS_trial = squeeze(DLS_data(:, :, itrial))';
            V1_trial  = squeeze(V1_data(:, :, itrial))'; % NEW
            
            if ~isempty(DMS_trial) && ~isempty(ACC_trial)
                all_cross_area_correlations_DMSACC(itrial, :, :) = corr(DMS_trial, ACC_trial, 'Rows', 'complete');
            end
            if ~isempty(DMS_trial) && ~isempty(DLS_trial)
                all_cross_area_correlations_DMSDLS(itrial, :, :) = corr(DMS_trial, DLS_trial, 'Rows', 'complete');
            end
            % NEW: V1 correlations
            if ~isempty(V1_trial) && ~isempty(ACC_trial)
                all_cross_area_correlations_V1ACC(itrial, :, :) = corr(V1_trial, ACC_trial, 'Rows', 'complete');
            end
            if ~isempty(V1_trial) && ~isempty(DMS_trial)
                all_cross_area_correlations_V1DMS(itrial, :, :) = corr(V1_trial, DMS_trial, 'Rows', 'complete');
            end
        end
        
        mean_cross_area_corr_DMSACC = squeeze(mean(all_cross_area_correlations_DMSACC, [2, 3], 'omitnan'));
        mean_cross_area_corr_DMSDLS = squeeze(mean(all_cross_area_correlations_DMSDLS, [2, 3], 'omitnan'));
        mean_abs_cross_area_corr_DMSACC = squeeze(mean(abs(all_cross_area_correlations_DMSACC), [2, 3], 'omitnan'));
        mean_abs_cross_area_corr_DMSDLS = squeeze(mean(abs(all_cross_area_correlations_DMSDLS), [2, 3], 'omitnan'));
        
        % NEW: Average V1 correlations
        if n_neurons_V1 > 0
            mean_cross_area_corr_V1ACC = squeeze(mean(all_cross_area_correlations_V1ACC, [2, 3], 'omitnan'));
            mean_cross_area_corr_V1DMS = squeeze(mean(all_cross_area_correlations_V1DMS, [2, 3], 'omitnan'));
            mean_abs_cross_area_corr_V1ACC = squeeze(mean(abs(all_cross_area_correlations_V1ACC), [2, 3], 'omitnan'));
            mean_abs_cross_area_corr_V1DMS = squeeze(mean(abs(all_cross_area_correlations_V1DMS), [2, 3], 'omitnan'));
        else
            mean_cross_area_corr_V1ACC = nan(n_trials, 1);
            mean_cross_area_corr_V1DMS = nan(n_trials, 1);
            mean_abs_cross_area_corr_V1ACC = nan(n_trials, 1);
            mean_abs_cross_area_corr_V1DMS = nan(n_trials, 1);
        end

        % 8. PCA (Stim & Dark)
        calc_dim = @(data, N) find(cumsum(pca(data', 'Centered', true)) >= 90, 1) / N;
        
        % Stim PCA
        n_total_neurons = size(spatial_binned_fr_all, 1);
        stim_data_reshaped = reshape(spatial_binned_fr_all, n_total_neurons, []);
        [~, ~, ~, ~, explained_stim] = pca(stim_data_reshaped', 'Centered', true);
        stim_dimensionality_all = find(cumsum(explained_stim) >= 90, 1) / n_total_neurons;
        
        % Dark PCA
        dark_data_reshaped = reshape(temp_binned_dark_fr, n_total_neurons, []);
        [~, ~, ~, ~, explained_dark] = pca(dark_data_reshaped', 'Centered', true);
        dark_dimensionality_all = find(cumsum(explained_dark) >= 90, 1) / n_total_neurons;
        
        % Sub-area PCA (Stim)
        DMS_reshaped = reshape(DMS_data, n_neurons_DMS, []);
        DLS_reshaped = reshape(DLS_data, n_neurons_DLS, []);
        ACC_reshaped = reshape(ACC_data, n_neurons_ACC, []);
        
        stim_dimensionality_DMS = calc_dim(DMS_reshaped, n_neurons_DMS);
        stim_dimensionality_DLS = calc_dim(DLS_reshaped, n_neurons_DLS);
        stim_dimensionality_ACC = calc_dim(ACC_reshaped, n_neurons_ACC);
        
        % Sub-area PCA (Dark)
        DMS_dark = reshape(temp_binned_dark_fr(is_dms,:,:), n_neurons_DMS, []);
        DLS_dark = reshape(temp_binned_dark_fr(is_dls,:,:), n_neurons_DLS, []);
        ACC_dark = reshape(temp_binned_dark_fr(is_acc,:,:), n_neurons_ACC, []);
        
        dark_dimensionality_DMS = calc_dim(DMS_dark, n_neurons_DMS);
        dark_dimensionality_DLS = calc_dim(DLS_dark, n_neurons_DLS);
        dark_dimensionality_ACC = calc_dim(ACC_dark, n_neurons_ACC);
        
        % NEW: V1 Dimensionality
        if n_neurons_V1 > 0
            V1_reshaped = reshape(V1_data, n_neurons_V1, []);
            stim_dimensionality_V1 = calc_dim(V1_reshaped, n_neurons_V1);
            
            V1_dark = reshape(temp_binned_dark_fr(is_v1,:,:), n_neurons_V1, []);
            dark_dimensionality_V1 = calc_dim(V1_dark, n_neurons_V1);
        else
            stim_dimensionality_V1 = NaN;
            dark_dimensionality_V1 = NaN;
        end

        % 9. Generalized Variance
        generalized_variances_stim = zeros(1, n_trials);
        generalized_variances_dark = zeros(1, n_trials);
        
        try 
            sv_stim = pagesvd(spatial_binned_fr_all, 'econ');
            sv_dark = pagesvd(temp_binned_dark_fr, 'econ');
            generalized_variances_stim = sum(log(sv_stim.^2), 1) / n_total_neurons;
            generalized_variances_dark = sum(log(sv_dark.^2), 1) / n_total_neurons;
            generalized_variances_stim = squeeze(generalized_variances_stim)';
            generalized_variances_dark = squeeze(generalized_variances_dark)';
        catch
            for itrial = 1:n_trials
                generalized_variances_stim(itrial) = sum(log(svd(spatial_binned_fr_all(:,:,itrial), 'econ').^2)) / n_total_neurons;
                generalized_variances_dark(itrial) = sum(log(svd(temp_binned_dark_fr(:,:,itrial), 'econ').^2)) / n_total_neurons;
            end
        end
        
        % --- Save Results ---
        preprocessed_data(ianimal).trialData = trialData;
        preprocessed_data(ianimal).is_dms = is_dms;
        preprocessed_data(ianimal).is_dls = is_dls;
        preprocessed_data(ianimal).is_acc = is_acc;
        preprocessed_data(ianimal).is_v1  = is_v1; % NEW
        
        preprocessed_data(ianimal).binned_spikes_trials = binned_spikes_trials;
        preprocessed_data(ianimal).npx_times_trials = npx_times_trials;
        
        preprocessed_data(ianimal).trial_metrics = trial_metrics;
        preprocessed_data(ianimal).change_point_mean = change_point_mean;
        
        preprocessed_data(ianimal).trial_average_fr_dms = trial_average_fr_dms;
        preprocessed_data(ianimal).trial_sem_fr_dms = trial_sem_fr_dms;
        preprocessed_data(ianimal).trial_average_fr_dls = trial_average_fr_dls;
        preprocessed_data(ianimal).trial_sem_fr_dls = trial_sem_fr_dls;
        preprocessed_data(ianimal).trial_average_fr_acc = trial_average_fr_acc;
        preprocessed_data(ianimal).trial_sem_fr_acc = trial_sem_fr_acc;
        preprocessed_data(ianimal).trial_average_fr_v1  = trial_average_fr_v1; % NEW
        preprocessed_data(ianimal).trial_sem_fr_v1      = trial_sem_fr_v1; % NEW
        
        preprocessed_data(ianimal).darkData = darkData;
        preprocessed_data(ianimal).corridorData = corridorData;
        
        preprocessed_data(ianimal).trial_lick_positions = trial_lick_positions;
        preprocessed_data(ianimal).trial_lick_errors = trial_lick_errors;
        preprocessed_data(ianimal).shuffled_lick_error_means = shuffled_lick_error_means;
        preprocessed_data(ianimal).shuffled_lick_error_stds = shuffled_lick_error_stds;
        preprocessed_data(ianimal).trial_lick_fractions = trial_lick_fractions;
        preprocessed_data(ianimal).zscored_lick_errors = zscored_lick_errors;
        
        preprocessed_data(ianimal).spatial_binned_data = spatial_binned_data; 
        preprocessed_data(ianimal).temp_binned_dark_fr = temp_binned_dark_fr;
        
        preprocessed_data(ianimal).spatial_binned_fr_all = spatial_binned_fr_all;
        preprocessed_data(ianimal).z_spatial_binned_fr_all = z_spatial_binned_fr_all;
        
        preprocessed_data(ianimal).mean_cross_area_corr_DMSACC = mean_cross_area_corr_DMSACC;
        preprocessed_data(ianimal).mean_cross_area_corr_DMSDLS = mean_cross_area_corr_DMSDLS;
        preprocessed_data(ianimal).mean_abs_cross_area_corr_DMSACC = mean_abs_cross_area_corr_DMSACC;
        preprocessed_data(ianimal).mean_abs_cross_area_corr_DMSDLS = mean_abs_cross_area_corr_DMSDLS;
        
        % NEW V1 Correlations Saved
        preprocessed_data(ianimal).mean_cross_area_corr_V1ACC = mean_cross_area_corr_V1ACC;
        preprocessed_data(ianimal).mean_cross_area_corr_V1DMS = mean_cross_area_corr_V1DMS;
        preprocessed_data(ianimal).mean_abs_cross_area_corr_V1ACC = mean_abs_cross_area_corr_V1ACC;
        preprocessed_data(ianimal).mean_abs_cross_area_corr_V1DMS = mean_abs_cross_area_corr_V1DMS;
        
        % NEW V1 Dimensionalities Saved
        preprocessed_data(ianimal).stim_dimensionality_V1 = stim_dimensionality_V1;
        preprocessed_data(ianimal).dark_dimensionality_V1 = dark_dimensionality_V1;
        
        preprocessed_data(ianimal).n_trials = n_trials;
        preprocessed_data(ianimal).final_neurontypes = all_data(ianimal).final_neurontypes;
        fprintf('Done with animal %d\n', ianimal);
    end
    
    save('preprocessed_data.mat', 'preprocessed_data', '-v7.3');
end

%%
% Define colors for each area
color_dms = [0, 0.4470, 0.7410];       % Deep Blue for DMS
color_dls =  [0.4660, 0.6740, 0.1880];  % Forest Green for DLS
color_acc = [0.8500, 0.3250, 0.0980];  % Crimson Red for ACC

average_lick_precision = cellfun(@(x) mean(x, 'omitmissing'), {preprocessed_data(:).zscored_lick_errors});


for ianimal = 1:n_animals
    % Get area indices
    is_dms = preprocessed_data(ianimal).is_dms;
    is_dls = preprocessed_data(ianimal).is_dls;
    is_acc = preprocessed_data(ianimal).is_acc;

    % Get neuron counts
    n_neurons_DMS = sum(is_dms);
    n_neurons_DLS = sum(is_dls);
    n_neurons_ACC = sum(is_acc);
    n_total_neurons = size(preprocessed_data(ianimal).temp_binned_dark_fr, 1);

    % Get number of trials for this animal
    n_trials = preprocessed_data(ianimal).n_trials;

    % Extract data for the trials up to n_trials
    DMS_data = preprocessed_data(ianimal).temp_binned_dark_fr(is_dms, :, 1:n_trials);
    DLS_data = preprocessed_data(ianimal).temp_binned_dark_fr(is_dls, :, 1:n_trials);
    ACC_data = preprocessed_data(ianimal).temp_binned_dark_fr(is_acc, :, 1:n_trials);
    dark_data = preprocessed_data(ianimal).temp_binned_dark_fr(:, :, 1:n_trials);

    % Reshape data for PCA
    DMS_data_reshaped = DMS_data(:, :);
    DLS_data_reshaped = DLS_data(:, :);
    ACC_data_reshaped = ACC_data(:, :);
    dark_data_reshaped = dark_data(:, :);

    % Perform PCA on the reshaped data
    [~, ~, ~, ~, explained_dark, ~] = pca(dark_data_reshaped', "Centered", true);
    cumsum_explained_dark = cumsum(explained_dark);
    dark_dimensionality_all = find(cumsum_explained_dark >= 90, 1) / n_total_neurons;

    [~, ~, ~, ~, explained_dark_DMS, ~] = pca(DMS_data_reshaped', "Centered", true);
    cumsum_explained_dark_DMS = cumsum(explained_dark_DMS);
    dark_dimensionality_DMS = find(cumsum_explained_dark_DMS >= 90, 1) / n_neurons_DMS;

    [~, ~, ~, ~, explained_dark_DLS, ~] = pca(DLS_data_reshaped', "Centered", true);
    cumsum_explained_dark_DLS = cumsum(explained_dark_DLS);
    dark_dimensionality_DLS = find(cumsum_explained_dark_DLS >= 90, 1) / n_neurons_DLS;

    [~, ~, ~, ~, explained_dark_ACC, ~] = pca(ACC_data_reshaped', "Centered", true);
    cumsum_explained_dark_ACC = cumsum(explained_dark_ACC);
    dark_dimensionality_ACC = find(cumsum_explained_dark_ACC >= 90, 1) / n_neurons_ACC;

    % Store results in preprocessed_data
    preprocessed_data(ianimal).pca_dark_dimensionality_all = dark_dimensionality_all;
    preprocessed_data(ianimal).pca_dark_dimensionality_dms = dark_dimensionality_DMS;
    preprocessed_data(ianimal).pca_dark_dimensionality_dls = dark_dimensionality_DLS;
    preprocessed_data(ianimal).pca_dark_dimensionality_acc = dark_dimensionality_ACC;
end

% Collect data across animals
dimensionality_dark_all = cell2mat({preprocessed_data(:).pca_dark_dimensionality_all});
dimensionality_dark_DMS = cell2mat({preprocessed_data(:).pca_dark_dimensionality_dms});
dimensionality_dark_DLS = cell2mat({preprocessed_data(:).pca_dark_dimensionality_dls});
dimensionality_dark_ACC = cell2mat({preprocessed_data(:).pca_dark_dimensionality_acc});

% Define groups for ANOVA
area_groups = [ones(size(dimensionality_dark_DMS)), 2*ones(size(dimensionality_dark_DLS)), 3*ones(size(dimensionality_dark_ACC))];
animal_groups = [1:numel(dimensionality_dark_DMS), 1:numel(dimensionality_dark_DLS), 1:numel(dimensionality_dark_ACC)];

% Plot results
figure
my_errorbar_plot(dimensionality_dark_DMS, dimensionality_dark_DLS, dimensionality_dark_ACC)
ylabel('Relative Dimensionality')
xticklabels({'DMS', 'DLS', 'ACC'})
title('Dimensionality During Dark')

% Statistical analysis
[~, ~, stats] = anovan([dimensionality_dark_DMS, dimensionality_dark_DLS, dimensionality_dark_ACC]', {area_groups, animal_groups}, "varnames", {'area', 'animal'}, 'display', 'off');
[comp, ~] = multcompare(stats, 'Display','off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6))

save_to_svg('area_dark_dimensionality')

% Compare task vs dark dimensionality
figure
my_errorbar_plot(dimensionality_stim_all, dimensionality_dark_all, true)
[~, pval] = ttest(dimensionality_stim_all', dimensionality_dark_all');
sigstar({[1, 2]}, pval)
xticklabels({'Stim', 'Dark'})
ylabel('Relative Dimensionality')
title('Task vs Dark Dimensionality')


%% Decoding of position

% Example usage with custom options
options = struct('bin_size', 5, 'model_type', 'linear', 'area', 'all', 'n_bootstraps', 20, 'neuron_counts', [50]);

% Run decoder
[decoded_positions, decoder_performance, decoder_coefficients] = decode_position(preprocessed_data, options);

neuron_counts_to_plot = options.neuron_counts;  % Adjust selection as needed

% Visualization of decoded positions
visualize_decoding_results(decoded_positions, decoder_performance, options.bin_size);

% Visualization of trial evolution
visualize_trial_evolution(decoded_positions, decoder_performance, options.bin_size, neuron_counts_to_plot);

% Performance scaling with neuron count
visualize_performance_vs_neuron_count(decoder_performance);

% Decoding vs behavioral performance correlations
visualize_performance_correlations(preprocessed_data, decoded_positions, decoder_performance, options.bin_size, neuron_counts_to_plot);

all_decoder_performances = {decoder_performance(:).r2};
valid_animals = cellfun(@(x) size(x, 1) > 0, all_decoder_performances);
mean_decoding_performance = cellfun(@(x) mean(x(end, :), "omitmissing"), all_decoder_performances(valid_animals));


options_mle = struct('bin_size', 5, 'area', 'all', 'n_bootstraps', 100, 'neuron_counts', [100]);
[decoded_positions_mle, decoder_performance_mle] = decode_position_mld(preprocessed_data, options_mle);
plot_decoder_accuracy_vs_chance(decoder_performance_mle, 100)
visualize_decoding_results(decoded_positions_mle, decoder_performance_mle, options_mle.bin_size);


for ianimal = 1:n_animals
    preprocessed_data(ianimal).decoded_positions = decoded_positions{ianimal};
    preprocessed_data(ianimal).decoding_options = options;
    preprocessed_data(ianimal).decoding_performance = decoder_performance(ianimal);

    % preprocessed_data(ianimal).decoded_positions_mle = decoded_positions_mle{ianimal};
    % preprocessed_data(ianimal).decoding_options_mle = options_mle;
    % preprocessed_data(ianimal).decoding_performance_mle = decoder_performance_mle(ianimal);
end
save('preprocessed_data.mat', 'preprocessed_data', '-v7.3');


%% Decoding with behaviour

% Assume the following variables are available:
% - preprocessed_data: struct containing data for each animal
% - decoded_positions: cell array from the decode_position function
% - decoder_performance: struct containing performance metrics
% - options: options used in the decode_position function
% - n_animals: number of animals

% Step 1: Identify trial indices for each condition
% Load z-scored lick errors for all animals
zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

% Initialize logical indices for each condition
first_idx = cell(1, n_animals);
precise_idx = cell(1, n_animals);
imprecise_idx = cell(1, n_animals);

for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;

    % First 3 trials
    temp_first_idx = false(1, n_trials);
    temp_first_idx(1:min(3, n_trials)) = true; % Adjust for animals with fewer than 3 trials
    first_idx{ianimal} = temp_first_idx;

    % Precise trials (z-scored lick error <= -3)
    precise_idx{ianimal} = zscored_lick_errors_all{ianimal} <= -2;
    % Exclude the first 3 trials
    precise_idx{ianimal}(1:min(3, n_trials)) = false;

    % Imprecise (random) trials (z-scored lick error > 0)
    imprecise_idx{ianimal} = zscored_lick_errors_all{ianimal} > -1;
    % Exclude the first 3 trials
    imprecise_idx{ianimal}(1:min(3, n_trials)) = false;
end

% Step 2: For each animal, select one neuron count and find the best bootstrap
n_conditions = 3;
condition_names = {'First 3 Trials', 'Precise Trials', 'Imprecise Trials'};

% Initialize matrices to store mean RMSE and R² per condition per animal
mean_rmse_per_animal = nan(n_animals, n_conditions);
mean_r2_per_animal = nan(n_animals, n_conditions);

selected_neuron_count = 50;

for ianimal = 1:n_animals
    fprintf('Processing animal %d...\n', ianimal);

    % Get available neuron counts
    neuron_counts = decoder_performance(ianimal).neuron_counts;
    n_counts = length(neuron_counts);

    max_count_idx = find(neuron_counts == selected_neuron_count);
    if isempty(max_count_idx)
        continue
    end
    % Select the maximum neuron count
    % [max_neuron_count, max_count_idx] = max(neuron_counts);
    % fprintf('Selected neuron count: %d\n', max_neuron_count);

    % Get the number of bootstrap iterations
    n_bootstraps = options.n_bootstraps;
    avg_rmse_per_bootstrap = nan(n_bootstraps, 1);

    % Step 2a: Find the best bootstrap iteration based on average RMSE
    for ibootstrap = 1:n_bootstraps
        % Check if decoded positions are available
        if isempty(decoded_positions{ianimal}{max_count_idx, ibootstrap})
            continue;
        end

        % Get decoded positions for this bootstrap
        decoded_positions_matrix = decoded_positions{ianimal}{max_count_idx, ibootstrap}; % [n_pos_bins x n_trials]
        n_trials = size(decoded_positions_matrix, 2);
        n_pos_bins = size(decoded_positions_matrix, 1);

        % True positions (same for all trials)
        true_positions_vector = (1:n_pos_bins)' * options.bin_size; % [n_pos_bins x 1]

        % Compute RMSE per trial
        rmse_per_trial = nan(n_trials, 1);
        for itrial = 1:n_trials
            y_pred = decoded_positions_matrix(:, itrial);
            y_true = true_positions_vector;

            % Remove NaNs (if any)
            valid_idx = ~isnan(y_pred) & ~isnan(y_true);
            y_pred = y_pred(valid_idx);
            y_true = y_true(valid_idx);

            % Compute RMSE
            rmse = sqrt(mean((y_pred - y_true).^2));
            rmse_per_trial(itrial) = rmse;
        end

        % Compute average RMSE across all trials
        avg_rmse = mean(rmse_per_trial, 'omitnan');
        avg_rmse_per_bootstrap(ibootstrap) = avg_rmse;
    end

    % Check if any valid bootstraps were found
    if all(isnan(avg_rmse_per_bootstrap))
        warning('No valid bootstraps found for animal %d. Skipping.', ianimal);
        continue;
    end

    % Step 2b: Select the bootstrap with the lowest average RMSE
    [~, best_bootstrap_idx] = min(avg_rmse_per_bootstrap);
    fprintf('Selected best bootstrap iteration: %d\n', best_bootstrap_idx);

    % Step 3: Compute per-trial performance metrics using the best bootstrap
    decoded_positions_matrix = decoded_positions{ianimal}{max_count_idx, best_bootstrap_idx};
    n_trials = size(decoded_positions_matrix, 2);
    n_pos_bins = size(decoded_positions_matrix, 1);

    % True positions (same for all trials)
    true_positions_vector = (1:n_pos_bins)' * options.bin_size; % [n_pos_bins x 1]

    % Initialize arrays to store performance metrics per trial
    rmse_per_trial = nan(n_trials, 1);
    r2_per_trial = nan(n_trials, 1);

    % Compute performance metrics per trial
    for itrial = 1:n_trials
        y_pred = decoded_positions_matrix(:, itrial);
        y_true = true_positions_vector;

        % Remove NaNs (if any)
        valid_idx = ~isnan(y_pred) & ~isnan(y_true);
        y_pred = y_pred(valid_idx);
        y_true = y_true(valid_idx);

        % Compute RMSE
        rmse = sqrt(mean((y_pred - y_true).^2));

        % Compute R²
        ss_res = sum((y_true - y_pred).^2);
        ss_tot = sum((y_true - mean(y_true)).^2);
        r2 = 1 - ss_res / ss_tot;

        % Store the metrics
        rmse_per_trial(itrial) = rmse;
        r2_per_trial(itrial) = r2;
    end

    % Step 4: Group the trials into conditions and compute mean metrics per condition
    idx_first = first_idx{ianimal};
    idx_precise = precise_idx{ianimal};
    idx_imprecise = imprecise_idx{ianimal};

    % For each condition, compute mean RMSE and R²
    for icondition = 1:n_conditions
        switch icondition
            case 1 % First 3 trials
                trial_idx = idx_first;
            case 2 % Precise trials
                trial_idx = idx_precise;
            case 3 % Imprecise trials
                trial_idx = idx_imprecise;
        end

        % Get the performance metrics for these trials
        rmse_trials = rmse_per_trial(trial_idx);
        r2_trials = r2_per_trial(trial_idx);

        % Compute mean values, handling cases with no trials
        if ~isempty(rmse_trials)
            mean_rmse_per_animal(ianimal, icondition) = mean(rmse_trials, 'omitnan');
        end
        if ~isempty(r2_trials)
            mean_r2_per_animal(ianimal, icondition) = mean(r2_trials, 'omitnan');
        end
    end
end

% Remove animals with NaNs in all conditions
valid_animals = ~all(isnan(mean_rmse_per_animal), 2);
mean_rmse_per_animal = mean_rmse_per_animal(valid_animals, :);
mean_r2_per_animal = mean_r2_per_animal(valid_animals, :);
n_animals_valid = sum(valid_animals);

% Step 5: Perform Repeated-Measures ANOVA

% Define the within-subjects factor
within = table({'First3'; 'Precise'; 'Imprecise'}, 'VariableNames', {'Condition'});

% For RMSE
% Create a table for RMSE
rmse_table = array2table(mean_rmse_per_animal, 'VariableNames', {'First3', 'Precise', 'Imprecise'});
rmse_table.AnimalID = (1:n_animals_valid)';

% Fit the repeated-measures model
rm_rm = fitrm(rmse_table, 'First3,Precise,Imprecise~1', 'WithinDesign', within);

% Perform the ANOVA
ranovatbl_rmse = ranova(rm_rm);
disp('RMSE ANOVA Results:');
disp(ranovatbl_rmse);

% Post-hoc comparisons for RMSE
[rmse_multcompare] = multcompare(rm_rm, 'Condition', 'ComparisonType', 'bonferroni');
disp('RMSE Post-hoc Comparisons:');
disp(rmse_multcompare);

% For R²
% Create a table for R²
r2_table = array2table(mean_r2_per_animal, 'VariableNames', {'First3', 'Precise', 'Imprecise'});
r2_table.AnimalID = (1:n_animals_valid)';

% Fit the repeated-measures model
r2_rm = fitrm(r2_table, 'First3,Precise,Imprecise~1', 'WithinDesign', within);

% Perform the ANOVA
ranovatbl_r2 = ranova(r2_rm);
disp('R² ANOVA Results:');
disp(ranovatbl_r2);

% Post-hoc comparisons for R²
[r2_multcompare] = multcompare(r2_rm, 'Condition', 'ComparisonType', 'bonferroni');
disp('R² Post-hoc Comparisons:');
disp(r2_multcompare);

% Step 6: Plot the results and add significance markers

% Prepare data for plotting
rmse_plot_data = num2cell(mean_rmse_per_animal, 1);
r2_plot_data = num2cell(mean_r2_per_animal, 1);

% Plot RMSE with error bars
figure;
my_errorbar_plot(rmse_plot_data);
xticks(1:n_conditions);
xticklabels({'First 3', 'Precise', 'Imprecise'});
ylabel('RMSE');
title('Decoding RMSE Across Conditions');

% Add significance markers for RMSE
hold on;
comparison_pairs = [1 3; 1 2; 2 3]; % Pairs of conditions
p_values = rmse_multcompare.pValue;

for i = 1:size(comparison_pairs, 1)
    cond1 = comparison_pairs(i, 1);
    cond2 = comparison_pairs(i, 2);
    p = p_values(i);
    if p < 0.05
        sigstar({[cond1, cond2]}, p);
    end
end
hold off;

% Plot R² with error bars
figure;
my_errorbar_plot(r2_plot_data);
xticks(1:n_conditions);
xticklabels({'First 3', 'Precise', 'Imprecise'});
ylabel('R²');
title('Decoding R² Across Conditions');

% Add significance markers for R²
hold on;
p_values = r2_multcompare.pValue;

for i = 1:size(comparison_pairs, 1)
    cond1 = comparison_pairs(i, 1);
    cond2 = comparison_pairs(i, 2);
    p = p_values(i);
    if p < 0.05
        sigstar({[cond1, cond2]}, p);
    end
end
hold off;


%% Behaviour plotting

for ianimal = 1:n_animals
    n_trials = size(preprocessed_data(ianimal).zscored_lick_errors, 2);
    trial_lick_numbers = preprocessed_data(ianimal).trial_metrics.trial_lick_no(1:n_trials);

    zscored_lick_errors = preprocessed_data(ianimal).zscored_lick_errors(1:n_trials);

    trial_lick_fractions = preprocessed_data(ianimal).trial_lick_fractions(1:n_trials); % Adjust if necessary


    % Plot the correlations between areas
    figure
    mean_corr_DMSDLS = preprocessed_data(ianimal).mean_cross_area_corr_DMSDLS(1:n_trials);
    mean_corr_DMSACC = preprocessed_data(ianimal).mean_cross_area_corr_DMSACC(1:n_trials);

    subplot(1, 2, 1)
    hold on
    scatter(mean_corr_DMSDLS, zscored_lick_errors')
    xlabel('Cross-area correlation')
    ylabel('Lick error')
    title('DMS-DLS')
    axis tight
    lsline
    [rho, pval] = corr(mean_corr_DMSDLS, zscored_lick_errors', "Rows", "complete");
    legend(sprintf('\\rho = %.3f, pval = %.4f', rho, pval))


    subplot(1, 2, 2)
    hold on
    scatter(mean_corr_DMSACC, zscored_lick_errors)
    xlabel('Cross-area correlation')
    ylabel('Lick error')
    title('DMS-ACC')
    axis tight
    lsline
    [rho, pval] = corr(mean_corr_DMSACC, zscored_lick_errors', "Rows", "complete");
    legend(sprintf('\\rho = %.3f, pval = %.4f', rho, pval))

    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    sgtitle(sprintf('animal %d', ianimal))

    save_to_svg(sprintf('cross_area_lickerror_animal%d', ianimal))


    % Plot licking performance
    mov_window_size = 5;

    figure
    subplot(5, 1, 1)
    shadedErrorBar(1:length(trial_lick_fractions), movmean(trial_lick_fractions, mov_window_size, 'omitmissing'), movstd(trial_lick_fractions, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    ylabel('precise lick fraction')
    axis tight
    subplot(5, 1, 2)
    trials = n_trials;
    shadedErrorBar(1:trials, movmean(zscored_lick_errors, mov_window_size, 'omitmissing'), movstd(zscored_lick_errors, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    % Find indices where z-scores are greater than 2 or less than -2
    outlier_idx = find(zscored_lick_errors <= -2);

    % Get the maximum y-value of the data
    y_max = max(movmean(zscored_lick_errors, mov_window_size, 'omitmissing') + movstd(zscored_lick_errors, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size));

    % Set y-level for asterisks slightly above the maximum value
    asterisk_y = y_max + 0.01 * (y_max - min(zscored_lick_errors));  % Adjust 0.1 as necessary
    hold on;
    plot(outlier_idx, repmat(asterisk_y, size(outlier_idx)), 'r*', 'MarkerSize', 2);  % Red asterisks
    ylabel('lick error')
    axis tight
    subplot(5, 1, 3)
    shadedErrorBar(1:trials, movmean(trial_lick_numbers, mov_window_size), movstd(trial_lick_numbers, mov_window_size)/sqrt(mov_window_size))
    ylabel('lick no')
    axis tight
    % xlabel('trial #')

    subplot(5, 1, 4)
    shadedErrorBar(1:trials, movmean(mean_corr_DMSDLS, mov_window_size, 'omitmissing'), movstd(mean_corr_DMSDLS, mov_window_size, [], 1, 'omitmissing')/sqrt(mov_window_size))
    axis tight
    title('DMS-DLS correlation')
    ylabel(sprintf('\\rho'))
    subplot(5, 1, 5)
    shadedErrorBar(1:trials, movmean(mean_corr_DMSACC, mov_window_size, 'omitmissing'), movstd(mean_corr_DMSACC, mov_window_size, [], 1, 'omitmissing')/sqrt(mov_window_size))
    axis tight
    title('DMS-ACC correlation')
    ylabel(sprintf('\\rho'))
    xlabel('trial #')

    sgtitle(sprintf('animal %d', ianimal))
    fig = gcf();
    fig.Position = [1145, 15, 560, 980];
    save_to_svg(sprintf('lickquant_animal%d', ianimal))
end


%% Trial-to-Trial Correlation

% Logical indices for each area
areas = {'DMS', 'DLS', 'ACC'};
area_colors = {color_dms, color_dls, color_acc};

avg_neuron_corrs = cell(1, n_animals); % Initialize cell array
avg_neuron_corrs_spear = cell(1, n_animals); % Initialize cell array
avg_population_corrs = cell(1, n_animals);


for ianimal = 1:n_animals
    all_activity = preprocessed_data(ianimal).spatial_binned_fr_all;
    is_dms = preprocessed_data(ianimal).is_dms;
    is_dls = preprocessed_data(ianimal).is_dls;
    is_acc = preprocessed_data(ianimal).is_acc;

    area_indices = {is_dms, is_dls, is_acc};

    [neurons, spatial_bins, ~] = size(all_activity);
    trials = 50;
    window_size = 5;
    half_window = floor(window_size / 2);

    % Preallocate the output matrix with NaNs to handle cases with insufficient data
    avg_corrs = NaN(neurons, trials);
    avg_corrs_spear = NaN(neurons, trials);

    % Loop over each neuron
    for n = 1:neurons
        % Loop over each trial
        for t = 1:trials
            % Define the window of trials centered on trial t
            trial_start = max(1, t - half_window);
            trial_end = min(trials, t + half_window);
            trials_in_window = trial_start:trial_end;
            num_trials_in_window = length(trials_in_window);

            % Proceed only if we have at least two trials to correlate
            if num_trials_in_window > 1
                % Extract the data for the current neuron and window
                data_block = squeeze(all_activity(n, :, trials_in_window)); % Size: [spatial_bins x num_trials_in_window]

                % Compute the correlation matrix for the trials in the current window
                R_spear = corr(data_block, 'Type', 'Spearman');
                R = corr(data_block);

                % Extract the upper triangle of the correlation matrix (excluding the diagonal)
                upper_triangle = triu(R, 1);
                upper_vals = upper_triangle(upper_triangle ~= 0);

                % Compute the average correlation
                avg_corrs(n, t) = mean(upper_vals);


                upper_triangle_spear = triu(R_spear, 1);
                upper_vals_spear = upper_triangle_spear(upper_triangle_spear ~= 0);

                % Compute the average correlation
                avg_corrs_spear(n, t) = mean(upper_vals_spear);
            else
                % If only one trial is available, set the average correlation to NaN
                avg_corrs(n, t) = NaN;
            end
        end
    end

    avg_neuron_corrs{ianimal} = avg_corrs;

    avg_neuron_corrs_spear{ianimal} = avg_corrs_spear;

    % Preallocate a matrix for population correlations: [areas x trials]
    pop_corrs = NaN(numel(areas), trials);

    for a = 1:numel(areas)
        is_area = area_indices{a};
        area_neurons = find(is_area);
        n_area_neurons = sum(is_area);

        if n_area_neurons < 2
            continue; % Skip if not at least 2 neurons in the area
        end

        for t = 1:trials
            % Define the sliding window centered on trial t
            trial_start = max(1, t - half_window);
            trial_end = min(trials, t + half_window);
            trials_in_window = trial_start:trial_end;
            num_trials_in_window = length(trials_in_window);

            if num_trials_in_window > 1
                % Extract firing rate matrices for all neurons in the area within the window
                % [neurons_in_area x spatial_bins x window_size]
                data_block = all_activity(area_neurons, :, trials_in_window);

                % Initialize a vector to store corr2 values for all unique trial pairs in the window
                corr2_vals = [];

                % Compute corr2 for each unique pair of trials within the window
                for i = 1:(num_trials_in_window-1)
                    for j = (i+1):num_trials_in_window
                        trial_i = squeeze(data_block(:, :, i)); % [neurons_in_area x spatial_bins]
                        trial_j = squeeze(data_block(:, :, j)); % [neurons_in_area x spatial_bins]
                        R = corr2(trial_i, trial_j);
                        corr2_vals(end+1) = R; %#ok<SAGROW>
                    end
                end

                % Compute the average corr2 for the window
                pop_corrs(a, t) = mean(corr2_vals, 'omitnan');
            else
                pop_corrs(a, t) = NaN;
            end
        end
    end

    avg_population_corrs{ianimal} = pop_corrs;

    % Plotting
    figure
    subplot(3, 2, [1, 3, 5])
    imagesc(avg_corrs)
    ylabel('Neurons')
    xlabel('Trials')
    yline(sum(is_dms), 'Color', 'w', 'LineWidth', 1)
    yline(sum(is_dms) + sum(is_dls), 'Color', 'w', 'LineWidth', 1)
    colorbar

    if sum(is_dms) > 1
        subplot(3, 2, 2)
        shadedErrorBar(1:trials, mean(avg_corrs(is_dms, :), 'omitmissing'), sem(avg_corrs(is_dms, :)), 'lineprops', {'Color', color_dms})
        ylabel('Corr')
        xlabel('Trials')
        title('DMS Only')
        axis tight
        % ylim([-0.05, 0.5])
    end

    if sum(is_dls) > 1
        subplot(3, 2, 4)
        shadedErrorBar(1:trials, mean(avg_corrs(is_dls, :), 'omitmissing'), sem(avg_corrs(is_dls, :)), 'lineprops', {'Color', color_dls})
        ylabel('Corr')
        xlabel('Trials')
        title('DLS Only')
        axis tight
        % ylim([-0.05, 0.5])
    end

    if sum(is_acc) > 1
        subplot(3, 2, 6)
        shadedErrorBar(1:trials, mean(avg_corrs(is_acc, :), 'omitmissing'), sem(avg_corrs(is_acc, 1:50)), 'lineprops', {'Color', color_acc})
        ylabel('Corr')
        xlabel('Trials')
        title('ACC Only')
        axis tight
        % ylim([-0.05, 0.5])
    end

    sgtitle(sprintf('Average Trial-to-Trial Correlation - Animal %d', ianimal))
    fig = gcf();
    fig.Position = [933, 11, 750, 950];

    % save_to_svg(sprintf('average_trial_to_trial_correlation_animal%d', ianimal))

end

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals
    nexttile
    pop_corr = avg_population_corrs{ianimal}; % [areas x trials]
    % trials = 1:size(pop_corr, 2);
    trials = 1:50;

    hold on
    for a = 1:numel(areas)
        if any(~isnan(pop_corr(a, :)))
            plot(trials, pop_corr(a, trials), 'Color', area_colors{a}, 'LineWidth', 1);
        end
    end
    if ianimal == 1
        legend(areas, 'Location', 'best');
    end
end

xlabel(t, 'Trial Number');
ylabel(t, 'Population Correlation (Pearson''s corr2)');
title(t, 'Population-Level Trial-to-Trial Correlations per Area');



% close all
% figure
% t = tiledlayout('flow');
%
% for ianimal = 1:n_animals
%
%     nexttile
%     scatter(avg_neuron_corrs{ianimal}(:), avg_neuron_corrs_spear{ianimal}(:))
%     lsline
%     identity_line
%     axis tight
% end
% xlabel(t, 'pearsons rho')
% ylabel(t, 'spearmans rho')

% figure
% ianimal = 3;
% hold on
% for ineuron = 1:size(avg_neuron_corrs{ianimal}, 1)
%     scatter(squeeze(mean(preprocessed_data(ianimal).spatial_binned_fr_all(ineuron, :, :), 2)), avg_neuron_corrs{ianimal}(ineuron, :)')
%     title(num2str(ineuron))
%     lsline
% end


%% Trial-to-Trial Correlation of Behaviour

avg_lick_corrs = cell(1, n_animals);
avg_occupancy_corrs = cell(1, n_animals);

for ianimal = 1:n_animals
    n_trials = task_data(ianimal).n_trials;
    % n_trials = 50;

    % lick_data = preprocessed_data(ianimal).spatial_binned_data.licks(1:n_trials, :);
    % occupancy_data = preprocessed_data(ianimal).spatial_binned_data.durations(1:n_trials, :);
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

% Concatenate data across animals
max_trials = max(cellfun(@length, avg_lick_corrs));
avg_lick_corrs_matrix = nan(n_animals, max_trials);
avg_occupancy_corrs_matrix = nan(n_animals, max_trials);

for ianimal = 1:n_animals
    n_trials = length(avg_lick_corrs{ianimal});
    avg_lick_corrs_matrix(ianimal, 1:n_trials) = avg_lick_corrs{ianimal};
    avg_occupancy_corrs_matrix(ianimal, 1:n_trials) = avg_occupancy_corrs{ianimal};
end

% Plotting
figure
subplot(2, 2, 1)
shadedErrorBar(1:max_trials, mean(avg_lick_corrs_matrix, 'omitnan'), sem(avg_lick_corrs_matrix))
ylabel('Corr')
xlabel('Trial #')
title('Trial-to-Trial Lick Correlation')

subplot(2, 2, 2)
shadedErrorBar(1:max_trials, mean(avg_occupancy_corrs_matrix, 'omitnan'), sem(avg_occupancy_corrs_matrix))
ylabel('Corr')
xlabel('Trial #')
title('Trial-to-Trial Occupancy Correlation')

subplot(2, 2, 3)
scatter(avg_lick_corrs_matrix(:), avg_occupancy_corrs_matrix(:), 'filled', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
[rho, pval] = corr(avg_lick_corrs_matrix(:), avg_occupancy_corrs_matrix(:), 'Rows', 'complete');
lsline
legend(sprintf('\\rho = %.2f, pval = %.3f', rho, pval))

%% Stability of Behaviour vs Stability of Neural Activity

figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    % n_trials = preprocessed_data(ianimal).n_trials;
    n_trials = 50;
    avg_lick_corrs_animal = avg_lick_corrs{ianimal};  % This is now a vector
    avg_neuron_corrs_animal = mean(avg_neuron_corrs{ianimal}, 'omitnan');  % Mean over neurons, size [1 x n_trials]

    % Ensure both vectors are of the same length
    min_length = min(length(avg_lick_corrs_animal), length(avg_neuron_corrs_animal));
    avg_lick_corrs_animal = avg_lick_corrs_animal(1:min_length);
    avg_neuron_corrs_animal = avg_neuron_corrs_animal(1:min_length);

    nexttile
    scatter(avg_lick_corrs_animal, avg_neuron_corrs_animal, 'filled', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_lick_corrs_animal', avg_neuron_corrs_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d', ianimal))
end
xlabel(t, 'Lick Stability')
ylabel(t, 'Neural Stability')

% Define colors for each area
color_dms = [0, 0.4470, 0.7410];       % Deep Blue for DMS
color_dls = [0.4660, 0.6740, 0.1880];  % Forest Green for DLS
color_acc = [0.8500, 0.3250, 0.0980];  % Crimson Red for ACC

% Plot for DMS
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;
    is_dms = preprocessed_data(ianimal).is_dms;
    avg_lick_corrs_animal = avg_lick_corrs{ianimal};
    avg_neuron_corrs_dms = mean(avg_neuron_corrs{ianimal}(is_dms, :), 'omitnan');  % Mean over DMS neurons

    % Ensure both vectors are of the same length
    min_length = min(length(avg_lick_corrs_animal), length(avg_neuron_corrs_dms));
    avg_lick_corrs_animal = avg_lick_corrs_animal(1:min_length);
    avg_neuron_corrs_dms = avg_neuron_corrs_dms(1:min_length);

    nexttile
    scatter(avg_lick_corrs_animal, avg_neuron_corrs_dms, 'filled', 'MarkerFaceColor', color_dms, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_lick_corrs_animal', avg_neuron_corrs_dms', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d', ianimal))
end
xlabel(t, 'Lick Stability')
ylabel(t, 'Neural Stability')
title(t, 'DMS')

% Plot for DLS
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;
    is_dls = preprocessed_data(ianimal).is_dls;
    avg_lick_corrs_animal = avg_lick_corrs{ianimal};
    avg_neuron_corrs_dls = mean(avg_neuron_corrs{ianimal}(is_dls, :), 'omitnan');  % Mean over DLS neurons

    % Ensure both vectors are of the same length
    min_length = min(length(avg_lick_corrs_animal), length(avg_neuron_corrs_dls));
    avg_lick_corrs_animal = avg_lick_corrs_animal(1:min_length);
    avg_neuron_corrs_dls = avg_neuron_corrs_dls(1:min_length);

    nexttile
    scatter(avg_lick_corrs_animal, avg_neuron_corrs_dls, 'filled', 'MarkerFaceColor', color_dls, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_lick_corrs_animal', avg_neuron_corrs_dls', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d', ianimal))
end
xlabel(t, 'Lick Stability')
ylabel(t, 'Neural Stability')
title(t, 'DLS')

% Plot for ACC
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;
    is_acc = preprocessed_data(ianimal).is_acc;
    avg_lick_corrs_animal = avg_lick_corrs{ianimal};
    avg_neuron_corrs_acc = mean(avg_neuron_corrs{ianimal}(is_acc, :), 'omitnan');  % Mean over ACC neurons

    % Ensure both vectors are of the same length
    min_length = min(length(avg_lick_corrs_animal), length(avg_neuron_corrs_acc));
    avg_lick_corrs_animal = avg_lick_corrs_animal(1:min_length);
    avg_neuron_corrs_acc = avg_neuron_corrs_acc(1:min_length);

    nexttile
    scatter(avg_lick_corrs_animal, avg_neuron_corrs_acc, 'filled', 'MarkerFaceColor', color_acc, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_lick_corrs_animal', avg_neuron_corrs_acc', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d', ianimal))
end
xlabel(t, 'Lick Stability')
ylabel(t, 'Neural Stability')
title(t, 'ACC')

%% Stability of Neural Activity vs Behavioural Performance

figure

t = tiledlayout('flow', 'TileSpacing', 'compact');

zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

for ianimal = 1:n_animals
    % n_trials = preprocessed_data(ianimal).n_trials;
    n_trials = 50;

    avg_neuron_corrs_animal = mean(avg_neuron_corrs{ianimal}, 'omitnan');  % Mean over neurons
    zscored_lick_errors_animal = zscored_lick_errors_all{ianimal};

    % Ensure both vectors are of the same length
    min_length = min(length(avg_neuron_corrs_animal), length(zscored_lick_errors_animal));
    avg_neuron_corrs_animal = avg_neuron_corrs_animal(1:min_length);
    zscored_lick_errors_animal = zscored_lick_errors_animal(1:min_length);

    nexttile
    scatter(avg_neuron_corrs_animal, zscored_lick_errors_animal, 'filled', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_neuron_corrs_animal', zscored_lick_errors_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d', ianimal))
end
ylabel(t, 'Lick Error')
xlabel(t, 'Neural Stability')

zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

% Plot for DMS
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    is_dms = preprocessed_data(ianimal).is_dms;
    avg_neuron_corrs_dms = mean(avg_neuron_corrs{ianimal}(is_dms, :), 'omitnan');  % Mean over DMS neurons
    zscored_lick_errors_animal = zscored_lick_errors_all{ianimal};

    % Ensure both vectors are of the same length
    min_length = min(length(avg_neuron_corrs_dms), length(zscored_lick_errors_animal));
    avg_neuron_corrs_dms = avg_neuron_corrs_dms(1:min_length);
    zscored_lick_errors_animal = zscored_lick_errors_animal(1:min_length);

    nexttile
    scatter(avg_neuron_corrs_dms, zscored_lick_errors_animal, 'filled', 'MarkerFaceColor', color_dms, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_neuron_corrs_dms', zscored_lick_errors_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d - DMS', ianimal))
end
xlabel(t, 'Neural Stability')
ylabel(t, 'Lick Error')
title(t, 'DMS: Neural Stability vs Behavioral Performance')

% Plot for DLS
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    is_dls = preprocessed_data(ianimal).is_dls;
    avg_neuron_corrs_dls = mean(avg_neuron_corrs{ianimal}(is_dls, :), 'omitnan');  % Mean over DLS neurons
    zscored_lick_errors_animal = zscored_lick_errors_all{ianimal};

    % Ensure both vectors are of the same length
    min_length = min(length(avg_neuron_corrs_dls), length(zscored_lick_errors_animal));
    avg_neuron_corrs_dls = avg_neuron_corrs_dls(1:min_length);
    zscored_lick_errors_animal = zscored_lick_errors_animal(1:min_length);

    nexttile
    scatter(avg_neuron_corrs_dls, zscored_lick_errors_animal, 'filled', 'MarkerFaceColor', color_dls, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_neuron_corrs_dls', zscored_lick_errors_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d - DLS', ianimal))
end
xlabel(t, 'Neural Stability')
ylabel(t, 'Lick Error')
title(t, 'DLS: Neural Stability vs Behavioral Performance')

% Plot for ACC
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    is_acc = preprocessed_data(ianimal).is_acc;
    avg_neuron_corrs_acc = mean(avg_neuron_corrs{ianimal}(is_acc, :), 'omitnan');  % Mean over ACC neurons
    zscored_lick_errors_animal = zscored_lick_errors_all{ianimal};

    % Ensure both vectors are of the same length
    min_length = min(length(avg_neuron_corrs_acc), length(zscored_lick_errors_animal));
    avg_neuron_corrs_acc = avg_neuron_corrs_acc(1:min_length);
    zscored_lick_errors_animal = zscored_lick_errors_animal(1:min_length);

    nexttile
    scatter(avg_neuron_corrs_acc, zscored_lick_errors_animal, 'filled', 'MarkerFaceColor', color_acc, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_neuron_corrs_acc', zscored_lick_errors_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d - ACC', ianimal))
end
xlabel(t, 'Neural Stability')
ylabel(t, 'Lick Error')
title(t, 'ACC: Neural Stability vs Behavioral Performance')

%% Stability of Population Activity vs Behavioural Performance

zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

% Plot for DMS
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    is_dms = preprocessed_data(ianimal).is_dms;
    avg_pop_corrs_dms = avg_population_corrs{ianimal}(1, :);
    zscored_lick_errors_animal = zscored_lick_errors_all{ianimal};

    % Ensure both vectors are of the same length
    min_length = min(length(avg_pop_corrs_dms), length(zscored_lick_errors_animal));
    avg_pop_corrs_dms = avg_pop_corrs_dms(1:min_length);
    zscored_lick_errors_animal = zscored_lick_errors_animal(1:min_length);

    nexttile
    scatter(avg_pop_corrs_dms, zscored_lick_errors_animal, 'filled', 'MarkerFaceColor', color_dms, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_pop_corrs_dms', zscored_lick_errors_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d - DMS', ianimal))
end
xlabel(t, 'Neural Stability')
ylabel(t, 'Lick Error')
title(t, 'DMS: Neural Stability vs Behavioral Performance')

% Plot for DLS
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    is_dls = preprocessed_data(ianimal).is_dls;
    avg_pop_corrs_dls = avg_population_corrs{ianimal}(2, :);
    zscored_lick_errors_animal = zscored_lick_errors_all{ianimal};

    % Ensure both vectors are of the same length
    min_length = min(length(avg_pop_corrs_dls), length(zscored_lick_errors_animal));
    avg_pop_corrs_dls = avg_pop_corrs_dls(1:min_length);
    zscored_lick_errors_animal = zscored_lick_errors_animal(1:min_length);

    nexttile
    scatter(avg_pop_corrs_dls, zscored_lick_errors_animal, 'filled', 'MarkerFaceColor', color_dls, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_pop_corrs_dls', zscored_lick_errors_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d - DLS', ianimal))
end
xlabel(t, 'Neural Stability')
ylabel(t, 'Lick Error')
title(t, 'DLS: Neural Stability vs Behavioral Performance')

% Plot for ACC
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals

    avg_pop_corrs_acc = avg_population_corrs{ianimal}(3, :);
    zscored_lick_errors_animal = zscored_lick_errors_all{ianimal};

    % Ensure both vectors are of the same length
    min_length = min(length(avg_pop_corrs_acc), length(zscored_lick_errors_animal));
    avg_pop_corrs_acc = avg_pop_corrs_acc(1:min_length);
    zscored_lick_errors_animal = zscored_lick_errors_animal(1:min_length);

    nexttile
    scatter(avg_pop_corrs_acc, zscored_lick_errors_animal, 'filled', 'MarkerFaceColor', color_acc, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_pop_corrs_acc', zscored_lick_errors_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d - ACC', ianimal))
end
xlabel(t, 'Neural Stability')
ylabel(t, 'Lick Error')
title(t, 'ACC: Neural Stability vs Behavioral Performance')


%% Population vs neuron correlations

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals
    nexttile
    is_dms = preprocessed_data(ianimal).is_dms;
    avg_pop_corrs_dms = avg_population_corrs{ianimal}(1, :);
    avg_neuron_corrs_dms = mean(avg_neuron_corrs{ianimal}(is_dms, :), 'omitnan');  % Mean over ACC neurons

    scatter(avg_pop_corrs_dms, avg_neuron_corrs_dms, 'filled', 'MarkerFaceColor', color_dms, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    lsline
end
title(t, 'DMS')
xlabel(t, 'population stability')
ylabel(t, 'average neuronal stability')

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals
    nexttile
    is_dls = preprocessed_data(ianimal).is_dls;
    avg_pop_corrs_dls = avg_population_corrs{ianimal}(2, :);
    avg_neuron_corrs_dls = mean(avg_neuron_corrs{ianimal}(is_dls, :), 'omitnan');  % Mean over ACC neurons

    scatter(avg_pop_corrs_dls, avg_neuron_corrs_dls, 'filled', 'MarkerFaceColor', color_dls, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    lsline
end
title(t, 'DLS')
xlabel(t, 'population stability')
ylabel(t, 'average neuronal stability')

figure
t = tiledlayout('flow');
for ianimal = 1:n_animals
    nexttile
    is_acc = preprocessed_data(ianimal).is_acc;
    avg_pop_corrs_acc = avg_population_corrs{ianimal}(3, :);
    avg_neuron_corrs_acc = mean(avg_neuron_corrs{ianimal}(is_acc, :), 'omitnan');  % Mean over ACC neurons

    scatter(avg_pop_corrs_acc, avg_neuron_corrs_acc, 'filled', 'MarkerFaceColor', color_acc, 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    lsline
end
title(t, 'ACC')
xlabel(t, 'population stability')
ylabel(t, 'average neuronal stability')

%% Neural Stability Across Conditions for All Neurons

% Assume the following variables are available:
% - preprocessed_data: struct containing data for each animal
% - avg_neuron_corrs: cell array containing [n_neurons x n_trials] matrices per animal
% - zscored_lick_errors_all: cell array containing [1 x n_trials] vectors per animal
% - n_animals: number of animals

% Step 1: Identify trial indices for each condition
% Load z-scored lick errors for all animals
zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

% Initialize logical indices for each condition
first_idx = cell(1, n_animals);
precise_idx = cell(1, n_animals);
imprecise_idx = cell(1, n_animals);

for ianimal = 1:n_animals
    % n_trials = preprocessed_data(ianimal).n_trials;
    n_trials = 50;

    % First 3 trials
    temp_first_idx = false(1, n_trials);
    temp_first_idx(1:min(3, n_trials)) = true; % Adjust for animals with fewer than 3 trials
    first_idx{ianimal} = temp_first_idx;

    % Precise trials (z-scored lick error <= -2)
    precise_idx{ianimal} = zscored_lick_errors_all{ianimal}(1:n_trials) <= -2;
    % Exclude the first 3 trials
    precise_idx{ianimal}(1:min(3, n_trials)) = false;

    % Imprecise trials (z-scored lick error > -1)
    imprecise_idx{ianimal} = zscored_lick_errors_all{ianimal}(1:n_trials) > -1;
    % Exclude the first 3 trials
    imprecise_idx{ianimal}(1:min(3, n_trials)) = false;
end

% Define conditions
n_conditions = 3;
condition_names = {'First 3 Trials', 'Precise Trials', 'Imprecise Trials'};

% Initialize matrix to store mean neural stability per condition per animal (all neurons)
mean_neural_stability_all = nan(n_animals, n_conditions);

for ianimal = 1:n_animals
    fprintf('Processing animal %d...\n', ianimal);

    % Get avg_neuron_corrs for this animal
    avg_neuron_corrs_animal = avg_neuron_corrs{ianimal}; % [n_neurons x n_trials]

    % Get indices for conditions
    idx_first = first_idx{ianimal};
    idx_precise = precise_idx{ianimal};
    idx_imprecise = imprecise_idx{ianimal};

    % For each condition
    for icondition = 1:n_conditions
        switch icondition
            case 1 % First 3 trials
                trial_idx = idx_first;
            case 2 % Precise trials
                trial_idx = idx_precise;
            case 3 % Imprecise trials
                trial_idx = idx_imprecise;
        end

        % Neural stability across all neurons and selected trials
        neural_stability_all = avg_neuron_corrs_animal(:, trial_idx); % [n_neurons x n_trials_in_condition]
        mean_stability_all = mean(neural_stability_all, 'all', 'omitnan');
        mean_neural_stability_all(ianimal, icondition) = mean_stability_all;
    end
end

% Remove animals with NaNs in all conditions
valid_animals_all = ~all(isnan(mean_neural_stability_all), 2);
mean_neural_stability_all = mean_neural_stability_all(valid_animals_all, :);

% Step 4: Perform Repeated-Measures ANOVA

% Define the within-subjects factor
within = table({'First3'; 'Precise'; 'Imprecise'}, 'VariableNames', {'Condition'});

% All neurons
if ~isempty(mean_neural_stability_all)
    n_animals_all = size(mean_neural_stability_all, 1);

    % Create a table for All Neurons
    neural_stability_table_all = array2table(mean_neural_stability_all, 'VariableNames', {'First3', 'Precise', 'Imprecise'});
    neural_stability_table_all.AnimalID = (1:n_animals_all)';

    % Fit the repeated-measures model
    rm_all = fitrm(neural_stability_table_all, 'First3,Precise,Imprecise~1', 'WithinDesign', within);

    % Perform the ANOVA
    ranovatbl_all = ranova(rm_all);
    disp('All Neurons Neural Stability ANOVA Results:');
    disp(ranovatbl_all);

    % Post-hoc comparisons
    [all_multcompare] = multcompare(rm_all, 'Condition', 'ComparisonType', 'tukey-kramer');
    disp('All Neurons Neural Stability Post-hoc Comparisons:');
    disp(all_multcompare);

    % Step 5: Plot the results and add significance markers
    % Prepare data for plotting
    neural_stability_plot_data_all = num2cell(mean_neural_stability_all, 1);

    % Plot with error bars
    figure;
    my_errorbar_plot(neural_stability_plot_data_all);
    xticks(1:n_conditions);
    xticklabels({'First 3', 'Precise', 'Imprecise'});
    ylabel('Neural Stability');
    title('All Neurons Neural Stability Across Conditions');

    % Add significance markers
    hold on;
    comparison_pairs = [1 3; 1 2; 2 3]; % Pairs of conditions
    p_values = all_multcompare.pValue;

    which_comparison = 0;
    for i = [1, 2, 4]
        which_comparison = which_comparison + 1;
        cond1 = comparison_pairs(which_comparison, 1);
        cond2 = comparison_pairs(which_comparison, 2);
        p = p_values(i);
        if p < 0.05
            sigstar({[cond1, cond2]}, p);
        end
    end
    hold off;
end


% Initialize matrices to store mean neural stability per condition per animal per area
mean_neural_stability_dms = nan(n_animals, n_conditions);
mean_neural_stability_dls = nan(n_animals, n_conditions);
mean_neural_stability_acc = nan(n_animals, n_conditions);

for ianimal = 1:n_animals
    fprintf('Processing animal %d...\n', ianimal);

    % Get avg_neuron_corrs for this animal
    avg_neuron_corrs_animal = avg_neuron_corrs{ianimal}; % [n_neurons x n_trials]

    % Get indices for neurons in each area
    is_dms = preprocessed_data(ianimal).is_dms;
    is_dls = preprocessed_data(ianimal).is_dls;
    is_acc = preprocessed_data(ianimal).is_acc;

    % Get indices for conditions
    idx_first = first_idx{ianimal};
    idx_precise = precise_idx{ianimal};
    idx_imprecise = imprecise_idx{ianimal};

    % For each condition
    for icondition = 1:n_conditions
        switch icondition
            case 1 % First 3 trials
                trial_idx = idx_first;
            case 2 % Precise trials
                trial_idx = idx_precise;
            case 3 % Imprecise trials
                trial_idx = idx_imprecise;
        end

        % For each area
        % DMS
        if any(is_dms)
            neural_stability_dms = avg_neuron_corrs_animal(is_dms, trial_idx); % [n_dms_neurons x n_trials_in_condition]
            mean_stability_dms = mean(neural_stability_dms, 'all', 'omitnan');
            mean_neural_stability_dms(ianimal, icondition) = mean_stability_dms;
        end

        % DLS
        if any(is_dls)
            neural_stability_dls = avg_neuron_corrs_animal(is_dls, trial_idx); % [n_dls_neurons x n_trials_in_condition]
            mean_stability_dls = mean(neural_stability_dls, 'all', 'omitnan');
            mean_neural_stability_dls(ianimal, icondition) = mean_stability_dls;
        end

        % ACC
        if any(is_acc)
            neural_stability_acc = avg_neuron_corrs_animal(is_acc, trial_idx); % [n_acc_neurons x n_trials_in_condition]
            mean_stability_acc = mean(neural_stability_acc, 'all', 'omitnan');
            mean_neural_stability_acc(ianimal, icondition) = mean_stability_acc;
        end
    end
end

% Remove animals with NaNs in all conditions for each area
valid_animals_dms = ~all(isnan(mean_neural_stability_dms), 2);
valid_animals_dls = ~all(isnan(mean_neural_stability_dls), 2);
valid_animals_acc = ~all(isnan(mean_neural_stability_acc), 2);

% For each area, get the data for valid animals
mean_neural_stability_dms = mean_neural_stability_dms(valid_animals_dms, :);
mean_neural_stability_dls = mean_neural_stability_dls(valid_animals_dls, :);
mean_neural_stability_acc = mean_neural_stability_acc(valid_animals_acc, :);

% Step 4: Perform Repeated-Measures ANOVA per area

% Define the within-subjects factor
within = table({'First3'; 'Precise'; 'Imprecise'}, 'VariableNames', {'Condition'});

% DMS
if ~isempty(mean_neural_stability_dms)
    n_animals_dms = size(mean_neural_stability_dms, 1);

    % Create a table for DMS
    neural_stability_table_dms = array2table(mean_neural_stability_dms, 'VariableNames', {'First3', 'Precise', 'Imprecise'});
    neural_stability_table_dms.AnimalID = (1:n_animals_dms)';

    % Fit the repeated-measures model
    rm_dms = fitrm(neural_stability_table_dms, 'First3,Precise,Imprecise~1', 'WithinDesign', within);

    % Perform the ANOVA
    ranovatbl_dms = ranova(rm_dms);
    disp('DMS Neural Stability ANOVA Results:');
    disp(ranovatbl_dms);

    % Post-hoc comparisons
    [dms_multcompare] = multcompare(rm_dms, 'Condition', 'ComparisonType', 'tukey-kramer');
    disp('DMS Neural Stability Post-hoc Comparisons:');
    disp(dms_multcompare);

    % Step 5: Plot the results and add significance markers
    % Prepare data for plotting
    neural_stability_plot_data_dms = num2cell(mean_neural_stability_dms, 1);

    % Plot with error bars
    figure;
    my_errorbar_plot(neural_stability_plot_data_dms);
    xticks(1:n_conditions);
    xticklabels({'First 3', 'Precise', 'Imprecise'});
    ylabel('Neural Stability');
    title('DMS Neural Stability Across Conditions');

    % Add significance markers
    hold on;
    comparison_pairs = [1 3; 1 2; 2 3]; % Pairs of conditions
    p_values = dms_multcompare.pValue;

    which_comparison = 0;
    for i = [1, 2, 4]
        which_comparison = which_comparison + 1;
        cond1 = comparison_pairs(which_comparison, 1);
        cond2 = comparison_pairs(which_comparison, 2);
        p = p_values(i);
        if p < 0.05
            sigstar({[cond1, cond2]}, p);
        end
    end
    hold off;
end

% Repeat for DLS
if ~isempty(mean_neural_stability_dls)
    n_animals_dls = size(mean_neural_stability_dls, 1);

    % Create a table for DLS
    neural_stability_table_dls = array2table(mean_neural_stability_dls, 'VariableNames', {'First3', 'Precise', 'Imprecise'});
    neural_stability_table_dls.AnimalID = (1:n_animals_dls)';

    % Fit the repeated-measures model
    rm_dls = fitrm(neural_stability_table_dls, 'First3,Precise,Imprecise~1', 'WithinDesign', within);

    % Perform the ANOVA
    ranovatbl_dls = ranova(rm_dls);
    disp('DLS Neural Stability ANOVA Results:');
    disp(ranovatbl_dls);

    % Post-hoc comparisons
    [dls_multcompare] = multcompare(rm_dls, 'Condition', 'ComparisonType', 'tukey-kramer');
    disp('DLS Neural Stability Post-hoc Comparisons:');
    disp(dls_multcompare);

    % Step 5: Plot the results and add significance markers
    % Prepare data for plotting
    neural_stability_plot_data_dls = num2cell(mean_neural_stability_dls, 1);

    % Plot with error bars
    figure;
    my_errorbar_plot(neural_stability_plot_data_dls);
    xticks(1:n_conditions);
    xticklabels({'First 3', 'Precise', 'Imprecise'});
    ylabel('Neural Stability');
    title('DLS Neural Stability Across Conditions');

    % Add significance markers
    hold on;
    comparison_pairs = [1 3; 1 2; 2 3]; % Pairs of conditions
    p_values = dls_multcompare.pValue;

    which_comparison = 0;
    for i = [1, 2, 4]
        which_comparison = which_comparison + 1;
        cond1 = comparison_pairs(which_comparison, 1);
        cond2 = comparison_pairs(which_comparison, 2);
        p = p_values(i);
        if p < 0.05
            sigstar({[cond1, cond2]}, p);
        end
    end
    hold off;
end

% Repeat for ACC
if ~isempty(mean_neural_stability_acc)
    n_animals_acc = size(mean_neural_stability_acc, 1);

    % Create a table for ACC
    neural_stability_table_acc = array2table(mean_neural_stability_acc, 'VariableNames', {'First3', 'Precise', 'Imprecise'});
    neural_stability_table_acc.AnimalID = (1:n_animals_acc)';

    % Fit the repeated-measures model
    rm_acc = fitrm(neural_stability_table_acc, 'First3,Precise,Imprecise~1', 'WithinDesign', within);

    % Perform the ANOVA
    ranovatbl_acc = ranova(rm_acc);
    disp('ACC Neural Stability ANOVA Results:');
    disp(ranovatbl_acc);

    % Post-hoc comparisons
    [acc_multcompare] = multcompare(rm_acc, 'Condition', 'ComparisonType', 'tukey-kramer');
    disp('ACC Neural Stability Post-hoc Comparisons:');
    disp(acc_multcompare);

    % Step 5: Plot the results and add significance markers
    % Prepare data for plotting
    neural_stability_plot_data_acc = num2cell(mean_neural_stability_acc, 1);

    % Plot with error bars
    figure;
    my_errorbar_plot(neural_stability_plot_data_acc);
    xticks(1:n_conditions);
    xticklabels({'First 3', 'Precise', 'Imprecise'});
    ylabel('Neural Stability');
    title('ACC Neural Stability Across Conditions');

    % Add significance markers
    hold on;
    comparison_pairs = [1 3; 1 2; 2 3]; % Pairs of conditions
    p_values = acc_multcompare.pValue;

    which_comparison = 0;
    for i = [1, 2, 4]
        which_comparison = which_comparison + 1;
        cond1 = comparison_pairs(which_comparison, 1);
        cond2 = comparison_pairs(which_comparison, 2);
        p = p_values(i);
        if p < 0.05
            sigstar({[cond1, cond2]}, p);
        end
    end
    hold off;
end



%% Stability of behaviour vs performance

zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};


figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;
    avg_lick_corrs_animal = avg_lick_corrs{ianimal};  % This is now a vector\

    zscored_lick_errors_animal = zscored_lick_errors_all{ianimal};

    % Ensure both vectors are of the same length
    min_length = min(length(avg_lick_corrs_animal), length(zscored_lick_errors_animal));
    avg_lick_corrs_animal = avg_lick_corrs_animal(1:min_length);
    zscored_lick_errors_animal = zscored_lick_errors_animal(1:min_length);

    nexttile
    scatter(avg_lick_corrs_animal, zscored_lick_errors_animal, 'filled', 'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.75)
    [rho, pval] = corr(avg_lick_corrs_animal', zscored_lick_errors_animal', 'Rows', 'complete');
    lsline
    legend(sprintf('\\rho = %.2f, p = %.3f', rho, pval))
    title(sprintf('Animal %d', ianimal))
end
xlabel(t, 'Lick Stability')
ylabel(t, 'Lick Error')
%% Lick heatmap
figure
t = tiledlayout('flow');
for ianimal = 1:n_animals
% ianimal = 15;
licks_to_plot = preprocessed_data(ianimal).spatial_binned_data.licks./preprocessed_data(ianimal).spatial_binned_data.durations;
licks_to_plot(licks_to_plot > quantile(licks_to_plot, 0.99, 'all')) = nan;
nexttile
imagesc(licks_to_plot)
xlabel('spatial bin (x5cm)')
ylabel('trial')
xline([20, 25], 'Color', 'w')

title(sprintf('mouse %d, dms: %d, dls: %d, acc: %d', ianimal, sum(preprocessed_data(ianimal).is_dms),...
    sum(preprocessed_data(ianimal).is_dls), sum(preprocessed_data(ianimal).is_acc)))

end
%% Example neuron

ianimal = 10;
neuron_to_plot = squeeze(preprocessed_data(ianimal).spatial_binned_fr_all(2, :, :));
figure
imagesc(neuron_to_plot')
xline([20, 25], 'Color', 'w')
colorbar

figure
shadedErrorBar(1:num_bins, mean(neuron_to_plot, 2), sem(neuron_to_plot, 2))
xline([20, 25], 'Color', 'w')

figure
imagesc(neuron_to_plot')
xline([20, 25], 'Color', 'w')
colorbar
ylim([2.5, 7.5])
yticks(3:7)

figure
scatter(neuron_to_plot(:, 3), neuron_to_plot(:, 4), 'filled')
lsline


% Define the trials of interest
selected_trials = 3:7;

% Extract the data subset for trials 3 to 7
data_subset = neuron_to_plot(:, selected_trials); % [spatial_bins x 5]

% Compute the Pearson correlation matrix
corr_matrix = corrcoef(data_subset); % [5 x 5]

% --- Visualization Using imagesc ---
figure;
imagesc(corr_matrix);
colorbar;

% Enhance the heatmap appearance
colormap('hot'); % Choose a color map (e.g., 'jet', 'hot', 'parula')
axis square;      % Make the axes square for better visualization

xticks(1:5)
yticks(1:5)

