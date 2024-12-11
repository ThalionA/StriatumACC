%% Run preprocessing analysis
clearvars -except all_data
clc

reward_zone_start_cm = 125; % in cm
visual_zone_start_au = 80;
reward_zone_start_au = 100;
reward_zone_end_au = 135;

bin_size = 4; % x1.25 = cm
% bin_edges = 0:bin_size:200;
% bin_edges(end) = 202;

% Modified code:
bin_edges = 0:bin_size:200;
bin_edges(end) = 200 + bin_size;

bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres);

% Adjust reward zone bins accordingly
reward_zone_start_bins = reward_zone_start_au / bin_size;
reward_zone_end_bins = reward_zone_end_au / bin_size;

% Check if preprocessed data exists
if exist('Striatum project/preprocessed_data_control.mat', 'file')
    fprintf('Loading preprocessed data...\n');
    load('preprocessed_data_control.mat', 'preprocessed_data');

    n_animals = numel(preprocessed_data);
else
    if ~exist('all_data', 'var')
        load('all_data_control.mat');
    end

    n_animals = numel(all_data);

    fprintf('Processing data for all animals...\n');
    preprocessed_data = struct();

    for ianimal = 1:n_animals
        fprintf('Processing data for animal %d...\n', ianimal);

        % Cut data per trial
        trialData = cut_data_per_trial(all_data, ianimal);

        % Reorganize spikes by area
        all_data = reorganize_spikes_by_area(all_data, ianimal);

        % Align neural data
        n_npx_datapoints = length(all_data(ianimal).npx_time);
        npxStartIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialData.trialStartTimes_vr, 'nearest', 'extrap');
        npxEndIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialData.trialEndTimes_vr, 'nearest', 'extrap');

        % Extract binned spikes per trial
        [binned_spikes_trials, npx_times_trials] = extract_binned_spikes(all_data, ianimal, npxStartIdx, npxEndIdx);

        % Compute trial metrics
        trial_metrics = compute_trial_metrics(trialData);

        % Find change points
        mov_window_size = 5;
        % change_point_mean = find_change_points(trialData.trialDurations_vr, trial_metrics, mov_window_size);

        % Determine number of trials to process
        n_trials = trialData.n_trials - 1;

        % Compute firing rates for DMS, DLS, and ACC
        is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
        is_dls = strcmp(all_data(ianimal).final_areas, 'DLS');
        is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

        final_spikes_dms = all_data(ianimal).final_spikes(is_dms, :);
        final_spikes_dls = all_data(ianimal).final_spikes(is_dls, :);
        final_spikes_acc = all_data(ianimal).final_spikes(is_acc, :);

        binned_spikes_trials_dms = arrayfun(@(s, e) final_spikes_dms(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
        binned_spikes_trials_dls = arrayfun(@(s, e) final_spikes_dls(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
        binned_spikes_trials_acc = arrayfun(@(s, e) final_spikes_acc(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);

        [trial_average_fr_dms, trial_sem_fr_dms] = compute_firing_rates(binned_spikes_trials_dms, trialData.trialDurations_vr);
        [trial_average_fr_dls, trial_sem_fr_dls] = compute_firing_rates(binned_spikes_trials_dls, trialData.trialDurations_vr);
        [trial_average_fr_acc, trial_sem_fr_acc] = compute_firing_rates(binned_spikes_trials_acc, trialData.trialDurations_vr);

        % Separate dark and corridor periods
        [darkData, corridorData] = separate_dark_and_corridor_periods(trialData, binned_spikes_trials, npx_times_trials);
        trial_lick_positions = cellfun(@(x, y) x(logical(y)), corridorData.trial_position, corridorData.trial_licks, 'UniformOutput', false);

        % Calculate lick performance
        [trial_lick_errors, shuffled_lick_error_means, shuffled_lick_error_stds, ~] = cellfun(@(x) calculate_lick_precision(x, reward_zone_start_au), trial_lick_positions);
        trial_lick_fractions = cellfun(@(x) (sum((x > reward_zone_start_au - 20) & x < reward_zone_start_au) + 1) / (sum(x > 0 & x < reward_zone_start_au) + 1), trial_lick_positions);

        % Limit metrics to n_trials
        trial_lick_numbers = trial_metrics.trial_lick_no(1:n_trials);
        trial_lick_errors = trial_lick_errors(1:n_trials);
        shuffled_lick_error_means = shuffled_lick_error_means(1:n_trials);
        shuffled_lick_error_stds = shuffled_lick_error_stds(1:n_trials);
        trial_lick_fractions = trial_lick_fractions(1:n_trials);
        trial_lick_positions = trial_lick_positions(1:n_trials);

        % Handle outliers
        outlier_trials = isoutlier(trial_lick_errors, "percentiles", [0, 99]);
        trial_lick_errors(outlier_trials) = nan;
        trial_lick_errors(1) = nan;
        shuffled_lick_error_means(outlier_trials) = nan;
        shuffled_lick_error_means(1) = nan;

        zscored_lick_errors = (trial_lick_errors - shuffled_lick_error_means) ./ shuffled_lick_error_stds;

        % Perform spatial binning
        spatial_binned_data = spatial_binning(corridorData, bin_edges, num_bins);

        n_units = size(darkData.binned_spikes{1}, 1);

        % Bin dark data in bins
        temporal_bin_duration = 100; % in ms
        temp_bin_edges = 1:temporal_bin_duration:5001;
        num_temp_bins = numel(temp_bin_edges) - 1;

        temp_binned_dark_spikes = nan(n_units, num_temp_bins, n_trials);

        for itrial = 1:n_trials
            [~, ~, bin_idx] = histcounts(1:length(darkData.binned_spikes{itrial}), temp_bin_edges);
            for ibin = 1:num_temp_bins
                idx_in_bin = (bin_idx == ibin);
                if any(idx_in_bin)
                    % Compute total spikes
                    temp_binned_dark_spikes(:, ibin, itrial) = sum(darkData.binned_spikes{itrial}(:, idx_in_bin), 2);
                end
            end
        end
        temp_binned_dark_fr = temp_binned_dark_spikes / (temporal_bin_duration / 1000);

        z_temp_binned_dark_fr = zscore(temp_binned_dark_fr, [], [2, 3]);

        % Prepare data for TCA
        spatial_binned_fr_all = cat(3, spatial_binned_data.firing_rates{1:n_trials});
        spatial_binned_fr_all = spatial_binned_fr_all(:, :, 1:n_trials);

        z_spatial_binned_fr_all = nan_zscore(spatial_binned_fr_all, [2, 3]);

        % Cross-area pairwise correlations
        DMS_data = spatial_binned_fr_all(is_dms, :, :);
        DLS_data = spatial_binned_fr_all(is_dls, :, :);
        ACC_data = spatial_binned_fr_all(is_acc, :, :);
        n_neurons_DMS = sum(is_dms);
        n_neurons_DLS = sum(is_dls);
        n_neurons_ACC = sum(is_acc);
        n_total_neurons = size(spatial_binned_fr_all, 1);

        all_cross_area_correlations_DMSACC = nan(n_trials, n_neurons_DMS, n_neurons_ACC);
        all_cross_area_correlations_DMSDLS = nan(n_trials, n_neurons_DMS, n_neurons_DLS);

        for itrial = 1:n_trials
            % Get trial data
            DMS_trial = squeeze(DMS_data(:, :, itrial)); % [n_neurons_DMS x n_spatial_bins]
            DLS_trial = squeeze(DLS_data(:, :, itrial)); % [n_neurons_DLS x n_spatial_bins]
            ACC_trial = squeeze(ACC_data(:, :, itrial)); % [n_neurons_ACC x n_spatial_bins]

            % Compute correlations
            for iNeuron_DMS = 1:n_neurons_DMS
                for iNeuron_ACC = 1:n_neurons_ACC
                    all_cross_area_correlations_DMSACC(itrial, iNeuron_DMS, iNeuron_ACC) = corr(DMS_trial(iNeuron_DMS, :)', ACC_trial(iNeuron_ACC, :)');
                end

                for iNeuron_DLS = 1:n_neurons_DLS
                    all_cross_area_correlations_DMSDLS(itrial, iNeuron_DMS, iNeuron_DLS) = corr(DMS_trial(iNeuron_DMS, :)', DLS_trial(iNeuron_DLS, :)');
                end
            end
        end
        mean_cross_area_corr_DMSACC = squeeze(mean(all_cross_area_correlations_DMSACC, [2, 3], 'omitnan'));
        mean_cross_area_corr_DMSDLS = squeeze(mean(all_cross_area_correlations_DMSDLS, [2, 3], 'omitnan'));

        mean_abs_cross_area_corr_DMSACC = squeeze(mean(abs(all_cross_area_correlations_DMSACC), [2, 3], 'omitnan'));
        mean_abs_cross_area_corr_DMSDLS = squeeze(mean(abs(all_cross_area_correlations_DMSDLS), [2, 3], 'omitnan'));

        % Perform TCA
        [best_mdl, variance_explained, mean_cv_errors, sem_cv_errors] = tca_with_cv(spatial_binned_fr_all, zscored_lick_errors, 'cp_nmu', 'min-max', 5, 10, 50);

        % Perform PCA
        DMS_data_reshaped = reshape(DMS_data, n_neurons_DMS, []);
        DLS_data_reshaped = reshape(DLS_data, n_neurons_DLS, []);
        ACC_data_reshaped = reshape(ACC_data, n_neurons_ACC, []);
        stim_data_reshaped = reshape(spatial_binned_fr_all, n_total_neurons, []);

        [~, ~, ~, ~, explained_stim] = pca(stim_data_reshaped', 'Centered', true);
        stim_dimensionality_all = find(cumsum(explained_stim) >= 90, 1) / n_total_neurons;

        [~, ~, ~, ~, explained_stim_DMS] = pca(DMS_data_reshaped', 'Centered', true);
        stim_dimensionality_DMS = find(cumsum(explained_stim_DMS) >= 90, 1) / n_neurons_DMS;

        [~, ~, ~, ~, explained_stim_DLS] = pca(DLS_data_reshaped', 'Centered', true);
        stim_dimensionality_DLS = find(cumsum(explained_stim_DLS) >= 90, 1) / n_neurons_DLS;

        [~, ~, ~, ~, explained_stim_ACC] = pca(ACC_data_reshaped', 'Centered', true);
        stim_dimensionality_ACC = find(cumsum(explained_stim_ACC) >= 90, 1) / n_neurons_ACC;

        % Calculate generalized variance
        generalized_variances_stim = zeros(1, n_trials);
        generalized_variances_dark = zeros(1, n_trials);

        for itrial = 1:n_trials
            % Stimulus period
            X_stim = spatial_binned_fr_all(:, :, itrial);
            singular_values_stim = svd(X_stim, 'econ');
            generalized_variances_stim(itrial) = sum(log(singular_values_stim .^ 2)) / n_total_neurons;

            % Dark period
            X_dark = temp_binned_dark_fr(:, :, itrial);
            singular_values_dark = svd(X_dark, 'econ');
            generalized_variances_dark(itrial) = sum(log(singular_values_dark .^ 2)) / n_total_neurons;
        end

        % Store all relevant variables into preprocessed_data struct
        preprocessed_data(ianimal).trialData = trialData;
        preprocessed_data(ianimal).is_dms = is_dms;
        preprocessed_data(ianimal).is_dls = is_dls;
        preprocessed_data(ianimal).is_acc = is_acc;
        preprocessed_data(ianimal).binned_spikes_trials = binned_spikes_trials(1:n_trials);
        preprocessed_data(ianimal).npx_times_trials = npx_times_trials(1:n_trials);
        preprocessed_data(ianimal).trial_metrics = trial_metrics;
        preprocessed_data(ianimal).trial_average_fr_dms = trial_average_fr_dms(1:n_trials);
        preprocessed_data(ianimal).trial_sem_fr_dms = trial_sem_fr_dms(1:n_trials);
        preprocessed_data(ianimal).trial_average_fr_dls = trial_average_fr_dls(1:n_trials);
        preprocessed_data(ianimal).trial_sem_fr_dls = trial_sem_fr_dls(1:n_trials);
        preprocessed_data(ianimal).trial_average_fr_acc = trial_average_fr_acc(1:n_trials);
        preprocessed_data(ianimal).trial_sem_fr_acc = trial_sem_fr_acc(1:n_trials);
        % preprocessed_data(ianimal).change_point_mean = change_point_mean;
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
        preprocessed_data(ianimal).tca_best_mdl = best_mdl;
        preprocessed_data(ianimal).tca_variance_explained = variance_explained;
        preprocessed_data(ianimal).tca_mean_cv_errors = mean_cv_errors;
        preprocessed_data(ianimal).tca_sem_cv_errors = sem_cv_errors;
        preprocessed_data(ianimal).pca_explained_all = explained_stim;
        preprocessed_data(ianimal).pca_stim_dimensionality_all = stim_dimensionality_all;
        preprocessed_data(ianimal).pca_explained_dms = explained_stim_DMS;
        preprocessed_data(ianimal).pca_stim_dimensionality_dms = stim_dimensionality_DMS;
        preprocessed_data(ianimal).pca_explained_dls = explained_stim_DLS;
        preprocessed_data(ianimal).pca_stim_dimensionality_dls = stim_dimensionality_DLS;
        preprocessed_data(ianimal).pca_explained_acc = explained_stim_ACC;
        preprocessed_data(ianimal).pca_stim_dimensionality_acc = stim_dimensionality_ACC;
        preprocessed_data(ianimal).generalized_variances_stim = generalized_variances_stim;
        preprocessed_data(ianimal).generalized_variances_dark = generalized_variances_dark;
        preprocessed_data(ianimal).n_trials = n_trials;  % Store n_trials

        fprintf('Done with animal %d\n', ianimal);
    end

    % Save the preprocessed data struct
    save('preprocessed_data_control.mat', 'preprocessed_data', '-v7.3');
end

% Define colors for each area
color_dms = [0, 0.4470, 0.7410];       % Deep Blue for DMS
color_dls =  [0.4660, 0.6740, 0.1880];  % Forest Green for DLS
color_acc = [0.8500, 0.3250, 0.0980];  % Crimson Red for ACC

% Initialize logical indices for first vs rest
first_idx = cell(1, n_animals);
rest_idx = cell(1, n_animals);
for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;
    temp_first_idx = false(1, n_trials);
    temp_first_idx(1:min(3, n_trials)) = true;
    first_idx{ianimal} = temp_first_idx;
    rest_idx{ianimal} = ~temp_first_idx;
end

%% Plot area dimensionality

dimensionality_stim_all = cell2mat({preprocessed_data(:).pca_stim_dimensionality_all});
dimensionality_stim_DMS = cell2mat({preprocessed_data(:).pca_stim_dimensionality_dms});
dimensionality_stim_DLS = cell2mat({preprocessed_data(:).pca_stim_dimensionality_dls});
dimensionality_stim_ACC = cell2mat({preprocessed_data(:).pca_stim_dimensionality_acc});

area_groups = [ones(size(dimensionality_stim_DMS)), 2*ones(size(dimensionality_stim_DLS)), 3*ones(size(dimensionality_stim_ACC))];
animal_groups = [1:size(dimensionality_stim_DMS, 2), 1:size(dimensionality_stim_DLS, 2), 2:size(dimensionality_stim_ACC, 2)+1];

figure
my_errorbar_plot({dimensionality_stim_DMS, dimensionality_stim_DLS, dimensionality_stim_ACC})
ylabel('relative dimensionality')
xticklabels({'DMS', 'DLS', 'ACC'})
title('dimensionality during task')

[~, ~, stats] = anovan([dimensionality_stim_DMS, dimensionality_stim_DLS, dimensionality_stim_ACC]', {area_groups, animal_groups}, "varnames", {'area', 'animal'}, 'display', 'off');
[comp, ~] = multcompare(stats, 'Display','off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6))

save_to_svg('area_stim_dimensionality_control')


%% Dark dimensionality

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
animal_groups = [1:numel(dimensionality_dark_DMS), 1:numel(dimensionality_dark_DLS), 2:numel(dimensionality_dark_ACC)+1];

% Plot results
figure
my_errorbar_plot({dimensionality_dark_DMS, dimensionality_dark_DLS, dimensionality_dark_ACC})
ylabel('Relative Dimensionality')
xticklabels({'DMS', 'DLS', 'ACC'})
title('Dimensionality During Dark')

% Statistical analysis
[~, ~, stats] = anovan([dimensionality_dark_DMS, dimensionality_dark_DLS, dimensionality_dark_ACC]', {area_groups, animal_groups}, "varnames", {'area', 'animal'}, 'display', 'off');
[comp, ~] = multcompare(stats, 'Display','off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6))

save_to_svg('area_dark_dimensionality_control')

% Compare task vs dark dimensionality
figure
my_errorbar_plot(dimensionality_stim_all, dimensionality_dark_all, true)
[~, pval] = ttest(dimensionality_stim_all', dimensionality_dark_all');
sigstar({[1, 2]}, pval)
xticklabels({'Stim', 'Dark'})
ylabel('Relative Dimensionality')
title('Task vs Dark Dimensionality')

%% Plot Generalized Variance

genvar = cell(1, n_animals);
for ianimal = 1:n_animals
    genvar{ianimal} = estimate_trialwise_variance(preprocessed_data(ianimal).z_spatial_binned_fr_all);
end

gv_first = cellfun(@(x, y) x(y), genvar, first_idx, 'UniformOutput', false);
gv_rest = cellfun(@(x, y) x(y), genvar, rest_idx, 'UniformOutput', false);

all_gv_first = [gv_first{:}];
all_gv_rest = [gv_rest{:}];

% Extract generalized variance values for first 3 vs rest
mean_gv_first = cellfun(@(x, y) mean(x(y), 'omitnan'), genvar, first_idx, 'UniformOutput', true);
mean_gv_rest = cellfun(@(x, y) mean(x(y), 'omitnan'), genvar, rest_idx, 'UniformOutput', true);

% Plot means with error bars
figure
my_errorbar_plot(mean_gv_first, mean_gv_rest)
xticklabels({'First 3', 'Rest'})
ylabel('Generalized Variance')
title('Generalized Variance: First 3 vs Rest')

% Perform a paired t-test
[~, p_val] = ttest(mean_gv_first, mean_gv_rest);
sigstar({[1, 2]}, p_val)


%% Cluster TCA factors based on spatial profiles

num_clusters = 4; % Adjust as needed


% Initialize logical indices for first trials and rest
first_idx = cell(1, n_animals);
rest_idx = cell(1, n_animals);
for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;
    temp_first_idx = false(1, n_trials);
    temp_first_idx(1:min(3, n_trials)) = true;
    first_idx{ianimal} = temp_first_idx;
    rest_idx{ianimal} = ~temp_first_idx;
end

% Initialize variables
all_spatial_patterns = [];
pattern_labels = [];
all_trial_patterns = {};
trial_patterns_by_condition = [];
factor_counter = 0;

for ianimal = 1:n_animals
    tca_model = preprocessed_data(ianimal).tca_best_mdl;
    n_Factors = size(tca_model.U{2}, 2);
    n_trials = preprocessed_data(ianimal).n_trials;

    for iFactor = 1:n_Factors
        factor_counter = factor_counter + 1;
        pattern = tca_model.U{2}(:, iFactor);
        all_spatial_patterns = [all_spatial_patterns, pattern];
        pattern_labels = [pattern_labels; struct('animal', ianimal, 'factor', iFactor)];

        trial_pattern = tca_model.U{3}(:, iFactor);
        trial_pattern = trial_pattern / max(trial_pattern);

        % Compute mean factor loading for the first 3 trials vs the rest
        trial_pattern_first = mean(trial_pattern(first_idx{ianimal}), 'omitnan');
        trial_pattern_rest = mean(trial_pattern(rest_idx{ianimal}), 'omitnan');

        trial_patterns_by_condition(factor_counter, :) = [trial_pattern_first, trial_pattern_rest];

        all_trial_patterns{factor_counter} = trial_pattern;
    end
end

% Normalize each spatial pattern to have unit norm
normalized_spatial_patterns = all_spatial_patterns ./ vecnorm(all_spatial_patterns);

% Normalize each trial pattern individually (optional)
normalized_trial_patterns = cellfun(@(x) x ./ norm(x), all_trial_patterns, 'UniformOutput', false);

% Compute the pairwise Pearson correlation coefficients for spatial patterns
similarity_matrix = corrcoef(normalized_spatial_patterns);

% Perform hierarchical clustering using the distance matrix
distance_matrix = 1 - similarity_matrix;
Z = linkage(squareform(distance_matrix), 'average');

% Determine the number of clusters
cmap = lines(num_clusters);

% Assign clusters
cluster_assignments = cluster(Z, 'maxclust', num_clusters);

% Compute the color threshold for dendrogram
c_thresh = median([Z(end - num_clusters + 2, 3), Z(end - num_clusters + 1, 3)]);

% Plot the dendrogram with colored clusters
figure;
[H, T, outperm] = dendrogram(Z, 0, 'ColorThreshold', c_thresh);
xlabel('Patterns');
ylabel('Distance');
title('Hierarchical Clustering of Spatial Patterns');
hold on;
hLegend = gobjects(num_clusters, 1);
leg_entries = cell(num_clusters, 1);
for iclust = 1:num_clusters
    hLegend(iclust) = plot(NaN, NaN, '-', 'Color', cmap(iclust, :), 'LineWidth', 2);
    leg_entries{iclust} = sprintf('Cluster %d', iclust);
end
hold off;
legend(hLegend, leg_entries, 'Location', 'best');

% Initialize a cell array to hold clusters
clusters = cell(num_clusters, 1);

for iclust = 1:num_clusters
    % Find indices of patterns in the current cluster
    cluster_indices = find(cluster_assignments == iclust);
    clusters{iclust} = cluster_indices;

    % Get the animals represented in this cluster
    animals_in_cluster = unique([pattern_labels(cluster_indices).animal]);

    % Check if the cluster contains patterns from multiple animals
    if numel(animals_in_cluster) > 1
        fprintf('Cluster %d contains patterns from animals: %s\n', iclust, num2str(animals_in_cluster'));
    end
end

% Visualize clusters spatially
figure;
t = tiledlayout(1, num_clusters, 'TileSpacing', 'compact', 'Padding', 'compact');
for iclust = 1:num_clusters
    nexttile;
    hold on;
    title(sprintf('Cluster %d', iclust));

    xline(reward_zone_start_bins, 'r');
    xline(20, 'r');
    cluster_indices = clusters{iclust};

    plot(normalized_spatial_patterns(:, cluster_indices), 'Color', [0.5 0.5 0.5]);
    plot(mean(normalized_spatial_patterns(:, cluster_indices), 2), 'Color', cmap(iclust, :), 'LineWidth', 2);

    hold off;
    axis tight;
end
xlabel(t, 'Spatial Bin');
ylabel(t, 'Factor Loading');

% Plot trial patterns for each cluster
figure;
t = tiledlayout(1, num_clusters, 'TileSpacing', 'compact', 'Padding', 'compact');
for iclust = 1:num_clusters
    nexttile;
    hold on;
    cluster_indices = clusters{iclust};
    title(sprintf('Cluster %d', iclust));

    max_trials_cluster = max(cellfun(@length, all_trial_patterns(cluster_indices)));

    trial_pattern_matrix = nan(length(cluster_indices), max_trials_cluster);
    pattern_counter = 0;

    % Plot each trial pattern individually
    for idx = cluster_indices'
        pattern_counter = pattern_counter + 1;
        trial_pattern = all_trial_patterns{idx};
        trial_pattern_matrix(pattern_counter, 1:length(trial_pattern)) = trial_pattern;
    end
    shadedErrorBar(1:max_trials_cluster, mean(trial_pattern_matrix, 1, 'omitmissing'), ...
        sem(trial_pattern_matrix, 1), 'lineprops', {'Color', cmap(iclust, :)})

    hold off
    axis tight
end
xlabel(t, 'Trial');
ylabel(t, 'Factor Loading');

% Plot trial patterns for each cluster, averaged per mouse
figure;
t = tiledlayout(1, num_clusters, 'TileSpacing', 'compact', 'Padding', 'compact');
for iclust = 1:num_clusters
    nexttile;
    hold on;
    cluster_indices = clusters{iclust};
    title(sprintf('Cluster %d', iclust));

    % Get the animals represented in this cluster
    animals_in_cluster = unique([pattern_labels(cluster_indices).animal]);

    % Prepare colors for different animals
    num_animals = numel(animals_in_cluster);
    animal_colors = lines(num_animals);

    % For legend
    legend_entries = cell(num_animals, 1);

    for iA = 1:num_animals
        animal_id = animals_in_cluster(iA);

        % Find patterns that belong to this animal in the cluster
        is_animal = arrayfun(@(x) x.animal == animal_id, pattern_labels(cluster_indices));
        patterns_for_animal = cluster_indices(is_animal);

        % Collect trial patterns
        trial_patterns_list = all_trial_patterns(patterns_for_animal);

        % Find maximum length of trial patterns for this animal
        max_length = max(cellfun(@length, trial_patterns_list));

        % Initialize matrix to hold trial patterns
        trial_pattern_matrix = nan(numel(trial_patterns_list), max_length);

        for idx = 1:numel(trial_patterns_list)
            trial_pattern = trial_patterns_list{idx};
            len = length(trial_pattern);
            trial_pattern_matrix(idx, 1:len) = trial_pattern;
        end

        % Average over patterns (rows)
        mean_trial_pattern = mean(trial_pattern_matrix, 1, 'omitmissing');

        % Plot the averaged trial pattern
        plot(movmean(mean_trial_pattern, 5), 'Color', animal_colors(iA, :), 'LineWidth', 2);

        % For legend
        legend_entries{iA} = sprintf('Animal %d', animal_id);
    end

    legend(legend_entries, 'Location', 'best');
    hold off;
    axis tight;
end
xlabel(t, 'Trial');
ylabel(t, 'Factor Loading');

% Analyze cluster behavior - now only two conditions: First 3 vs Rest
figure;
t = tiledlayout(1, num_clusters, 'TileSpacing', 'compact', 'Padding', 'compact');

for iclust = 1:num_clusters
    nexttile;

    cluster_indices = clusters{iclust};

    data_to_plot = [trial_patterns_by_condition(cluster_indices, 1), ...
        trial_patterns_by_condition(cluster_indices, 2)];

    my_errorbar_plot(data_to_plot);

    animal_groups = repmat([pattern_labels(cluster_indices).animal]', 1, 2);
    condition_groups = repmat(1:2, numel(cluster_indices), 1);

    % Perform ANOVA
    [~, ~, stats] = anovan(data_to_plot(:), {condition_groups(:), animal_groups(:)}, ...
        'varnames', {'Condition', 'Animal'}, 'display', 'off');
    [comp, ~] = multcompare(stats, 'Display', 'off');
    comp_groups = num2cell(comp(:, 1:2), 2);
    sig_ind = comp(:, 6) < 0.05;
    sigstar(comp_groups(sig_ind), comp(sig_ind, 6));

    xticklabels({'First 3', 'Rest'});
    title(sprintf('Cluster %d', iclust));
end

ylabel(t, 'Factor Loading');

%% Area-specific TCA


for ianimal = 1:n_animals
    spatial_binned_fr_all = preprocessed_data(ianimal).spatial_binned_fr_all;
    zscored_lick_errors = preprocessed_data(ianimal).zscored_lick_errors;
    is_dms = preprocessed_data(ianimal).is_dms;
    is_dls = preprocessed_data(ianimal).is_dls;
    is_acc = preprocessed_data(ianimal).is_acc;

    % Get the number of trials for this animal
    n_trials = preprocessed_data(ianimal).n_trials;

    % TCA on DMS neurons
    if sum(is_dms) > 0
        [best_mdl_dms, variance_explained_dms, mean_cv_errors_dms, sem_cv_errors_dms] = ...
            tca_with_cv(spatial_binned_fr_all(is_dms, :, 1:n_trials), zscored_lick_errors(1:n_trials), ...
            'cp_nmu', 'min-max', 5, 10, 50);
        preprocessed_data(ianimal).tca_best_mdl_dms = best_mdl_dms;
    else
        warning('No DMS neurons found for animal %d', ianimal);
    end

    % TCA on DLS neurons
    if sum(is_dls) > 0
        try
            [best_mdl_dls, variance_explained_dls, mean_cv_errors_dls, sem_cv_errors_dls] = ...
                tca_with_cv(spatial_binned_fr_all(is_dls, :, 1:n_trials), zscored_lick_errors(1:n_trials), ...
                'cp_nmu', 'min-max', 5, 10, 50);
            preprocessed_data(ianimal).tca_best_mdl_dls = best_mdl_dls;
        catch ME
            warning('TCA computation failed for DLS neurons in animal %d: %s', ianimal, ME.message);
        end
    else
        warning('No DLS neurons found for animal %d', ianimal);
    end

    % TCA on ACC neurons
    if sum(is_acc) > 0
        [best_mdl_acc, variance_explained_acc, mean_cv_errors_acc, sem_cv_errors_acc] = ...
            tca_with_cv(spatial_binned_fr_all(is_acc, :, 1:n_trials), zscored_lick_errors(1:n_trials), ...
            'cp_nmu', 'min-max', 5, 10, 50);
        preprocessed_data(ianimal).tca_best_mdl_acc = best_mdl_acc;
    else
        warning('No ACC neurons found for animal %d', ianimal);
    end
end

% Save the preprocessed data struct
save('preprocessed_data.mat', 'preprocessed_data', '-v7.3');


%% Area-specific TCA plots

% Define the areas you want to include
areas = {'dms', 'dls', 'acc'}; % Adjust according to your data

% Assuming you have preprocessed_data, n_animals, highest_common_n_trials, and reward_zone_start_bins defined

% Process data
[clusters, cluster_assignments, normalized_spatial_patterns, pattern_labels, all_trial_patterns] = ...
    process_tca_data(preprocessed_data, n_animals, areas, 3);

% Plot results
plot_tca_results(clusters, cluster_assignments, normalized_spatial_patterns, pattern_labels, ...
    all_trial_patterns, reward_zone_start_bins, areas);

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

% Initialize logical indices for first 3 trials vs rest
first_idx = cell(1, n_animals);
rest_idx = cell(1, n_animals);

for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;

    % First 3 trials
    temp_first_idx = false(1, n_trials);
    temp_first_idx(1:min(3, n_trials)) = true; % Adjust for animals with fewer than 3 trials
    first_idx{ianimal} = temp_first_idx;
    rest_idx{ianimal} = ~temp_first_idx;
end

% Step 2: For each animal, select one neuron count and find the best bootstrap
selected_neuron_count = 50;

% Initialize arrays to store mean RMSE and R² per condition per animal
% Now we have only two conditions: first 3 and rest
n_conditions = 2;
mean_rmse_per_animal = nan(n_animals, n_conditions);
mean_r2_per_animal = nan(n_animals, n_conditions);

for ianimal = 1:n_animals
    fprintf('Processing animal %d...\n', ianimal);

    neuron_counts = decoder_performance(ianimal).neuron_counts;
    max_count_idx = find(neuron_counts == selected_neuron_count);
    if isempty(max_count_idx)
        continue
    end

    % Get number of bootstraps
    n_bootstraps = options.n_bootstraps;
    avg_rmse_per_bootstrap = nan(n_bootstraps, 1);

    % Find the best bootstrap based on lowest average RMSE
    for ibootstrap = 1:n_bootstraps
        if isempty(decoded_positions{ianimal}{max_count_idx, ibootstrap})
            continue;
        end

        decoded_positions_matrix = decoded_positions{ianimal}{max_count_idx, ibootstrap};
        [rmse_per_trial, ~] = compute_decoding_metrics(decoded_positions_matrix, options.bin_size);
        avg_rmse_per_bootstrap(ibootstrap) = mean(rmse_per_trial, 'omitnan');
    end

    if all(isnan(avg_rmse_per_bootstrap))
        warning('No valid bootstraps found for animal %d. Skipping.', ianimal);
        continue;
    end

    [~, best_bootstrap_idx] = min(avg_rmse_per_bootstrap);
    fprintf('Selected best bootstrap iteration: %d\n', best_bootstrap_idx);

    % Compute per-trial performance metrics using the best bootstrap
    decoded_positions_matrix = decoded_positions{ianimal}{max_count_idx, best_bootstrap_idx};
    [rmse_per_trial, r2_per_trial] = compute_decoding_metrics(decoded_positions_matrix, options.bin_size);

    idx_first = first_idx{ianimal};
    idx_rest = rest_idx{ianimal};

    % Compute mean RMSE and R² for first 3 vs rest
    mean_rmse_per_animal(ianimal, 1) = mean(rmse_per_trial(idx_first), 'omitnan');
    mean_rmse_per_animal(ianimal, 2) = mean(rmse_per_trial(idx_rest), 'omitnan');

    mean_r2_per_animal(ianimal, 1) = mean(r2_per_trial(idx_first), 'omitnan');
    mean_r2_per_animal(ianimal, 2) = mean(r2_per_trial(idx_rest), 'omitnan');
end

% Remove animals with NaNs in both conditions
valid_animals = ~all(isnan(mean_rmse_per_animal), 2);
mean_rmse_per_animal = mean_rmse_per_animal(valid_animals, :);
mean_r2_per_animal = mean_r2_per_animal(valid_animals, :);

% Statistical comparisons: paired t-tests
[~, p_rmse] = ttest(mean_rmse_per_animal(:,1), mean_rmse_per_animal(:,2));
[~, p_r2] = ttest(mean_r2_per_animal(:,1), mean_r2_per_animal(:,2));

% Prepare data for plotting
rmse_plot_data = {mean_rmse_per_animal(:,1), mean_rmse_per_animal(:,2)};
r2_plot_data = {mean_r2_per_animal(:,1), mean_r2_per_animal(:,2)};

% Plot RMSE
figure;
my_errorbar_plot(rmse_plot_data);
xticks(1:2);
xticklabels({'First 3', 'Rest'});
ylabel('RMSE');
title('Decoding RMSE: First 3 vs Rest');
sigstar({[1,2]}, p_rmse);

% Plot R²
figure;
my_errorbar_plot(r2_plot_data);
xticks(1:2);
xticklabels({'First 3', 'Rest'});
ylabel('R²');
title('Decoding R²: First 3 vs Rest');
sigstar({[1,2]}, p_r2);



%% PCA plotting

for ianimal = 1:n_animals
    % Run pca plotting ALL
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all, 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr)
    sgtitle(sprintf('all areas - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    % save_to_svg(sprintf('pca_3d_animal%d_all', ianimal))

    % Run pca plotting DMS
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dms, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dms, :, :))
    sgtitle(sprintf('DMS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    % save_to_svg(sprintf('pca_3d_animal%d_dms', ianimal))

    % Run pca plotting DLS
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dls, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dls, :, :))
    sgtitle(sprintf('DLS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    % save_to_svg(sprintf('pca_3d_animal%d_dls', ianimal))

    % Run pca plotting ACC
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_acc, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_acc, :, :))
    sgtitle(sprintf('ACC only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    % save_to_svg(sprintf('pca_3d_animal%d_acc', ianimal))
end

%% New pca plotting
highest_common_n_trials = 50;

for ianimal = 1:n_animals

    zscored_lick_errors = preprocessed_data(ianimal).zscored_lick_errors;

    % Run pca plotting ALL
    plot_striatum_pca_new(preprocessed_data(ianimal).spatial_binned_fr_all, 3, highest_common_n_trials, zscored_lick_errors, preprocessed_data(ianimal).temp_binned_dark_fr)
    sgtitle(sprintf('all areas - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    % save_to_svg(sprintf('pca_3d_animal%d_all', ianimal))

    % Run pca plotting DMS
    plot_striatum_pca_new(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dms, :, :), 3, highest_common_n_trials, zscored_lick_errors, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dms, :, :))
    sgtitle(sprintf('DMS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    % save_to_svg(sprintf('pca_3d_animal%d_dms', ianimal))

    % Run pca plotting DLS
    plot_striatum_pca_new(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dls, :, :), 3, highest_common_n_trials, zscored_lick_errors, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dls, :, :))
    sgtitle(sprintf('DLS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    % save_to_svg(sprintf('pca_3d_animal%d_dls', ianimal))

    % Run pca plotting ACC
    plot_striatum_pca_new(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_acc, :, :), 3, highest_common_n_trials, zscored_lick_errors, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_acc, :, :))
    sgtitle(sprintf('ACC only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    % save_to_svg(sprintf('pca_3d_animal%d_acc', ianimal))
end


%% Behaviour plotting

for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;
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

    % save_to_svg(sprintf('cross_area_lickerror_animal%d', ianimal))


    % Plot licking performance
    mov_window_size = 5;

    figure
    subplot(5, 1, 1)
    shadedErrorBar(1:length(trial_lick_fractions), movmean(trial_lick_fractions, mov_window_size, 'omitmissing'), movstd(trial_lick_fractions, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    ylabel('precise lick fraction')
    axis tight
    subplot(5, 1, 2)
    trials = preprocessed_data(ianimal).n_trials;
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
    % save_to_svg(sprintf('lickquant_animal%d', ianimal))
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

    [neurons, spatial_bins, trials] = size(all_activity);
    % trials = 50;
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
                R = corr(data_block, 'Rows', 'complete');

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
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1)

    subplot(3, 2, 2)
    shadedErrorBar(1:trials, mean(avg_corrs(is_dms, :), 'omitmissing'), sem(avg_corrs(is_dms, :)), 'lineprops', {'Color', color_dms})
    ylabel('Corr')
    xlabel('Trials')
    title('DMS Only')
    axis tight
    % ylim([-0.05, 0.5])

    if any(is_dls)
        subplot(3, 2, 4)
        shadedErrorBar(1:trials, mean(avg_corrs(is_dls, :), 'omitmissing'), sem(avg_corrs(is_dls, :)), 'lineprops', {'Color', color_dls})
        ylabel('Corr')
        xlabel('Trials')
        title('DLS Only')
        axis tight
        % ylim([-0.05, 0.5])
    end

    if any(is_acc)
        subplot(3, 2, 6)
        shadedErrorBar(1:trials, mean(avg_corrs(is_acc, :), 'omitnan'), sem(avg_corrs(is_acc, :)), 'lineprops', {'Color', color_acc})
        ylabel('Corr')
        xlabel('Trials')
        title('ACC Only')
        axis tight
    end
    % ylim([-0.05, 0.5])

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
    n_trials = preprocessed_data(ianimal).n_trials;
    % n_trials = 50;

    lick_data = preprocessed_data(ianimal).spatial_binned_data.licks(1:n_trials, :);
    occupancy_data = preprocessed_data(ianimal).spatial_binned_data.durations(1:n_trials, :);

    window_size = 5;
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

% Define two conditions: First 3 trials vs Rest
first_idx = cell(1, n_animals);
rest_idx = cell(1, n_animals);

for ianimal = 1:n_animals
    n_trials = preprocessed_data(ianimal).n_trials;
    temp_first_idx = false(1, n_trials);
    temp_first_idx(1:min(3, n_trials)) = true; % Handle if fewer than 3 trials
    first_idx{ianimal} = temp_first_idx;
    rest_idx{ianimal} = ~temp_first_idx;
end

% Initialize matrix to store mean neural stability per condition (First 3 vs Rest) per animal (all neurons)
n_conditions = 2;
mean_neural_stability_all = nan(n_animals, n_conditions);

for ianimal = 1:n_animals
    fprintf('Processing animal %d...\n', ianimal);

    % Get avg_neuron_corrs for this animal
    avg_neuron_corrs_animal = avg_neuron_corrs{ianimal}; % [n_neurons x n_trials]

    % Indices for conditions
    idx_first = first_idx{ianimal};
    idx_rest = rest_idx{ianimal};

    % Compute mean neural stability (average correlation) across all neurons for each condition
    neural_stability_first = avg_neuron_corrs_animal(:, idx_first);
    neural_stability_rest = avg_neuron_corrs_animal(:, idx_rest);

    mean_neural_stability_first = mean(neural_stability_first, 'all', 'omitnan');
    mean_neural_stability_rest = mean(neural_stability_rest, 'all', 'omitnan');

    mean_neural_stability_all(ianimal, 1) = mean_neural_stability_first;
    mean_neural_stability_all(ianimal, 2) = mean_neural_stability_rest;
end

% Remove animals with NaNs in both conditions
valid_animals_all = ~all(isnan(mean_neural_stability_all), 2);
mean_neural_stability_all = mean_neural_stability_all(valid_animals_all, :);

% Perform a paired t-test (since we have two related conditions per animal)
[~, p_all] = ttest(mean_neural_stability_all(:,1), mean_neural_stability_all(:,2));

% Plot results
figure;
my_errorbar_plot({mean_neural_stability_all(:,1), mean_neural_stability_all(:,2)});
xticks(1:n_conditions);
xticklabels({'First 3', 'Rest'});
ylabel('Neural Stability');
title('All Neurons Neural Stability: First 3 vs Rest');
sigstar({[1, 2]}, p_all);

%% Area-Specific Neural Stability

% Initialize matrices to store mean neural stability per condition per area
mean_neural_stability_dms = nan(n_animals, n_conditions);
mean_neural_stability_dls = nan(n_animals, n_conditions);
mean_neural_stability_acc = nan(n_animals, n_conditions);

for ianimal = 1:n_animals
    fprintf('Processing animal %d for areas...\n', ianimal);

    avg_neuron_corrs_animal = avg_neuron_corrs{ianimal}; % [n_neurons x n_trials]

    % Get area indices
    is_dms = preprocessed_data(ianimal).is_dms;
    is_dls = preprocessed_data(ianimal).is_dls;
    is_acc = preprocessed_data(ianimal).is_acc;

    % Indices for conditions
    idx_first = first_idx{ianimal};
    idx_rest = rest_idx{ianimal};

    % DMS
    if any(is_dms)
        neural_stability_dms_first = avg_neuron_corrs_animal(is_dms, idx_first);
        neural_stability_dms_rest = avg_neuron_corrs_animal(is_dms, idx_rest);

        mean_neural_stability_dms(ianimal, 1) = mean(neural_stability_dms_first, 'all', 'omitnan');
        mean_neural_stability_dms(ianimal, 2) = mean(neural_stability_dms_rest, 'all', 'omitnan');
    end

    % DLS
    if any(is_dls)
        neural_stability_dls_first = avg_neuron_corrs_animal(is_dls, idx_first);
        neural_stability_dls_rest = avg_neuron_corrs_animal(is_dls, idx_rest);

        mean_neural_stability_dls(ianimal, 1) = mean(neural_stability_dls_first, 'all', 'omitnan');
        mean_neural_stability_dls(ianimal, 2) = mean(neural_stability_dls_rest, 'all', 'omitnan');
    end

    % ACC
    if any(is_acc)
        neural_stability_acc_first = avg_neuron_corrs_animal(is_acc, idx_first);
        neural_stability_acc_rest = avg_neuron_corrs_animal(is_acc, idx_rest);

        mean_neural_stability_acc(ianimal, 1) = mean(neural_stability_acc_first, 'all', 'omitnan');
        mean_neural_stability_acc(ianimal, 2) = mean(neural_stability_acc_rest, 'all', 'omitnan');
    end
end

% Remove animals with NaNs for each area
valid_animals_dms = ~all(isnan(mean_neural_stability_dms), 2);
mean_neural_stability_dms = mean_neural_stability_dms(valid_animals_dms, :);

valid_animals_dls = ~all(isnan(mean_neural_stability_dls), 2);
mean_neural_stability_dls = mean_neural_stability_dls(valid_animals_dls, :);

valid_animals_acc = ~all(isnan(mean_neural_stability_acc), 2);
mean_neural_stability_acc = mean_neural_stability_acc(valid_animals_acc, :);

% Perform paired t-tests for each area
if ~isempty(mean_neural_stability_dms)
    [~, p_dms] = ttest(mean_neural_stability_dms(:,1), mean_neural_stability_dms(:,2));
    figure;
    my_errorbar_plot({mean_neural_stability_dms(:,1), mean_neural_stability_dms(:,2)});
    xticks(1:2);
    xticklabels({'First 3', 'Rest'});
    ylabel('Neural Stability');
    title('DMS Neural Stability: First 3 vs Rest');
    sigstar({[1, 2]}, p_dms);
end

if ~isempty(mean_neural_stability_dls)
    [~, p_dls] = ttest(mean_neural_stability_dls(:,1), mean_neural_stability_dls(:,2));
    figure;
    my_errorbar_plot({mean_neural_stability_dls(:,1), mean_neural_stability_dls(:,2)});
    xticks(1:2);
    xticklabels({'First 3', 'Rest'});
    ylabel('Neural Stability');
    title('DLS Neural Stability: First 3 vs Rest');
    sigstar({[1, 2]}, p_dls);
end

if ~isempty(mean_neural_stability_acc)
    [~, p_acc] = ttest(mean_neural_stability_acc(:,1), mean_neural_stability_acc(:,2));
    figure;
    my_errorbar_plot({mean_neural_stability_acc(:,1), mean_neural_stability_acc(:,2)});
    xticks(1:2);
    xticklabels({'First 3', 'Rest'});
    ylabel('Neural Stability');
    title('ACC Neural Stability: First 3 vs Rest');
    sigstar({[1, 2]}, p_acc);
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

ianimal = 1;
licks_to_plot = preprocessed_data(ianimal).spatial_binned_data.licks;
licks_to_plot(licks_to_plot > quantile(licks_to_plot, 0.99, 'all')) = nan;

figure
imagesc(licks_to_plot)
xlabel('spatial bin (x5cm)')
ylabel('trial')
% xline([20, 25], 'Color', 'w')

%% Example neuron

ianimal = 1;
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

