%% Run preprocessing analysis
clearvars -except all_data
clc

reward_zone_start_cm = 125; % in cm
reward_zone_start_au = 100;

bin_size = 4; % x1.25 = cm
bin_edges = 0:bin_size:200;
bin_edges(end) = 202;
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres);

reward_zone_start_bins = reward_zone_start_au / bin_size;

% Check if preprocessed data exists
if exist('preprocessed_data.mat', 'file')
    fprintf('Loading preprocessed data...\n');
    load('preprocessed_data.mat', 'preprocessed_data');

    n_animals = size(preprocessed_data, 2);
else
    if ~exist("all_data", 'var')
        load all_data.mat
    end

    n_animals = size(all_data, 2);

    highest_common_n_trials = min(cellfun(@max, {all_data(:).vr_trial})) - 1;

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

        % Compute firing rates for DMS, DLS, and ACC
        is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
        is_dls = strcmp(all_data(ianimal).final_areas, 'DLS');
        is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

        final_spikes_dms = all_data(ianimal).final_spikes(is_dms, :);
        final_spikes_dls = all_data(ianimal).final_spikes(is_dls, :);
        final_spikes_acc = all_data(ianimal).final_spikes(is_acc, :);

        binned_spikes_trials_dms = arrayfun(@(s,e) final_spikes_dms(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
        binned_spikes_trials_dls = arrayfun(@(s,e) final_spikes_dls(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
        binned_spikes_trials_acc = arrayfun(@(s,e) final_spikes_acc(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);

        [trial_average_fr_dms, trial_sem_fr_dms] = compute_firing_rates(binned_spikes_trials_dms, trialData.trialDurations_vr);
        [trial_average_fr_dls, trial_sem_fr_dls] = compute_firing_rates(binned_spikes_trials_dls, trialData.trialDurations_vr);
        [trial_average_fr_acc, trial_sem_fr_acc] = compute_firing_rates(binned_spikes_trials_acc, trialData.trialDurations_vr);

        % Find change points
        mov_window_size = 5;
        change_point_mean = find_change_points(trialData.trialDurations_vr, trial_metrics, mov_window_size);

        % Separate dark and corridor periods
        [darkData, corridorData] = separate_dark_and_corridor_periods(trialData, binned_spikes_trials, npx_times_trials);
        trial_lick_positions = cellfun(@(x, y) x(logical(y)), corridorData.trial_position, corridorData.trial_licks, 'UniformOutput', false);

        % Calculate lick performance
        [trial_lick_errors, shuffled_lick_error_means, shuffled_lick_error_stds, trial_lick_error_chance] = cellfun(@(x) calculate_lick_precision(x, reward_zone_start_au), trial_lick_positions);
        trial_lick_fractions = cellfun(@(x) (sum((x > reward_zone_start_au - 20) & x < reward_zone_start_au) + 1)/(sum(x > 0 & x < reward_zone_start_au)+1), trial_lick_positions);

        trial_lick_numbers = trial_metrics.trial_lick_no(1:highest_common_n_trials);
        trial_lick_errors = trial_lick_errors(1:highest_common_n_trials);
        shuffled_lick_error_means = shuffled_lick_error_means(1:highest_common_n_trials);
        shuffled_lick_error_stds = shuffled_lick_error_stds(1:highest_common_n_trials);
        outlier_trials = isoutlier(trial_lick_errors, "percentiles", [0, 99]);
        trial_lick_errors(outlier_trials) = nan;
        trial_lick_errors(1) = nan;
        shuffled_lick_error_means(outlier_trials) = nan;
        shuffled_lick_error_means(1) = nan;

        zscored_lick_errors = (trial_lick_errors - shuffled_lick_error_means)./shuffled_lick_error_stds;

        % Perform spatial binning
        spatial_binned_data = spatial_binning(corridorData, bin_edges, num_bins);

        n_units = size(darkData.binned_spikes{1}, 1);

        % Bin dark data in bins
        temporal_bin_duration = 100; % in ms
        temp_bin_edges = 1:temporal_bin_duration:5001;
        num_temp_bins = numel(temp_bin_edges) - 1;

        temp_binned_dark_spikes = nan(n_units, num_temp_bins, trialData.n_trials-1);

        for itrial = 1:highest_common_n_trials
            [~, ~, bin_idx] = histcounts(1:length(darkData.binned_spikes{itrial}), temp_bin_edges);
            for ibin = 1:num_temp_bins
                idx_in_bin = (bin_idx == ibin);
                if any(idx_in_bin)
                    % Compute total spikes
                    temp_binned_dark_spikes(:, ibin, itrial) = sum(darkData.binned_spikes{itrial}(:, idx_in_bin), 2);
                end
            end
        end
        temp_binned_dark_fr = temp_binned_dark_spikes/(temporal_bin_duration/1000);
        temp_binned_dark_fr = temp_binned_dark_fr(:, :, 1:highest_common_n_trials);

        z_temp_binned_dark_fr = zscore(temp_binned_dark_fr, [], [2, 3]);

        % Prepare data for TCA
        spatial_binned_fr_all = cat(3, spatial_binned_data.firing_rates{:});
        spatial_binned_fr_all = spatial_binned_fr_all(:, :, 1:highest_common_n_trials);

        z_spatial_binned_fr_all = zscore(spatial_binned_fr_all, [], [2, 3]);

        % Cross area pairwise correlations
        DMS_data = spatial_binned_fr_all(is_dms, :, :);
        DLS_data = spatial_binned_fr_all(is_dls, :, :);
        ACC_data = spatial_binned_fr_all(is_acc, :, :);
        n_neurons_DMS = sum(is_dms);
        n_neurons_DLS = sum(is_dls);
        n_neurons_ACC = sum(is_acc);
        n_total_neurons = size(spatial_binned_fr_all, 1);

        all_cross_area_correlations_DMSACC = nan(highest_common_n_trials, n_neurons_DMS, n_neurons_ACC);
        all_cross_area_correlations_DMSDLS = nan(highest_common_n_trials, n_neurons_DMS, n_neurons_DLS);

        for itrial = 1:highest_common_n_trials
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

        % Do tca
        [best_mdl, variance_explained, mean_cv_errors, sem_cv_errors] = tca_with_cv(spatial_binned_fr_all(:, :, 1:highest_common_n_trials), zscored_lick_errors, 'cp_nmu', 'min-max', 5, 10, 50);

        % Do pca
        DMS_data = spatial_binned_fr_all(is_dms, :, 1:highest_common_n_trials);
        DLS_data = spatial_binned_fr_all(is_dls, :, 1:highest_common_n_trials);
        ACC_data = spatial_binned_fr_all(is_acc, :, 1:highest_common_n_trials);
        stim_data = spatial_binned_fr_all(:, :, 1:highest_common_n_trials);

        DMS_data_reshaped = DMS_data(:, :);
        DLS_data_reshaped = DLS_data(:, :);
        ACC_data_reshaped = ACC_data(:, :);
        stim_data_reshaped = stim_data(:, :);

        [~, ~, ~, ~, explained_stim, ~] = pca(stim_data_reshaped', "Centered", true);
        cumsum_explained_stim = cumsum(explained_stim);
        stim_dimensionality_all = find(cumsum_explained_stim >= 90, 1)/(n_total_neurons);

        [~, ~, ~, ~, explained_stim_DMS, ~] = pca(DMS_data_reshaped', "Centered", true);
        cumsum_explained_stim_DMS = cumsum(explained_stim_DMS);
        stim_dimensionality_DMS = find(cumsum_explained_stim_DMS >= 90, 1)/(n_neurons_DMS);

        [~, ~, ~, ~, explained_stim_DLS, ~] = pca(DLS_data_reshaped', "Centered", true);
        cumsum_explained_stim_DLS = cumsum(explained_stim_DLS);
        stim_dimensionality_DLS = find(cumsum_explained_stim_DLS >= 90, 1)/(n_neurons_DLS);

        [~, ~, ~, ~, explained_stim_ACC, ~] = pca(ACC_data_reshaped', "Centered", true);
        cumsum_explained_stim_ACC = cumsum(explained_stim_ACC);
        stim_dimensionality_ACC = find(cumsum_explained_stim_ACC >= 90, 1)/(n_neurons_ACC);

        % Calculate generalised variance
        generalized_variances_stim = zeros(1, highest_common_n_trials);
        frobenius_norms_stim = zeros(1, highest_common_n_trials);

        generalized_variances_dark = zeros(1, highest_common_n_trials);
        frobenius_norms_dark = zeros(1, highest_common_n_trials);

        for itrial = 1:highest_common_n_trials
            % Extract z-scored data for trial t
            X = z_spatial_binned_fr_all(:, :, itrial);

            % Perform SVD
            [~, S, ~] = svd(X, 'econ');

            % Extract singular values
            singular_values = diag(S);

            % Compute generalized variance
            generalized_variances_stim(itrial) = sum(log(singular_values .^ 2))/n_total_neurons;

            % Compute Frobenius norm
            frobenius_norms_stim(itrial) = norm(X, 'fro')^2/n_total_neurons;


            % Same as above for dark
            X = z_temp_binned_dark_fr(:, :, itrial);
            [~, S, ~] = svd(X, 'econ');
            singular_values = diag(S);
            generalized_variances_dark(itrial) = sum(log(singular_values .^ 2))/n_total_neurons;
            frobenius_norms_dark(itrial) = norm(X, 'fro')^2/n_total_neurons;

        end


        % Store all relevant variables into preprocessed_data struct
        preprocessed_data(ianimal).trialData = trialData;
        preprocessed_data(ianimal).is_dms = is_dms;
        preprocessed_data(ianimal).is_dls = is_dls;
        preprocessed_data(ianimal).is_acc = is_acc;
        preprocessed_data(ianimal).binned_spikes_trials = binned_spikes_trials;
        preprocessed_data(ianimal).npx_times_trials = npx_times_trials;
        preprocessed_data(ianimal).trial_metrics = trial_metrics;
        preprocessed_data(ianimal).trial_average_fr_dms = trial_average_fr_dms;
        preprocessed_data(ianimal).trial_sem_fr_dms = trial_sem_fr_dms;
        preprocessed_data(ianimal).trial_average_fr_dls = trial_average_fr_dls;
        preprocessed_data(ianimal).trial_sem_fr_dls = trial_sem_fr_dls;
        preprocessed_data(ianimal).trial_average_fr_acc = trial_average_fr_acc;
        preprocessed_data(ianimal).trial_sem_fr_acc = trial_sem_fr_acc;
        preprocessed_data(ianimal).change_point_mean = change_point_mean;
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



        fprintf('Done with animal %d\n', ianimal);
    end


    for ianimal = 1:n_animals
        spatial_binned_fr_all = preprocessed_data(ianimal).spatial_binned_fr_all;
        zscored_lick_errors = preprocessed_data(ianimal).zscored_lick_errors;
        is_dms = preprocessed_data(ianimal).is_dms;
        is_dls = preprocessed_data(ianimal).is_dls;
        is_acc = preprocessed_data(ianimal).is_acc;
        [best_mdl_dms, variance_explained_dms, mean_cv_errors_dms, sem_cv_errors_dms] = tca_with_cv(spatial_binned_fr_all(is_dms, :, 1:highest_common_n_trials), zscored_lick_errors, 'cp_nmu', 'min-max', 5, 10, 50);
        preprocessed_data(ianimal).tca_best_mdl_dms = best_mdl_dms;

        try
            [best_mdl_dls, variance_explained_dls, mean_cv_errors_dls, sem_cv_errors_dls] = tca_with_cv(spatial_binned_fr_all(is_dls, :, 1:highest_common_n_trials), zscored_lick_errors, 'cp_nmu', 'min-max', 5, 10, 50);
            preprocessed_data(ianimal).tca_best_mdl_dls = best_mdl_dls;
        catch

        end

        [best_mdl_acc, variance_explained_acc, mean_cv_errors_acc, sem_cv_errors_acc] = tca_with_cv(spatial_binned_fr_all(is_acc, :, 1:highest_common_n_trials), zscored_lick_errors, 'cp_nmu', 'min-max', 5, 10, 50);
        preprocessed_data(ianimal).tca_best_mdl_acc = best_mdl_acc;
    end


    % Save the preprocessed data struct
    save('preprocessed_data.mat', 'preprocessed_data', '-v7.3');
end


highest_common_n_trials = min(cellfun(@length, {preprocessed_data(:).mean_abs_cross_area_corr_DMSDLS}));

%% Plot area dimensionality

dimensionality_stim_all = cell2mat({preprocessed_data(:).pca_stim_dimensionality_all});
dimensionality_stim_DMS = cell2mat({preprocessed_data(:).pca_stim_dimensionality_dms});
dimensionality_stim_DLS = cell2mat({preprocessed_data(:).pca_stim_dimensionality_dls});
dimensionality_stim_ACC = cell2mat({preprocessed_data(:).pca_stim_dimensionality_acc});

area_groups = [ones(size(dimensionality_stim_DMS)), 2*ones(size(dimensionality_stim_DLS)), 3*ones(size(dimensionality_stim_ACC))];
animal_groups = [1:size(dimensionality_stim_DMS, 2), 1:size(dimensionality_stim_DLS, 2), 1:size(dimensionality_stim_ACC, 2)];

figure
my_errorbar_plot(dimensionality_stim_DMS, dimensionality_stim_DLS, dimensionality_stim_ACC)
ylabel('relative dimensionality')
xticklabels({'DMS', 'DLS', 'ACC'})
title('dimensionality during task')

[~, ~, stats] = anovan([dimensionality_stim_DMS, dimensionality_stim_DLS, dimensionality_stim_ACC]', {area_groups, animal_groups}, "varnames", {'area', 'animal'}, 'display', 'off');
[comp, ~] = multcompare(stats, 'Display','off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6))

save_to_svg('area_stim_dimensionality')



%% Dark dimensionality

for ianimal = 1:n_animals
    % Do pca

    is_dms = preprocessed_data(ianimal).is_dms;
    is_dls = preprocessed_data(ianimal).is_dls;
    is_acc = preprocessed_data(ianimal).is_acc;

    n_neurons_DMS = sum(is_dms);
    n_neurons_DLS = sum(is_dls);
    n_neurons_ACC = sum(is_acc);
    n_total_neurons = size(preprocessed_data(ianimal).temp_binned_dark_fr, 1);

    DMS_data = preprocessed_data(ianimal).temp_binned_dark_fr(is_dms, :, 1:highest_common_n_trials);
    DLS_data = preprocessed_data(ianimal).temp_binned_dark_fr(is_dls, :, 1:highest_common_n_trials);
    ACC_data = preprocessed_data(ianimal).temp_binned_dark_fr(is_acc, :, 1:highest_common_n_trials);
    dark_data = preprocessed_data(ianimal).temp_binned_dark_fr(:, :, 1:highest_common_n_trials);

    DMS_data_reshaped = DMS_data(:, :);
    DLS_data_reshaped = DLS_data(:, :);
    ACC_data_reshaped = ACC_data(:, :);
    dark_data_reshaped = dark_data(:, :);

    [~, ~, ~, ~, explained_dark, ~] = pca(dark_data_reshaped', "Centered", true);
    cumsum_explained_dark = cumsum(explained_dark);
    dark_dimensionality_all = find(cumsum_explained_dark >= 90, 1)/(n_total_neurons);

    [~, ~, ~, ~, explained_dark_DMS, ~] = pca(DMS_data_reshaped', "Centered", true);
    cumsum_explained_dark_DMS = cumsum(explained_dark_DMS);
    dark_dimensionality_DMS = find(cumsum_explained_dark_DMS >= 90, 1)/(n_neurons_DMS);

    [~, ~, ~, ~, explained_dark_DLS, ~] = pca(DLS_data_reshaped', "Centered", true);
    cumsum_explained_dark_DLS = cumsum(explained_dark_DLS);
    dark_dimensionality_DLS = find(cumsum_explained_dark_DLS >= 90, 1)/(n_neurons_DLS);

    [~, ~, ~, ~, explained_dark_ACC, ~] = pca(ACC_data_reshaped', "Centered", true);
    cumsum_explained_dark_ACC = cumsum(explained_dark_ACC);
    dark_dimensionality_ACC = find(cumsum_explained_dark_ACC >= 90, 1)/(n_neurons_ACC);

    preprocessed_data(ianimal).pca_dark_dimensionality_all = dark_dimensionality_all;
    preprocessed_data(ianimal).pca_dark_dimensionality_dms = dark_dimensionality_DMS;
    preprocessed_data(ianimal).pca_dark_dimensionality_dls = dark_dimensionality_DLS;
    preprocessed_data(ianimal).pca_dark_dimensionality_acc = dark_dimensionality_ACC;

end

dimensionality_dark_all = cell2mat({preprocessed_data(:).pca_dark_dimensionality_all});
dimensionality_dark_DMS = cell2mat({preprocessed_data(:).pca_dark_dimensionality_dms});
dimensionality_dark_DLS = cell2mat({preprocessed_data(:).pca_dark_dimensionality_dls});
dimensionality_dark_ACC = cell2mat({preprocessed_data(:).pca_dark_dimensionality_acc});

area_groups = [ones(size(dimensionality_dark_DMS)), 2*ones(size(dimensionality_dark_DLS)), 3*ones(size(dimensionality_dark_ACC))];
animal_groups = [1:size(dimensionality_dark_DMS, 2), 1:size(dimensionality_dark_DLS, 2), 1:size(dimensionality_dark_ACC, 2)];

figure
my_errorbar_plot(dimensionality_dark_DMS, dimensionality_dark_DLS, dimensionality_dark_ACC)
ylabel('relative dimensionality')
xticklabels({'DMS', 'DLS', 'ACC'})
title('dimensionality during dark')

[~, ~, stats] = anovan([dimensionality_dark_DMS, dimensionality_dark_DLS, dimensionality_dark_ACC]', {area_groups, animal_groups}, "varnames", {'area', 'animal'}, 'display', 'off');
[comp, ~] = multcompare(stats, 'Display','off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6))

save_to_svg('area_dark_dimensionality')

figure
my_errorbar_plot(dimensionality_stim_all, dimensionality_dark_all, true)
[~, pval] = ttest(dimensionality_stim_all', dimensionality_dark_all');
sigstar([1, 2], pval)
xticklabels({'stim', 'dark'})
ylabel('relative dimensionality')
title('task vs dark dimensionality')

%% Plot generalised variance

generalized_variances_stim = {preprocessed_data(:).generalized_variances_stim};
generalized_variances_dark = {preprocessed_data(:).generalized_variances_dark};

zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

first_idx = cell(1, n_animals);
temp_idx = false(1, highest_common_n_trials);
temp_idx(1:3) = true;
[first_idx{:}] = deal(temp_idx);

precise_idx = cellfun(@(x) x<=-2, zscored_lick_errors_all, 'UniformOutput', false);
random_idx = cellfun(@(x) x>-2, zscored_lick_errors_all, 'UniformOutput', false);

for ianimal = 1:n_animals
    precise_idx{ianimal}(1:3) = false;
    random_idx{ianimal}(1:3) = false;
end

generalised_variance_first = cellfun(@(x, y) exp(x(y)), generalized_variances_stim, first_idx, 'UniformOutput', false);
generalised_variance_precise = cellfun(@(x, y) exp(x(y)), generalized_variances_stim, precise_idx, 'UniformOutput', false);
generalised_variance_random = cellfun(@(x, y) exp(x(y)), generalized_variances_stim, random_idx, 'UniformOutput', false);

mean_gv_first = cellfun(@mean, generalised_variance_first);
mean_gv_precise = cellfun(@mean, generalised_variance_precise);
mean_gv_random = cellfun(@mean, generalised_variance_random);

figure
my_errorbar_plot([generalised_variance_first{:}], [generalised_variance_precise{:}], [generalised_variance_random{:}])

condition_groups = [ones(1, length([generalised_variance_first{:}])), 2*ones(1, length([generalised_variance_precise{:}])), 3*ones(1, length([generalised_variance_random{:}]))];
animal_groups_first = [];
animal_groups_precise = [];
animal_groups_random = [];

for ianimal = 1:n_animals
    animal_groups_first = [animal_groups_first, ianimal*ones(size(generalised_variance_first{ianimal}))];
    animal_groups_precise = [animal_groups_precise, ianimal*ones(size(generalised_variance_precise{ianimal}))];
    animal_groups_random = [animal_groups_random, ianimal*ones(size(generalised_variance_random{ianimal}))];
end

animal_groups = [animal_groups_first, animal_groups_precise, animal_groups_random];
[~, ~, stats] = anovan([[generalised_variance_first{:}], [generalised_variance_precise{:}], [generalised_variance_random{:}]]', {condition_groups, animal_groups}, "varnames", {'condition', 'animal'}, 'display', 'on');
[comp, ~] = multcompare(stats, 'Display','off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6))


figure
my_errorbar_plot(mean_gv_first, mean_gv_precise, mean_gv_random)


figure
t = tiledlayout('flow');
for ianimal = 1:n_animals
    nexttile
    x = exp(generalized_variances_stim{ianimal});
    y = zscored_lick_errors_all{ianimal};
    outlier_trials_x = isoutlier(x, "percentiles", [0, 95]);
    outlier_trials_y = isoutlier(y, "percentiles", [0, 95]);
    x(outlier_trials_x | outlier_trials_y) = nan;
    y(outlier_trials_x | outlier_trials_y) = nan;
    y(x > 5) = nan;
    x(x > 5) = nan;

    scatter(x, y)
    
    % hold on
    % [brob, stats] = robustfit(x, y);
    % plot(x, brob(1)+brob(2)*x,'g')

    lsline
    [rho, pval] = corr(x', y', 'Rows', 'complete');
    legend(sprintf('\\rho = %.2f, pval = %.3f', rho, pval))
    title(sprintf('animal %d', ianimal))
end
ylabel(t, 'lick error')
xlabel(t, 'generalised variance')

xx = exp([generalized_variances_stim{:}]);
yy = [zscored_lick_errors_all{:}];
outlier_trials_x = isoutlier(xx, "percentiles", [0, 95]);
    outlier_trials_y = isoutlier(yy, "percentiles", [0, 95]);
    xx(outlier_trials_x | outlier_trials_y) = nan;
    yy(outlier_trials_x | outlier_trials_y) = nan;

figure
scatter(xx, yy)
lsline



% figure
% my_errorbar_plot(exp(cell2mat(generalized_variances_stim)), exp(cell2mat(generalized_variances_dark)), true)
% [~, pval] = ttest(exp(cell2mat(generalized_variances_stim)), exp(cell2mat(generalized_variances_dark)));
% sigstar([1, 2], pval)
% xticklabels({'stim', 'dark'})
% ylabel('generalised variance')


%% Cluster TCA factors based on spatial profiles

zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

first_idx = cell(1, n_animals);
temp_idx = false(1, highest_common_n_trials);
temp_idx(1:2) = true;
[first_idx{:}] = deal(temp_idx);

precise_idx = cellfun(@(x) x<=-3, zscored_lick_errors_all, 'UniformOutput', false);
random_idx = cellfun(@(x) x>0, zscored_lick_errors_all, 'UniformOutput', false);
for ianimal = 1:n_animals
    precise_idx{ianimal}(1:2) = false;
    random_idx{ianimal}(1:2) = false;
end

figure
nRows = 10; % Number of factors (adjust if necessary)
t = tiledlayout(nRows, n_animals, "TileSpacing", "compact", "TileIndexing", "columnmajor", "Padding", "compact");
for ianimal = 1:n_animals

    tca_model = preprocessed_data(ianimal).tca_best_mdl;
    n_Factors = size(tca_model.U{2}, 2);

    for iFactor = 1:n_Factors
        tile_index = (ianimal - 1) * nRows + iFactor;
        nexttile(tile_index)
        plot(tca_model.U{2}(:, iFactor))
        axis tight
        if iFactor ~= n_Factors
            xticks([])
        end
        xline(25)
        % ylabel(sprintf('Comp %d', iFactor))
        linkaxes
        xticks([])
        yticks([])
    end
    
end
xlabel(t, 'spatial bin');
ylabel(t, 'factor loading')

% Initialize variables
all_spatial_patterns = [];
pattern_labels = []; % To keep track of animal and component indices

all_trial_patterns = [];

trial_patterns_by_condition = nan(48, 3);
factor_counter = 0;

for ianimal = 1:n_animals
    tca_model = preprocessed_data(ianimal).tca_best_mdl;
    n_Factors = size(tca_model.U{2}, 2);
    
    for iFactor = 1:n_Factors
        factor_counter = factor_counter + 1;
        pattern = tca_model.U{2}(:, iFactor);
        all_spatial_patterns = [all_spatial_patterns, pattern];
        pattern_labels = [pattern_labels; struct('animal', ianimal, 'factor', iFactor)];

        trial_pattern = tca_model.U{3}(:, iFactor);
        trial_pattern = trial_pattern/max(trial_pattern);

        trial_pattern_first = mean(trial_pattern(first_idx{ianimal}));
        trial_pattern_precise = mean(trial_pattern(precise_idx{ianimal}));
        trial_pattern_random = mean(trial_pattern(random_idx{ianimal}));

        trial_patterns_by_condition(factor_counter, :) = [trial_pattern_first, trial_pattern_precise, trial_pattern_random];

        all_trial_patterns = [all_trial_patterns, trial_pattern];
    end
end

% Normalize each pattern to have unit norm
normalized_spatial_patterns = all_spatial_patterns ./ vecnorm(all_spatial_patterns);

normalized_trial_patterns = all_trial_patterns./vecnorm(all_trial_patterns);

% Compute the pairwise Pearson correlation coefficients
similarity_matrix = corrcoef(normalized_spatial_patterns);

% Perform hierarchical clustering using the distance matrix
distance_matrix = 1 - similarity_matrix; % Convert similarity to distance
Z = linkage(squareform(distance_matrix), 'average');

% Step 2: Determine the number of clusters
num_clusters = 4; % Example value, adjust as needed
cmap = lines(num_clusters); % Or define your own colors

% Step 3: Assign clusters
cluster_assignments = cluster(Z, 'maxclust', num_clusters);

% Step 4: Compute the color threshold
c_thresh = median([Z(end - num_clusters + 2, 3), Z(end - num_clusters + 1, 3)]);

% Step 5: Plot the dendrogram with colored clusters
figure;
[H, T, outperm] = dendrogram(Z, 0, 'ColorThreshold', c_thresh);
ax = gca;
set(ax, 'ColorOrder', cmap, 'NextPlot', 'replacechildren');

% Step 6: Customize the plot
xlabel('Patterns');
ylabel('Distance');
title('Hierarchical Clustering of Spatial Patterns');
% Create dummy lines for the legend
hold on;
hLegend = gobjects(num_clusters, 1);
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

% Visualize clusters
figure
t = tiledlayout(1, num_clusters, 'TileSpacing', 'compact', Padding='compact');

for iclust = 1:num_clusters
    nexttile
    hold on
    title(sprintf('Cluster %d', iclust))
    
    xline(reward_zone_start_bins, 'r')
    xline(20, 'r')
    cluster_indices = clusters{iclust};
    
    plot(normalized_spatial_patterns(:, cluster_indices), 'Color', [0.5 0.5 0.5]);
    plot(mean(normalized_spatial_patterns(:, cluster_indices), 2), 'Color', cmap(iclust, :), 'LineWidth', 2);

    hold off
    axis tight
end


xlabel(t, 'spatial bin');
ylabel(t, 'factor loading');

% save_to_svg('tca_factor_clusters')



figure
t = tiledlayout(1, num_clusters, 'TileSpacing', 'compact', Padding='compact');


for iclust = 1:num_clusters
    nexttile
    
    cluster_indices = clusters{iclust};
    plot(all_trial_patterns(:, cluster_indices))

end


%% Find how the clusters behave 
% Visualize clusters
figure
t = tiledlayout(1, num_clusters, 'TileSpacing', 'compact', Padding='compact');


for iclust = 1:num_clusters
    nexttile
    
    cluster_indices = clusters{iclust};

    data_to_plot = [trial_patterns_by_condition(cluster_indices, 1), trial_patterns_by_condition(cluster_indices, 2), trial_patterns_by_condition(cluster_indices, 3)];
    
    my_errorbar_plot(data_to_plot)

    animal_groups = repmat([pattern_labels(cluster_indices).animal]', 1, 3);
    condition_groups = repmat(1:3, numel(cluster_indices), 1);

    [~, ~, stats] = anovan(data_to_plot(:), {condition_groups(:), animal_groups(:)}, "varnames", {'condition', 'animal'}, 'display', 'on');
    [comp, ~] = multcompare(stats, 'Display','off');
    comp_groups = num2cell(comp(:, 1:2), 2);
    sig_ind = comp(:, 6) < 0.05;
    sigstar(comp_groups(sig_ind), comp(sig_ind, 6))
    xticklabels({'first 3', 'precise', 'random'})
    title(sprintf('Cluster %d', iclust))
end

ylabel(t, 'factor loading');

save_to_svg('tca_factor_cluster_behaviour')


%% Area-specific TCA


% tca_model = preprocessed_data(ianimal).tca_best_mdl;
tca_model = best_mdl;
n_Factors = size(tca_model.U{2}, 2);

figure
t = tiledlayout(n_Factors, 1);
for iFactor = 1:n_Factors
    nexttile
    plot(tca_model.U{2}(:, iFactor))
    axis tight
    if iFactor ~= n_Factors
        xticks([])
    end
    xline(25)
    % ylabel(sprintf('Comp %d', iFactor))
    linkaxes
    xticks([])
    yticks([])
end


%% Area-specific TCA plots

% Define the areas you want to include
areas = {'dms', 'dls', 'acc'}; % Adjust according to your data

% Assuming you have preprocessed_data, n_animals, highest_common_n_trials, and reward_zone_start_bins defined

% Process data
[clusters, cluster_assignments, normalized_spatial_patterns, pattern_labels, all_trial_patterns] = ...
    process_tca_data(preprocessed_data, n_animals, highest_common_n_trials, areas);

% Plot results
plot_tca_results(clusters, cluster_assignments, normalized_spatial_patterns, pattern_labels, ...
    all_trial_patterns, reward_zone_start_bins, areas);


%% Decoding of behaviour

% Assume data for multiple animals is stored in a cell array
neuronal_data_all = {preprocessed_data(:).spatial_binned_fr_all}; % Each cell contains neuronal data for one animal
lick_error_all = {preprocessed_data(:).zscored_lick_errors};    % Each cell contains lick error vector for one animal

neuron_importance_all = cell(1, n_animals);
spatial_importance_all = cell(1, n_animals);

for animal = 1:length(neuronal_data_all)
    neuronal_data = neuronal_data_all{animal};
    lick_error = lick_error_all{animal};
    
    % Preprocess data
    [n_neurons, n_spatial_bins, n_trials] = size(neuronal_data);
    X = reshape(neuronal_data, [n_neurons * n_spatial_bins, n_trials])';
    Y = lick_error';
    
    % Linear Model (LASSO)
    [B, FitInfo] = lasso(X, Y, 'CV', 5);
    idxLambdaMinMSE = FitInfo.IndexMinMSE;
    coefficients = B(:, idxLambdaMinMSE);
    coeff_matrix = reshape(coefficients, [n_neurons, n_spatial_bins]);
    neuron_importance = sum(abs(coeff_matrix), 2);
    neuron_importance = neuron_importance / sum(neuron_importance);

    spatial_importance = sum(abs(coeff_matrix), 1);
    spatial_importance = spatial_importance / sum(spatial_importance, 2);
    
    % Store results
    neuron_importance_all{animal} = neuron_importance;
    spatial_importance_all{animal} = spatial_importance;
    
end

all_spatial_importances = cat(1, spatial_importance_all{:});
figure
shadedErrorBar(1:n_spatial_bins, mean(all_spatial_importances), sem(all_spatial_importances))


%% Decoding of position

% Example usage with the enhanced decoder
options.cv_folds = 5;
options.bin_size = 5;
options.model_type = 'linear';  % or 'ridge', 'svm', 'ann', 'linear'
options.area = 'all';  % or 'DMS', 'DLS', 'ACC'
options.n_bootstraps = 100;
options.neuron_counts = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150];

% Run decoder
[decoded_positions, decoder_performance] = decode_position(preprocessed_data, options);

% Create regular visualization
visualize_decoding_results(decoded_positions, decoder_performance, options.bin_size);
sgtitle('Spatial Decoding Results')

% Create trial evolution visualization
visualize_trial_evolution(decoded_positions, decoder_performance, options.bin_size);


% figure
% scatter(cellfun(@length, {preprocessed_data(:).is_dms}), [decoder_performance.r2])
% lsline

%% TCA plotting (old)
for ianimal = 1:n_animals
    trial_lick_numbers = preprocessed_data(ianimal).trial_metrics.trial_lick_no(1:highest_common_n_trials);
    trial_lick_errors = preprocessed_data(ianimal).trial_lick_errors(1:highest_common_n_trials);
    shuffled_lick_error_means = preprocessed_data(ianimal).shuffled_lick_error_means(1:highest_common_n_trials);
    shuffled_lick_error_stds = preprocessed_data(ianimal).shuffled_lick_error_stds(1:highest_common_n_trials);
    outlier_trials = isoutlier(trial_lick_errors, "percentiles", [0, 99]);
    trial_lick_errors(outlier_trials) = nan;
    trial_lick_errors(1) = nan;
    shuffled_lick_error_means(outlier_trials) = nan;
    shuffled_lick_error_means(1) = nan;

    zscored_lick_errors = (trial_lick_errors - shuffled_lick_error_means)./shuffled_lick_error_stds;


    % Run TCA with cross-validation
    xlines_to_plot = [sum(preprocessed_data(ianimal).is_dms), sum(preprocessed_data(ianimal).is_dms) + sum(preprocessed_data(ianimal).is_dls), reward_zone_start_bins];
    tca_with_cv(preprocessed_data(ianimal).spatial_binned_fr_all(:, :, 1:highest_common_n_trials), zscored_lick_errors, 'cp_nmu', 'min-max', 5, 10, 50, xlines_to_plot);
end

%% PCA plotting

for ianimal = 1:n_animals
    % Run pca plotting ALL
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all, 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr)
    sgtitle(sprintf('all areas - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_all', ianimal))

    % Run pca plotting DMS
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dms, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dms, :, :))
    sgtitle(sprintf('DMS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_dms', ianimal))

    % Run pca plotting DLS
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dls, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dls, :, :))
    sgtitle(sprintf('DLS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_dls', ianimal))

    % Run pca plotting ACC
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_acc, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_acc, :, :))
    sgtitle(sprintf('ACC only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_acc', ianimal))
end

%% New pca plotting

for ianimal = 1:n_animals

    zscored_lick_errors = preprocessed_data(ianimal).zscored_lick_errors(1:highest_common_n_trials);

    % Run pca plotting ALL
    plot_striatum_pca_new(preprocessed_data(ianimal).spatial_binned_fr_all, 3, highest_common_n_trials, zscored_lick_errors, preprocessed_data(ianimal).temp_binned_dark_fr)
    sgtitle(sprintf('all areas - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_all', ianimal))

    % Run pca plotting DMS
    plot_striatum_pca_new(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dms, :, :), 3, highest_common_n_trials, zscored_lick_errors, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dms, :, :))
    sgtitle(sprintf('DMS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_dms', ianimal))

    % Run pca plotting DLS
    plot_striatum_pca_new(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dls, :, :), 3, highest_common_n_trials, zscored_lick_errors, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dls, :, :))
    sgtitle(sprintf('DLS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_dls', ianimal))

    % Run pca plotting ACC
    plot_striatum_pca_new(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_acc, :, :), 3, highest_common_n_trials, zscored_lick_errors, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_acc, :, :))
    sgtitle(sprintf('ACC only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_acc', ianimal))
end


%% Behaviour plotting

for ianimal = 1:n_animals
    trial_lick_numbers = preprocessed_data(ianimal).trial_metrics.trial_lick_no(1:highest_common_n_trials);

    zscored_lick_errors = preprocessed_data(ianimal).zscored_lick_errors(1:highest_common_n_trials);

    trial_lick_fractions = preprocessed_data(ianimal).trial_lick_fractions(1:highest_common_n_trials); % Adjust if necessary


    % Plot the correlations between areas
    figure
    mean_corr_DMSDLS = preprocessed_data(ianimal).mean_cross_area_corr_DMSDLS(1:highest_common_n_trials);
    mean_corr_DMSACC = preprocessed_data(ianimal).mean_cross_area_corr_DMSACC(1:highest_common_n_trials);

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
    trials = 1:length(trial_lick_errors);
    shadedErrorBar(1:length(trial_lick_errors), movmean(zscored_lick_errors, mov_window_size, 'omitmissing'), movstd(zscored_lick_errors, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
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
    shadedErrorBar(1:length(trial_lick_numbers), movmean(trial_lick_numbers, mov_window_size), movstd(trial_lick_numbers, mov_window_size)/sqrt(mov_window_size))
    ylabel('lick no')
    axis tight
    % xlabel('trial #')

    subplot(5, 1, 4)
    shadedErrorBar(1:length(mean_corr_DMSDLS), movmean(mean_corr_DMSDLS, mov_window_size, 'omitmissing'), movstd(mean_corr_DMSDLS, mov_window_size, [], 1, 'omitmissing')/sqrt(mov_window_size))
    axis tight
    title('DMS-DLS correlation')
    ylabel(sprintf('\\rho'))
    subplot(5, 1, 5)
    shadedErrorBar(1:length(mean_corr_DMSACC), movmean(mean_corr_DMSACC, mov_window_size, 'omitmissing'), movstd(mean_corr_DMSACC, mov_window_size, [], 1, 'omitmissing')/sqrt(mov_window_size))
    axis tight
    title('DMS-ACC correlation')
    ylabel(sprintf('\\rho'))
    xlabel('trial #')

    sgtitle(sprintf('animal %d', ianimal))
    fig = gcf();
    fig.Position = [1145, 15, 560, 980];
    save_to_svg(sprintf('lickquant_animal%d', ianimal))
end


%% Trial-to-trial correlation

% Define colors for each area
color_dms = [0, 0.4470, 0.7410];       % Deep Blue for DMS
color_dls =  [0.4660, 0.6740, 0.1880];  % Forest Green for DLS
color_acc = [0.8500, 0.3250, 0.0980];  % Crimson Red for ACC

% Logical indices for each area
areas = {'DMS', 'DLS', 'ACC'};
area_colors = {color_dms, color_dls, color_acc};


for ianimal = 1:n_animals
    all_activity = preprocessed_data(ianimal).spatial_binned_fr_all;
    is_dms = preprocessed_data(ianimal).is_dms;
    is_dls = preprocessed_data(ianimal).is_dls;
    is_acc = preprocessed_data(ianimal).is_acc;

    area_indices = {is_dms, is_dls, is_acc};


    % Assuming 'data' is your 3D matrix of size [neurons x spatial_bins x trials]
    [neurons, spatial_bins, trials] = size(all_activity);
    window_size = 5;
    half_window = floor(window_size / 2);

    % Preallocate the output matrix with NaNs to handle cases with insufficient data
    avg_corrs = NaN(neurons, trials);

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
                R = corrcoef(data_block);

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

    % figure
    % % Compute cosine similarity between trials
    % for a = 1:length(areas)
    %     idx = area_indices{a};
    %     if sum(idx) > 0 % Proceed if there are neurons in this area
    %         % Extract data for the area
    %         data_area = all_activity(idx, :, :);
    %         [neurons_area, ~, ~] = size(data_area);
    %
    %         % Flatten the population activity for each trial
    %         population_activity_area = reshape(data_area, [neurons_area * spatial_bins, trials]);
    %
    %         % Normalize the population activity vectors
    %         population_activity_norms_area = sqrt(sum(population_activity_area.^2, 1));
    %         population_activity_norms_area(population_activity_norms_area == 0) = 1;
    %         population_activity_normalized_area = bsxfun(@rdivide, population_activity_area, population_activity_norms_area);
    %
    %         % Preallocate the output vector
    %         avg_cosine_similarity_area = NaN(1, trials);
    %
    %         % Loop over each trial
    %         for t = 1:trials
    %             % Define the window of trials centered on trial t
    %             trial_start = max(1, t - half_window);
    %             trial_end = min(trials, t + half_window);
    %             trials_in_window = trial_start:trial_end;
    %
    %             if length(trials_in_window) > 1
    %                 % Extract data block
    %                 data_block = population_activity_normalized_area(:, trials_in_window);
    %
    %                 % Compute cosine similarity matrix
    %                 cosine_similarity_matrix = data_block' * data_block;
    %
    %                 % Extract unique pairwise similarities
    %                 upper_triangle = triu(cosine_similarity_matrix, 1);
    %                 upper_vals = upper_triangle(upper_triangle ~= 0);
    %
    %                 % Average cosine similarity
    %                 avg_cosine_similarity_area(t) = mean(upper_vals);
    %             else
    %                 avg_cosine_similarity_area(t) = NaN;
    %             end
    %         end
    %
    %         % Plot the average cosine similarity over trials for the area
    %         subplot(3, 1, a)
    %         plot(avg_cosine_similarity_area, 'LineWidth', 2, 'Color', area_colors{a})
    %         ylabel('average cosine similarity')
    %         title(areas{a})
    %         xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k')
    %         axis tight
    %         box off
    %     end
    % end
    % xlabel('trials')
    % sgtitle(sprintf('cosine similarity - animal %d', ianimal))
    % fig = gcf();
    % fig.Position = [1145, 15, 560, 980];
    % save_to_svg(sprintf('cosine similarity - animal %d', ianimal))




    % figure
    % subplot(3, 2, [1, 3, 5])
    % imagesc(avg_corrs)
    % ylabel('neurons')
    % xlabel('trials')
    % yline(sum(is_dms), 'Color', 'w', 'LineWidth', 1)
    % yline(sum(is_dms) + sum(is_dls), 'Color', 'w', 'LineWidth', 1)
    % colorbar
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1)
    %
    % subplot(3, 2, 2)
    % shadedErrorBar(1:trials, mean(avg_corrs(is_dms, :), 'omitmissing'), sem(avg_corrs(is_dms, :)), 'lineprops', {'Color', color_dms})
    % ylabel('corr')
    % xlabel('trials')
    % title('DMS only')
    % xline(preprocessed_data(ianimal).change_point_mean)
    % axis tight
    % ylim([-0.05, 0.5])
    %
    % subplot(3, 2, 4)
    % shadedErrorBar(1:trials, mean(avg_corrs(is_dls, :), 'omitmissing'), sem(avg_corrs(is_dls, :)), 'lineprops', {'Color', color_dls})
    % ylabel('corr')
    % xlabel('trials')
    % title('DLS only')
    % xline(preprocessed_data(ianimal).change_point_mean)
    % axis tight
    % ylim([-0.05, 0.5])
    %
    % subplot(3, 2, 6)
    % shadedErrorBar(1:trials, mean(avg_corrs(is_acc, :), 'omitmissing'), sem(avg_corrs(is_acc, :)), 'lineprops', {'Color', color_acc})
    % ylabel('corr')
    % xlabel('trials')
    % title('ACC only')
    % xline(preprocessed_data(ianimal).change_point_mean)
    % axis tight
    % ylim([-0.05, 0.5])
    %
    % sgtitle(sprintf('average trial-to-trial correlation - animal %d', ianimal))
    % fig = gcf();
    % fig.Position = [933, 11, 750, 950];
    %
    % save_to_svg(sprintf('average trial-to-trial correlation - animal %d', ianimal))



    % % Calculate the generalised variance for each trial, for each area separately
    % % Define regularization parameter
    % epsilon = 1e-4; % Small positive constant
    %
    % % Initialize arrays to store log-determinant values
    % logdet_dms = NaN(1, trials);
    % logdet_dls = NaN(1, trials);
    % logdet_acc = NaN(1, trials);
    %
    % logdet_values = {logdet_dms, logdet_dls, logdet_acc};
    %
    % for a = 1:length(areas)
    %     idx = area_indices{a};
    %     if sum(idx) > 0 % Proceed if there are neurons in this area
    %         neurons_in_area = sum(idx);
    %         logdet_area = NaN(1, trials); % Initialize array for this area
    %
    %         for t = 1:trials
    %             % Extract data for neurons in this area at trial t
    %             data_area = all_activity(idx, :, t); % Size: [neurons_in_area x spatial_bins]
    %
    %             % Transpose data so that rows are observations (spatial bins)
    %             % and columns are variables (neurons)
    %             data_area_t = data_area'; % Size: [spatial_bins x neurons_in_area]
    %
    %             % Check if there are enough observations to compute covariance
    %             if size(data_area_t, 1) >= size(data_area_t, 2)
    %                 % Compute covariance matrix
    %                 cov_matrix = cov(data_area_t);
    %             else
    %                 % If not enough observations, add zeros to match dimensions
    %                 padding = zeros(size(data_area_t, 2) - size(data_area_t, 1), size(data_area_t, 2));
    %                 cov_matrix = cov([data_area_t; padding]);
    %             end
    %
    %             % Regularize covariance matrix
    %             cov_matrix_regularized = cov_matrix + epsilon * eye(size(cov_matrix));
    %
    %             % Ensure the covariance matrix is positive definite
    %             % Compute eigenvalues
    %             eigenvalues = eig(cov_matrix_regularized);
    %
    %             % Handle non-positive eigenvalues (should not occur with regularization)
    %             eigenvalues(eigenvalues <= 0) = epsilon;
    %
    %             % Compute log-determinant
    %             logdet = sum(log(eigenvalues));
    %
    %             % Store the log-determinant value
    %             logdet_area(t) = logdet;
    %         end
    %
    %         % Save the array of log-determinant values
    %         logdet_values{a} = logdet_area;
    %     end
    % end

    % % Extract the computed log-determinant arrays
    % logdet_dms = logdet_values{1};
    % logdet_dls = logdet_values{2};
    % logdet_acc = logdet_values{3};
    %
    % % Plotting the results for each area
    % figure;
    %
    % % Plot for DMS
    % subplot(3, 1, 1);
    % plot(logdet_dms, 'LineWidth', 1, 'Color', color_dms);
    % ylabel('Log-Determinant');
    % title('DMS');
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k');
    % axis tight
    % box off
    %
    % % Plot for DLS
    % subplot(3, 1, 2);
    % plot(logdet_dls, 'LineWidth', 1, 'Color', color_dls);
    % ylabel('Log-Determinant');
    % title('DLS');
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k');
    % axis tight
    % box off
    %
    % % Plot for ACC
    % subplot(3, 1, 3);
    % plot(logdet_acc, 'LineWidth', 1, 'Color', color_acc);
    % xlabel('Trials');
    % ylabel('Log-Determinant');
    % title('ACC');
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k');
    % axis tight
    % box off
    %
    % sgtitle(sprintf('Log-Determinant of Covariance Matrix - Animal %d', ianimal));
    % fig = gcf();
    % fig.Position = [1145, 15, 560, 980];

end

%% Factorisation of licks

num_patterns = 4;
% Set options for reproducibility and convergence criteria
options = statset('MaxIter', 1000, 'Display', 'off');

for ianimal = 1:n_animals
    trial_lick_positions = preprocessed_data(ianimal).spatial_binned_data.licks(1:end-1, :);

    % Perform NMF on the entire dataset with the chosen number of factors
    [W, H] = nnmf(trial_lick_positions, num_patterns, 'algorithm', 'mult', 'options', options, 'replicates', 10);

    % Compute contribution of each factor
    factor_contributions = zeros(num_patterns, 1);
    for ipattern = 1:num_patterns
        % Reconstruct data using only this factor
        L_approx = W(:, ipattern) * H(ipattern, :);
        % Compute its Frobenius norm (squared)
        factor_contributions(ipattern) = norm(L_approx, 'fro')^2;
    end

    % Normalize contributions to sum to 1 (percentage)
    factor_contributions = factor_contributions / sum(factor_contributions);

    % Order factors by their contributions
    [sorted_contributions, sort_idx] = sort(factor_contributions, 'descend');
    W = W(:, sort_idx);
    H = H(sort_idx, :);

    % Plot the spatial factors ordered by contribution
    figure
    for ipattern = 1:num_patterns
        subplot(num_patterns, 1, ipattern)
        plot(H(ipattern, :))
        title(sprintf('Factor %d (%.2f%%)', ipattern, sorted_contributions(ipattern) * 100))
        ylabel('Intensity')
        xline(reward_zone_start_bins)
    end
    xlabel('Spatial Bins')
    sgtitle(sprintf('Licking Patterns NMF - Animal %d', ianimal))

    % Plot the trial factors ordered by contribution
    figure
    for ipattern = 1:num_patterns
        subplot(num_patterns, 1, ipattern)
        shadedErrorBar(1:size(trial_lick_positions, 1), movmean(W(:, ipattern), 10), movstd(W(:, ipattern), 10)/sqrt(10))
        title(sprintf('Factor %d (%.2f%%)', ipattern, sorted_contributions(ipattern) * 100))
        ylabel('Intensity')
    end
    xlabel('Trials')
    sgtitle(sprintf('Licking Patterns NMF - Animal %d', ianimal))
end