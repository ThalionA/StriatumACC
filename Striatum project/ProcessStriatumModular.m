%% Load

if ~exist("all_data", 'var')
    load all_data.mat
end

%% Run analysis
clearvars -except all_data
clc

n_animals = size(all_data, 2);

reward_zone_start_cm = 125; % in 
reward_zone_start_au = 100;

bin_size = 4; % x1.25 = cm
bin_edges = 0:bin_size:200;
bin_edges(end) = 202;
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres);

reward_zone_start_bins = reward_zone_start_au / bin_size;

for ianimal = 1:n_animals
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

    % Compute firing rates for DMS and ACC
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
    [trial_lick_errors, shuffled_lick_error_means, shuffled_lick_error_sems] = cellfun(@(x) calculate_lick_precision(x, reward_zone_start_au), trial_lick_positions);
    trial_lick_fractions = cellfun(@(x) (sum((x > reward_zone_start_au - 20) & x < reward_zone_start_au) + 1)/(sum(x > 0 & x < reward_zone_start_au)+1), trial_lick_positions); 
    figure
    subplot(3, 1, 1)
    shadedErrorBar(1:trialData.n_trials, movmean(trial_lick_fractions, mov_window_size, 'omitmissing'), movstd(trial_lick_fractions, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    xline(change_point_mean)
    ylabel('precise lick fraction')
    axis tight
    subplot(3, 1, 2)
    shadedErrorBar(1:trialData.n_trials, movmean(trial_lick_errors, mov_window_size, 'omitmissing'), movstd(trial_lick_errors, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    hold on
    shadedErrorBar(1:trialData.n_trials, movmean(shuffled_lick_error_means, mov_window_size, 'omitmissing'), movstd(shuffled_lick_error_means, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size), 'lineprops', {'Color', 'r'})
    xline(change_point_mean)
    ylabel('lick error')
    axis tight
    subplot(3, 1, 3)
    shadedErrorBar(1:trialData.n_trials, movmean(trial_metrics.trial_lick_no, mov_window_size), movstd(trial_metrics.trial_lick_no, mov_window_size)/sqrt(mov_window_size))
    ylabel('lick no')
    xline(change_point_mean)
    axis tight

    % Perform spatial binning
    spatial_binned_data = spatial_binning(corridorData, bin_edges, num_bins);

    n_units = size(darkData.binned_spikes{1}, 1);

    % Bin dark data in bins
    temporal_bin_duration = 100; % in ms
    temp_bin_edges = 1:temporal_bin_duration:5001;
    num_temp_bins = numel(temp_bin_edges) - 1;

    temp_binned_dark_spikes = nan(n_units, num_temp_bins, trialData.n_trials-1);

    for itrial = 1:trialData.n_trials
        [~, ~, bin_idx] = histcounts(1:length(darkData.binned_spikes{itrial}), temp_bin_edges);
        for ibin = 1:num_temp_bins
            idx_in_bin = (bin_idx == ibin);
            if any(idx_in_bin)
                % Compute total licks
                temp_binned_dark_spikes(:, ibin, itrial) = sum(darkData.binned_spikes{itrial}(:, idx_in_bin), 2);
            end
        end
    end
    temp_binned_dark_fr = temp_binned_dark_spikes/(temporal_bin_duration/1000);
    temp_binned_dark_fr = temp_binned_dark_fr(:, :, 1:end-1);

    % Prepare data for TCA
    spatial_binned_fr_all = cat(3, spatial_binned_data.firing_rates{:});
    spatial_binned_fr_all = spatial_binned_fr_all(:, :, 1:trialData.n_trials-1);

    z_spatial_binned_fr_all = zscore(spatial_binned_fr_all, [], [2, 3]);

    % Run TCA with cross-validation
    xlines_to_plot = [sum(is_dms), reward_zone_start_bins, change_point_mean];
    % tca_with_cv(spatial_binned_fr_all, 'cp_nmu', 'none', 5, 10, 100, xlines_to_plot);

    plot_striatum_pca(spatial_binned_fr_all, 3, change_point_mean, temp_binned_dark_fr)

    spatial_binned_fr_reshaped = spatial_binned_fr_all(:, :);
    [~, score, ~, ~, explained, ~] = pca(spatial_binned_fr_reshaped', "NumComponents", 3, "Centered", true);
    score_reshaped = permute(reshape(score, [num_bins, trialData.n_trials-1, 3]), [3, 1, 2]);
    % trial_population_variances = estimate_trialwise_variance(score_reshaped, change_point_mean);


    % Cross area pairwise correlations
    DMS_data = spatial_binned_fr_all(is_dms, :, :);
    DLS_data = spatial_binned_fr_all(is_dls, :, :);
    ACC_data = spatial_binned_fr_all(is_acc, :, :);
    n_neurons_DMS = sum(is_dms);
    n_neurons_DLS = sum(is_dls);
    n_neurons_ACC = sum(is_acc);
    
    all_cross_area_correlations_DMSACC = nan(trialData.n_trials-1, n_neurons_DMS, n_neurons_ACC);
    all_cross_area_correlations_DMSDLS = nan(trialData.n_trials-1, n_neurons_DMS, n_neurons_DLS);

    for itrial = 1:trialData.n_trials-1
        % Get trial data
        DMS_trial = squeeze(DMS_data(:, :, itrial)); % [n_neurons_DMS x n_spatial_bins]
        DLS_trial = squeeze(DLS_data(:, :, itrial)); % [n_neurons_DMS x n_spatial_bins]
        ACC_trial = squeeze(ACC_data(:, :, itrial)); % [n_neurons_ACC x n_spatial_bins]
    
        % Compute mean firing rates
        for iNeuron_DMS = 1:n_neurons_DMS
            for iNeuron_ACC = 1:n_neurons_ACC
                all_cross_area_correlations_DMSACC(itrial, iNeuron_DMS, iNeuron_ACC) = corr(DMS_trial(iNeuron_DMS, :)', ACC_trial(iNeuron_ACC, :)');
            end

            for iNeuronDLS = 1:n_neurons_DLS
                all_cross_area_correlations_DMSDLS(itrial, iNeuron_DMS, iNeuronDLS) = corr(DMS_trial(iNeuron_DMS, :)', DLS_trial(iNeuronDLS, :)');
            end

        end
        
    end
    mean_cross_area_corr_DMSACC = squeeze(mean(all_cross_area_correlations_DMSACC, [2, 3], 'omitmissing'));
    mean_cross_area_corr_DMSDLS = squeeze(mean(all_cross_area_correlations_DMSDLS, [2, 3], 'omitmissing'));

    figure
    subplot(1, 2, 1)
    scatter(mean_cross_area_corr_DMSDLS, trial_lick_fractions(1:end-1))
    xlabel('cross-area correlation')
    ylabel('precise lick fraction')
    title('DMS-DLS')
    axis tight
    lsline
    subplot(1, 2, 2)
    scatter(mean_cross_area_corr_DMSACC, trial_lick_fractions(1:end-1))
    xlabel('cross-area correlation')
    ylabel('precise lick fraction')
    title('DMS-ACC')
    axis tight
    lsline

    fprintf('Done with animal %d\n', ianimal);

end
