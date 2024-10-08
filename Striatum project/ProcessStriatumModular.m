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
    is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

    final_spikes_dms = all_data(ianimal).final_spikes(is_dms, :);
    final_spikes_acc = all_data(ianimal).final_spikes(is_acc, :);

    binned_spikes_trials_dms = arrayfun(@(s,e) final_spikes_dms(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
    binned_spikes_trials_acc = arrayfun(@(s,e) final_spikes_acc(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);

    [trial_average_fr_dms, trial_sem_fr_dms] = compute_firing_rates(binned_spikes_trials_dms, trialData.trialDurations_vr);
    [trial_average_fr_acc, trial_sem_fr_acc] = compute_firing_rates(binned_spikes_trials_acc, trialData.trialDurations_vr);

    % Find change points
    mov_window_size = 5;
    change_point_mean = find_change_points(trialData.trialDurations_vr, trial_metrics, mov_window_size);

    % Separate dark and corridor periods
    [darkData, corridorData] = separate_dark_and_corridor_periods(trialData, binned_spikes_trials, npx_times_trials);

    % Perform spatial binning
    spatial_binned_data = spatial_binning(corridorData, bin_edges, num_bins);

    temporal_bin_duration = 100;
    temp_bin_edges = 1:temporal_bin_duration:5001;
    num_temp_bins = numel(temp_bin_edges) - 1;

    

    n_units = size(darkData.binned_spikes{1}, 1);

    temp_binned_dark_spikes = nan(n_units, num_temp_bins, trialData.n_trials-1);

    for itrial = 1:trialData.n_trials
        [~, ~, bin_idx] = histcounts(1:length(darkData.binned_spikes{itrial}), temp_bin_edges);
        for ibin = 1:num_bins
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

    % Run TCA with cross-validation
    xlines_to_plot = [sum(is_dms), reward_zone_start_bins, change_point_mean];
    % tca_with_cv(spatial_binned_fr_all, 'cp_orth_als', 'z-score', 5, 8, 100, xlines_to_plot);

    plot_striatum_pca(spatial_binned_fr_all, 3, change_point_mean, temp_binned_dark_fr)

    fprintf('Done with animal %d\n', ianimal);
end
