%% Load

if ~exist("all_data", 'var')
    load all_data.mat
end

n_animals = size(all_data, 2);
figure
t = tiledlayout(5, n_animals, "TileIndexing", "columnmajor");

for ianimal = 1:n_animals
    
    % cut data per trial
    n_vr_datapoints = length(all_data(ianimal).corrected_vr_time);
    n_npx_datapoints = length(all_data(ianimal).npx_time);

    changeIdx_vr = [find(diff(all_data(ianimal).vr_trial) ~= 0), n_vr_datapoints];

    n_trials = numel(changeIdx_vr);

    trialStartIdx_vr = [1, changeIdx_vr(1:end-1) + 1];
    trialEndIdx_vr = changeIdx_vr;
    trialLengths_vr = changeIdx_vr - trialStartIdx_vr + 1;

    trialTimes_vr = mat2cell(all_data(ianimal).corrected_vr_time, 1, trialLengths_vr);
    trialStartTimes_vr = all_data(ianimal).corrected_vr_time(trialStartIdx_vr);
    trialEndTimes_vr = all_data(ianimal).corrected_vr_time(trialEndIdx_vr);

    trialDurations_vr = (trialEndTimes_vr - trialStartTimes_vr)/1000;
    trial_licks = mat2cell(all_data(ianimal).corrected_licks, 1, trialLengths_vr);
    trial_position = mat2cell(all_data(ianimal).vr_position, 1, trialLengths_vr);
    trial_reward = mat2cell(all_data(ianimal).vr_reward, 1, trialLengths_vr);
    trial_world = mat2cell(all_data(ianimal).vr_world, 1, trialLengths_vr);

    npxStartIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialStartTimes_vr, 'nearest', 'extrap');
    npxEndIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialEndTimes_vr, 'nearest', 'extrap');

    binned_spikes_trials = arrayfun(@(s,e) all_data(ianimal).final_spikes(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);

    all_data(ianimal).final_spikes_dms = all_data(ianimal).final_spikes(strcmp(all_data(ianimal).final_areas, 'DMS'), :);
    all_data(ianimal).final_spikes_acc = all_data(ianimal).final_spikes(strcmp(all_data(ianimal).final_areas, 'ACC'), :);

    binned_spikes_trials_dms = arrayfun(@(s,e) all_data(ianimal).final_spikes_dms(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
    binned_spikes_trials_acc = arrayfun(@(s,e) all_data(ianimal).final_spikes_acc(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);

    trial_lick_no = cellfun(@sum, trial_licks);

    trial_success = cellfun(@max, trial_reward);

    trial_average_fr_dms = cellfun(@(x) mean(sum(x, 2)), binned_spikes_trials_dms)./trialDurations_vr;
    trial_sem_fr_dms = cellfun(@(x) sem(sum(x, 2)), binned_spikes_trials_dms)./trialDurations_vr;

    trial_average_fr_acc = cellfun(@(x) mean(sum(x, 2)), binned_spikes_trials_acc)./trialDurations_vr;
    trial_sem_fr_acc = cellfun(@(x) sem(sum(x, 2)), binned_spikes_trials_acc)./trialDurations_vr;

    mov_window_size = 5;

    nexttile
    findpeaks(movmean(trialDurations_vr, mov_window_size), 'MinPeakProminence', 5)
    title(num2str(all_data(ianimal).mouseid))
    axis tight
    ylim([5, 40])
    ylabel('trial duration (s)')

    nexttile
    plot(movmean(trial_lick_no, mov_window_size))
    ylabel('lick #')
    axis tight
    xline(find(movmean(trial_lick_no, 10) < 20, 1))

    nexttile
    plot(movmean(trial_success, mov_window_size))
    ylabel('reward')
    axis tight
    ylim([-0.2, 1.2])
    xline(find(movmean(trial_success, 10) < 0.5, 1))

    nexttile
    shadedErrorBar(1:n_trials, movmean(trial_average_fr_dms, mov_window_size), movmean(trial_sem_fr_dms, mov_window_size))
    % plot(movmean(trial_average_fr_dms, mov_window_size))
    ylabel('DMS fr')
    axis tight

    nexttile
    shadedErrorBar(1:n_trials, movmean(trial_average_fr_acc, mov_window_size), movmean(trial_sem_fr_acc, mov_window_size))
    % plot(movmean(trial_average_fr_acc, mov_window_size))
    ylabel('ACC fr')
    axis tight
    
end

xlabel(t, 'trial #')