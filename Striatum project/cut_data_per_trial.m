function trialData = cut_data_per_trial(all_data, ianimal)
    % Extracts and organizes trial-based data from all_data for a given animal.

    n_vr_datapoints = length(all_data(ianimal).corrected_vr_time);

    % Find indices where trial changes occur
    changeIdx_vr = [find(diff(all_data(ianimal).vr_trial) ~= 0), n_vr_datapoints];

    n_trials = numel(changeIdx_vr);

    trialStartIdx_vr = [1, changeIdx_vr(1:end-1) + 1];
    trialEndIdx_vr = changeIdx_vr;
    trialLengths_vr = trialEndIdx_vr - trialStartIdx_vr + 1;

    % Extract trial times and zero them
    trialTimes_vr = mat2cell(all_data(ianimal).corrected_vr_time, 1, trialLengths_vr);
    trial_times_zeroed = cellfun(@(x) x - x(1), trialTimes_vr, 'UniformOutput', false);
    trialStartTimes_vr = all_data(ianimal).corrected_vr_time(trialStartIdx_vr);
    trialEndTimes_vr = all_data(ianimal).corrected_vr_time(trialEndIdx_vr);
    trialDurations_vr = (trialEndTimes_vr - trialStartTimes_vr) / 1000;  % Convert to seconds

    % Extract other trial data
    trial_licks = mat2cell(all_data(ianimal).corrected_licks, 1, trialLengths_vr);
    trial_position = mat2cell(all_data(ianimal).vr_position, 1, trialLengths_vr);
    trial_reward = mat2cell(all_data(ianimal).vr_reward, 1, trialLengths_vr);
    trial_world = mat2cell(all_data(ianimal).vr_world, 1, trialLengths_vr);

    % Package into a struct
    trialData = struct('n_trials', n_trials, 'trialStartTimes_vr', trialStartTimes_vr, ...
        'trialEndTimes_vr', trialEndTimes_vr, 'trialDurations_vr', trialDurations_vr, ...
        'trial_times_zeroed', {trial_times_zeroed}, 'trial_licks', {trial_licks}, ...
        'trial_position', {trial_position}, 'trial_reward', {trial_reward}, ...
        'trial_world', {trial_world}, 'trialLengths_vr', trialLengths_vr);
end