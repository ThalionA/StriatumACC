function change_point_mean = find_change_points(trialDurations_vr, trial_metrics, mov_window_size)
    % Identifies change points based on trial duration, licks, and success.

    [~, duration_peaks] = findpeaks(movmean(trialDurations_vr, mov_window_size), 'MinPeakProminence', 5);
    trial_licks_change = find(movmean(trial_metrics.trial_lick_no, 10) < 20, 1);
    trial_success_change = find(movmean(trial_metrics.trial_success, 10) < 0.5, 1);

    [~, loc1] = min(abs(duration_peaks - trial_licks_change));
    [~, loc2] = min(abs(duration_peaks - trial_success_change));
    most_likely_change_duration = mean([duration_peaks(loc1), duration_peaks(loc2)]);

    change_point_mean = floor(mean([trial_licks_change, trial_success_change, most_likely_change_duration]));
end