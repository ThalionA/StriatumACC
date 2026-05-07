function change_point_mean = find_change_points(trialDurations_vr, trial_metrics, mov_window_size)
    % Identifies change points based on trial duration, licks, and success.

    [~, duration_peaks] = findpeaks(movmean(trialDurations_vr, mov_window_size), 'MinPeakProminence', 5);
    trial_licks_change = find(movmean(trial_metrics.trial_lick_no, 10) < 20, 1);
    trial_success_change = find(movmean(trial_metrics.trial_success, 10) < 0.5, 1);

    if isempty(trial_success_change) && isempty(trial_licks_change)
        change_point_mean = nan;
        return
    end

    try
        [~, loc1] = min(abs(duration_peaks - trial_licks_change));
    catch
        loc1 = nan;
    end

    try
        [~, loc2] = min(abs(duration_peaks - trial_success_change));
    catch
        loc2 = nan;
    end

    if isnan(loc1)
        most_likely_change_duration = duration_peaks(loc2);
    elseif isnan(loc2)
        most_likely_change_duration = duration_peaks(loc1);
    else
        most_likely_change_duration = mean([duration_peaks(loc1), duration_peaks(loc2)]);
    end

    change_point_mean = floor(mean([trial_licks_change, trial_success_change, most_likely_change_duration], 'omitmissing'));
end