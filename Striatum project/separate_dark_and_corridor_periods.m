function [darkData, corridorData] = separate_dark_and_corridor_periods(trialData, binned_spikes_trials, npx_times_trials)
    % Splits data into dark and corridor periods for each trial.

    n_trials = trialData.n_trials;
    darkData = struct();
    corridorData = struct();

    % Preallocate cell arrays
    fields = {'binned_spikes', 'trial_position', 'trial_licks', 'trial_reward', 'trial_times'};
    for f = fields
        darkData.(f{1}) = cell(1, n_trials);
        corridorData.(f{1}) = cell(1, n_trials);
    end

    % Find corridor start indices
    corridor_start_idx_vr = cellfun(@(x) find(x > 6, 1), trialData.trial_world);
    corridor_start_time_vr = cellfun(@(x, y) x(find(y > 6, 1) + 1), trialData.trial_times_zeroed, trialData.trial_world);

    for itrial = 1:n_trials
        % Find corresponding NPX indices
        [~, corridor_start_idx_npx] = min(abs(npx_times_trials{itrial} - corridor_start_time_vr(itrial)));

        % Dark period
        darkData.binned_spikes{itrial} = binned_spikes_trials{itrial}(:, 1:corridor_start_idx_npx-1);
        darkData.trial_position{itrial} = trialData.trial_position{itrial}(1:corridor_start_idx_vr(itrial)-1);
        darkData.trial_licks{itrial} = trialData.trial_licks{itrial}(1:corridor_start_idx_vr(itrial)-1);
        darkData.trial_reward{itrial} = trialData.trial_reward{itrial}(1:corridor_start_idx_vr(itrial)-1);
        darkData.trial_times{itrial} = trialData.trial_times_zeroed{itrial}(1:corridor_start_idx_vr(itrial)-1);

        % Corridor period
        corridorData.binned_spikes{itrial} = binned_spikes_trials{itrial}(:, corridor_start_idx_npx:end);
        corridorData.trial_position{itrial} = trialData.trial_position{itrial}(corridor_start_idx_vr(itrial):end);
        corridorData.trial_licks{itrial} = trialData.trial_licks{itrial}(corridor_start_idx_vr(itrial):end);
        corridorData.trial_reward{itrial} = trialData.trial_reward{itrial}(corridor_start_idx_vr(itrial):end);
        corridorData.trial_times{itrial} = trialData.trial_times_zeroed{itrial}(corridor_start_idx_vr(itrial):end);
    end
end