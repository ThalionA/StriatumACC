function [trial_average_fr, trial_sem_fr] = compute_firing_rates(binned_spikes_trials, trialDurations_vr)
    % Computes average and SEM firing rates per trial.

    num_trials = length(binned_spikes_trials);
    total_spikes = zeros(num_trials, 1);

    for i = 1:num_trials
        total_spikes(i) = mean(sum(binned_spikes_trials{i}, 2));
    end

    trial_average_fr = total_spikes ./ trialDurations_vr;
    trial_sem_fr = std(total_spikes) ./ sqrt(num_trials) ./ trialDurations_vr;
end