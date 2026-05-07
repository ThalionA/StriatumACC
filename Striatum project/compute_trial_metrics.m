function trial_metrics = compute_trial_metrics(trialData)
    % Computes total licks and trial success per trial.

    trial_lick_no = cellfun(@sum, trialData.trial_licks);
    trial_success = cellfun(@max, trialData.trial_reward);

    trial_metrics = struct('trial_lick_no', trial_lick_no, 'trial_success', trial_success);
end