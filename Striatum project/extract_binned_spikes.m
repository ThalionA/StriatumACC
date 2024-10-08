function [binned_spikes_trials, npx_times_trials] = extract_binned_spikes(all_data, ianimal, npxStartIdx, npxEndIdx)
    % Extracts binned spikes and corresponding time indices per trial.

    binned_spikes_trials = arrayfun(@(s,e) all_data(ianimal).final_spikes(:, s:e), ...
        npxStartIdx, npxEndIdx, 'UniformOutput', false);
    npx_times_trials = cellfun(@(x) 0:size(x, 2)-1, binned_spikes_trials, 'UniformOutput', false);
end