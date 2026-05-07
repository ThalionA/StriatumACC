function spatial_binned_data = spatial_binning(corridorData, bin_edges, num_bins)
    % Performs spatial binning for each trial in the corridor period.
    n_trials = length(corridorData.binned_spikes);
    
    % Handle edge case where no trials exist
    if n_trials == 0
        spatial_binned_data = struct();
        return;
    end
    
    n_units = size(corridorData.binned_spikes{1}, 1);
    
    spatial_binned_data = struct();
    spatial_binned_data.spikes = cell(1, n_trials);
    spatial_binned_data.firing_rates = cell(1, n_trials);
    spatial_binned_data.licks = nan(n_trials, num_bins);
    spatial_binned_data.durations = nan(n_trials, num_bins);
    
    for itrial = 1:n_trials
        % --- ROBUSTNESS CHECK ---
        % If trial_times or trial_position is empty, skip this trial to avoid crash
        if isempty(corridorData.trial_times{itrial}) || isempty(corridorData.trial_position{itrial})
            spatial_binned_data.spikes{itrial} = nan(n_units, num_bins);
            spatial_binned_data.firing_rates{itrial} = nan(n_units, num_bins);
            continue; % Move to next trial
        end
        % ------------------------

        % Bin positions
        [~, ~, bin_idx] = histcounts(corridorData.trial_position{itrial}, bin_edges);
        
        spatial_binned_data.spikes{itrial} = nan(n_units, num_bins);
        
        % This line caused the error (now protected by the check above)
        trial_times_zeroed = corridorData.trial_times{itrial} - corridorData.trial_times{itrial}(1);
        
        for ibin = 1:num_bins
            idx_in_bin = (bin_idx == ibin);
            if any(idx_in_bin) && sum(idx_in_bin) > 1
                % Compute total licks
                spatial_binned_data.licks(itrial, ibin) = sum(corridorData.trial_licks{itrial}(idx_in_bin));
                
                % Compute durations
                bin_times = trial_times_zeroed(idx_in_bin);
                spatial_binned_data.durations(itrial, ibin) = (bin_times(end) - bin_times(1)) / 1000;
                
                % (Optional: You can keep or remove the pause)
                if spatial_binned_data.durations(itrial, ibin) <= 0
                    % pause() 
                end
                
                % Find corresponding NPX indices
                npx_bin_times = 0:size(corridorData.binned_spikes{itrial}, 2)-1;
                
                % Robust index finding
                if ~isempty(bin_times) && ~isempty(npx_bin_times)
                    [~, npx_bin_start_idx] = min(abs(bin_times(1) - npx_bin_times));
                    [~, npx_bin_end_idx] = min(abs(bin_times(end) - npx_bin_times));
                    
                    % Sum spikes in the bin
                    spatial_binned_data.spikes{itrial}(:, ibin) = sum(corridorData.binned_spikes{itrial}(:, npx_bin_start_idx:npx_bin_end_idx), 2);
                end
            end
        end
        % Compute firing rates
        spatial_binned_data.firing_rates{itrial} = spatial_binned_data.spikes{itrial} ./ spatial_binned_data.durations(itrial, :);
    end
end