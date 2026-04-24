%% Run preprocessing analysis for dark-only data
clearvars -except all_data
clc

% Fixed number of temporal bins per trial
num_temp_bins = 50;

% Check if preprocessed data exists
if exist('Striatum project/preprocessed_data_control2.mat', 'file')
    fprintf('Loading preprocessed data...\n');
    load('preprocessed_data_control2.mat', 'preprocessed_data');
    n_animals = numel(preprocessed_data);
else
    if ~exist('all_data', 'var')
        load('all_data_control2.mat');
    end

    % Firing rate threshold
    fr_threshold = 0.1; % Hz
    n_animals = numel(all_data);

    % Filter neurons based on firing rate
    for ianimal = 1:n_animals
        keep_neurons = all_data(ianimal).avg_fr_all >= fr_threshold;
        all_data(ianimal).final_spikes = all_data(ianimal).final_spikes(keep_neurons, :);
        all_data(ianimal).final_areas = all_data(ianimal).final_areas(keep_neurons);
        all_data(ianimal).avg_fr_all = all_data(ianimal).avg_fr_all(keep_neurons);
    end

    fprintf('Processing data for all animals...\n');
    preprocessed_data = struct();

    for ianimal = 1:n_animals
        fprintf('Processing data for animal %d...\n', ianimal);

        % Cut data per trial
        trialData = cut_data_per_trial(all_data, ianimal);
        n_trials = trialData.n_trials;

        % Align neural data
        n_npx_datapoints = length(all_data(ianimal).npx_time);
        npxStartIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialData.trialStartTimes_vr, 'nearest', 'extrap');
        npxEndIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialData.trialEndTimes_vr, 'nearest', 'extrap');

        % Extract binned spikes per trial
        [binned_spikes_trials, npx_times_trials] = extract_binned_spikes(all_data, ianimal, npxStartIdx, npxEndIdx);


        % Extract spikes and areas for DMS, DLS, and ACC
        is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
        is_dls = strcmp(all_data(ianimal).final_areas, 'DLS');
        is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

        final_spikes_dms = all_data(ianimal).final_spikes(is_dms, :);
        final_spikes_dls = all_data(ianimal).final_spikes(is_dls, :);
        final_spikes_acc = all_data(ianimal).final_spikes(is_acc, :);

        % Preallocate for the results
        num_trials = length(binned_spikes_trials);
        num_neurons = size(binned_spikes_trials{1}, 1); % Assuming all trials have the same number of neurons
        num_temp_bins = 50;

        % Preallocate results
        firing_rates_per_bin = nan(num_neurons, num_temp_bins, num_trials); % Neurons x Temporal Bins x Trials
        binned_times_per_trial = cell(1, num_trials); % To store bin centres for each trial

        for itrial = 1:num_trials
            % Get the spikes and times for this trial
            spikes_in_trial = binned_spikes_trials{itrial}; % Neurons x Time Points
            times_in_trial = npx_times_trials{itrial};     % 1 x Time Points

            % Define bin edges based on the trial's duration
            trial_duration = max(times_in_trial); % Assuming times are in ms
            bin_edges = linspace(0, trial_duration, num_temp_bins + 1); % Include start and end

            % Preallocate for this trial
            binned_spikes = nan(num_neurons, num_temp_bins); % Neurons x Temporal Bins

            % Bin the spikes
            for ibin = 1:num_temp_bins
                bin_start = bin_edges(ibin);
                bin_end = bin_edges(ibin + 1);

                % Logical index for time points within this bin
                idx_in_bin = (times_in_trial >= bin_start) & (times_in_trial < bin_end);

                % Compute firing rate for each neuron in this bin
                if any(idx_in_bin)
                    binned_spikes(:, ibin) = sum(spikes_in_trial(:, idx_in_bin), 2) / ...
                        ((bin_end - bin_start) / 1000); % Firing rate (Hz)
                else
                    binned_spikes(:, ibin) = 0; % If no spikes in the bin, set to 0
                end
            end

            % Store the results for this trial
            firing_rates_per_bin(:, :, itrial) = binned_spikes;
            binned_times_per_trial{itrial} = (bin_edges(1:end-1) + bin_edges(2:end)) / 2; % Bin centres
        end

        % Store results in preprocessed_data
        preprocessed_data(ianimal).firing_rates_per_bin = firing_rates_per_bin;
        preprocessed_data(ianimal).is_dms = is_dms;
        preprocessed_data(ianimal).is_dls = is_dls;
        preprocessed_data(ianimal).is_acc = is_acc;
        preprocessed_data(ianimal).trialData = trialData;
        fprintf('Done with animal %d\n', ianimal);
    end

    % Save the preprocessed data
    save('preprocessed_data_control2.mat', 'preprocessed_data', '-v7.3');
end

% Example visualisation of average firing rates per bin
figure;
for ianimal = 1:n_animals
    avg_firing_rates = mean(preprocessed_data(ianimal).firing_rates_per_bin, 3, 'omitnan'); % Average across trials
    subplot(ceil(sqrt(n_animals)), ceil(sqrt(n_animals)), ianimal);
    imagesc(avg_firing_rates);
    title(sprintf('Animal %d', ianimal));
    xlabel('Temporal Bin');
    ylabel('Neuron');
    colorbar;
end
sgtitle('Average Firing Rates Across Temporal Bins');