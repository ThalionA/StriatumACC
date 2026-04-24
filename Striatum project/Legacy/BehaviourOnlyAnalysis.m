%% Define mouse IDs and preallocate
all_mouse_ids = {'1105_M2', '1106_M3', '1201_M1', '1206_M2', '1212_M3', '1215_M2', '1217_M4', '1219_M1'};
num_mice = numel(all_mouse_ids);
all_data = struct();

%% Process each mouse – load and process behavioural data only
for imouse = 1:num_mice
    % Load behavioural data file (assumes file contains VR_data)
    filename = ['./BehaviourOnly/' all_mouse_ids{imouse} '.mat'];
    load(filename);  % Expected variable: VR_data
    
    % Correct time (in ms) using the first row of VR_data as time stamps
    corrected_vr_time = (VR_data(1,:) - VR_data(1,1)) * 1000;
    
    % Process licks (assuming process_licks is adapted for behavioural data)
    corrected_licks = process_licks(VR_data(8, :) >= 1, corrected_vr_time, 100);
    
    % Save the behavioural variables into all_data struct
    all_data(imouse).mouseid = all_mouse_ids{imouse};
    all_data(imouse).corrected_vr_time = corrected_vr_time;
    all_data(imouse).corrected_licks = corrected_licks';
    all_data(imouse).vr_position = VR_data(2, :);
    all_data(imouse).vr_world = VR_data(5, :);
    all_data(imouse).vr_reward = VR_data(6, :);
    all_data(imouse).vr_trial = VR_data(7, :);
    
    % Compute average lick rate (licks per second)
    total_duration_sec = (corrected_vr_time(end) - corrected_vr_time(1)) / 1000;
    average_lick_rate = sum(corrected_licks) / total_duration_sec;
    all_data(imouse).average_lick_rate = average_lick_rate;
    
    fprintf('Done with animal %s\n', all_mouse_ids{imouse});
end

%% Set up spatial binning parameters
reward_zone_start_cm = 125;    % in cm
visual_zone_start_au = 80;
reward_zone_start_au = 100;
reward_zone_end_au = 135;
corridor_end_au = 200;

bin_size = 4;  % e.g. for x1.25 conversion if needed for centimetres
bin_edges = 0:bin_size:corridor_end_au;
bin_edges(end) = corridor_end_au + bin_size;
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres);

%% Behavioural Trial Segmentation and Lick Precision Analysis
% Process each animal's data to segment trials and compute lick-based metrics.
for imouse = 1:num_mice
    % Use the trial segmentation function (adapted for behavioural data only)
    trialData = cut_data_per_trial(all_data, imouse);
    
    % Discard the last trial
    trialData.n_trials = trialData.n_trials - 1;
    trialData.trialStartTimes_vr(end) = [];
    trialData.trialEndTimes_vr(end) = [];
    trialData.trialDurations_vr(end) = [];
    trialData.trial_times_zeroed(end) = [];
    trialData.trial_licks(end) = [];
    trialData.trial_position(end) = [];
    trialData.trial_lick_positions(end) = [];
    trialData.trial_reward(end) = [];
    trialData.trial_world(end) = [];
    trialData.trialLengths_vr(end) = [];
    
    all_data(imouse).trialData = trialData;
    
    % Separate dark and corridor periods based on behavioural markers
    [darkData, corridorData] = separate_dark_and_corridor_periods_behavior(trialData);
    
    % Use corridor periods for lick analysis
    trial_lick_positions = cellfun(@(x, y) x(logical(y)), corridorData.trial_position, corridorData.trial_licks, 'UniformOutput', false);

    [trial_lick_errors, shuffled_lick_error_means, shuffled_lick_error_stds, ~] = ...
        cellfun(@(x) calculate_lick_precision(x, reward_zone_start_au), trial_lick_positions);

    outlier_trials = isoutlier(trial_lick_errors, "percentiles", [0, 99]);
    trial_lick_errors(outlier_trials) = nan;

    zscored_lick_errors = (trial_lick_errors - shuffled_lick_error_means) ./ shuffled_lick_error_stds;
    
    all_data(imouse).trial_lick_errors = trial_lick_errors;
    all_data(imouse).zscored_lick_errors = zscored_lick_errors;
    
    % Compute spatial binning based on VR position and lick events
    spatial_binned_data = spatial_binning(trialData, bin_edges, num_bins);
    all_data(imouse).spatial_binned_data = spatial_binned_data;
    
    fprintf('Processed trial data for animal %s\n', all_data(imouse).mouseid);
end

%% Plot behavioural metrics across animals
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');
mov_window_size = 5;

for imouse = 1:num_mice
    nexttile
    zscored_lick_errors = all_data(imouse).zscored_lick_errors;
    trials = size(zscored_lick_errors, 2);

    shadedErrorBar(1:trials, movmean(zscored_lick_errors, mov_window_size, 'omitmissing'), ...
        movstd(zscored_lick_errors, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    yline(-2, 'r')
    title(all_data(imouse).mouseid)
    xlabel('Trial')
    ylabel('Z-scored Lick Error')
    axis tight
end

%% Lick Heatmaps
figure

for imouse = 1:num_mice
    licks_to_plot = all_data(imouse).spatial_binned_data.lick_rate;
    licks_to_plot = cat(1, licks_to_plot{:});
    licks_to_plot(licks_to_plot > quantile(licks_to_plot, 0.99, 'all')) = nan;
    
    nexttile
    imagesc(licks_to_plot)
    xlabel('spatial bin (x5cm)')
    ylabel('trial')
    xline([20, 25], 'Color', 'w')
    title(all_data(imouse).mouseid)
end

%% Helper functions

function trialData = cut_data_per_trial(all_data, ianimal)
    % Extracts and organises trial-based data from all_data for a given animal.
    vr_time = all_data(ianimal).corrected_vr_time;
    vr_trial = all_data(ianimal).vr_trial;
    
    n_vr_datapoints = length(vr_time);
    changeIdx_vr = [find(diff(vr_trial) ~= 0), n_vr_datapoints];
    n_trials = numel(changeIdx_vr);
    
    trialStartIdx_vr = [1, changeIdx_vr(1:end-1) + 1];
    trialEndIdx_vr = changeIdx_vr;
    trialLengths_vr = trialEndIdx_vr - trialStartIdx_vr + 1;
    
    % Extract and zero trial times
    trialTimes_vr = mat2cell(vr_time, 1, trialLengths_vr);
    trial_times_zeroed = cellfun(@(x) x - x(1), trialTimes_vr, 'UniformOutput', false);
    trialStartTimes_vr = vr_time(trialStartIdx_vr);
    trialEndTimes_vr = vr_time(trialEndIdx_vr);
    trialDurations_vr = (trialEndTimes_vr - trialStartTimes_vr) / 1000;  % in seconds
    
    % Extract other trial-based variables
    trial_licks = mat2cell(all_data(ianimal).corrected_licks, 1, trialLengths_vr);
    trial_position = mat2cell(all_data(ianimal).vr_position, 1, trialLengths_vr);
    trial_reward = mat2cell(all_data(ianimal).vr_reward, 1, trialLengths_vr);
    trial_world = mat2cell(all_data(ianimal).vr_world, 1, trialLengths_vr);
    
    % Compute trial-specific lick positions
    trial_lick_positions = cellfun(@(pos, licks) pos(logical(licks)), trial_position, trial_licks, 'UniformOutput', false);
    
    % Package data into a struct
    trialData = struct(...
        'n_trials', n_trials, ...
        'trialStartTimes_vr', trialStartTimes_vr, ...
        'trialEndTimes_vr', trialEndTimes_vr, ...
        'trialDurations_vr', trialDurations_vr, ...
        'trial_times_zeroed', {trial_times_zeroed}, ...
        'trial_licks', {trial_licks}, ...
        'trial_position', {trial_position}, ...
        'trial_lick_positions', {trial_lick_positions}, ...
        'trial_reward', {trial_reward}, ...
        'trial_world', {trial_world}, ...
        'trialLengths_vr', trialLengths_vr);
end

function [p, shuffled_lick_precision_mean, shuffled_lick_precision_std, chance_quantile] = calculate_lick_precision(l, rz_start)
    % Calculates lick precision (error) based on how far lick positions are from
    % the reward zone start (rz_start). Lick positions beyond rz_start are ignored.
    l(l > rz_start) = nan;
    p = sum((l - rz_start).^2, 'omitnan');
    
    lick_number = sum(~isnan(l));
    shuffled_lick_precision_mean = nan;
    shuffled_lick_precision_std = nan;
    chance_quantile = nan;
    
    if lick_number > 0
        shuffled_l = rz_start * rand(500, lick_number);
        shuffled_lick_precision = sum((shuffled_l - rz_start).^2, 2);
        shuffled_lick_precision_mean = mean(shuffled_lick_precision);
        shuffled_lick_precision_std = std(shuffled_lick_precision);
        chance_quantile = scalar_to_quantile(shuffled_lick_precision, p);
    end
end

function quantile = scalar_to_quantile(shuffled_values, scalar)
    % Converts a scalar into its quantile value based on a distribution.
    quantile = sum(shuffled_values < scalar) / numel(shuffled_values);
end

function spatial_binned_data = spatial_binning(trialData, bin_edges, num_bins)
    % spatial_binning computes occupancy, lick counts and lick rates across spatial bins.
    % It uses the VR position data and the corresponding trial times.
    %
    % Inputs:
    %   trialData - structure from cut_data_per_trial containing:
    %       trial_position: cell array of VR positions per trial.
    %       trial_times_zeroed: cell array of trial times (in ms) per trial.
    %       trial_lick_positions: cell array of lick positions per trial.
    %   bin_edges - vector of edges for spatial bins.
    %   num_bins  - number of spatial bins.
    %
    % Outputs:
    %   spatial_binned_data - structure with fields:
    %       occupancy: cell array (per trial) of occupancy time (in seconds) per bin.
    %       lick_count: cell array (per trial) of number of licks per bin.
    %       lick_rate:  cell array (per trial) of lick rate (licks/s) per bin.
    
    n_trials = trialData.n_trials;
    occupancy = cell(1, n_trials);
    lick_count = cell(1, n_trials);
    lick_rate = cell(1, n_trials);
    
    for itrial = 1:n_trials
        % Get VR positions and corresponding times (in ms)
        pos = trialData.trial_position{itrial};
        t = trialData.trial_times_zeroed{itrial};
        
        % Estimate sampling interval (in seconds)
        if numel(t) > 1
            dt = median(diff(t)) / 1000;
        else
            dt = 0;
        end
        
        % Calculate occupancy: count of samples in each bin times dt gives time spent in each bin
        occ_counts = histcounts(pos, bin_edges);
        occupancy{itrial} = occ_counts * dt;
        
        % Lick counts: histogram of lick positions for the trial
        lick_pos = trialData.trial_lick_positions{itrial};
        lick_count{itrial} = histcounts(lick_pos, bin_edges);
        
        % Lick rate: number of licks per second (avoid division by zero)
        lick_rate{itrial} = lick_count{itrial} ./ (occupancy{itrial} + eps);
    end
    
    spatial_binned_data = struct();
    spatial_binned_data.occupancy = occupancy;
    spatial_binned_data.lick_count = lick_count;
    spatial_binned_data.lick_rate = lick_rate;
end

function [darkData, corridorData] = separate_dark_and_corridor_periods_behavior(trialData)
    % Splits each trial into dark and corridor periods based on trial_world.
    %
    % Here, we assume that a value in trial_world above 6 marks the start
    % of the corridor period.
    
    n_trials = trialData.n_trials;
    darkData = struct();
    corridorData = struct();
    
    % Preallocate fields for behavioural signals
    fields = {'trial_position', 'trial_licks', 'trial_reward', 'trial_times_zeroed'};
    for f = fields
        darkData.(f{1}) = cell(1, n_trials);
        corridorData.(f{1}) = cell(1, n_trials);
    end
    
    % Determine the corridor start index for each trial using trial_world
    corridor_start_idx = cellfun(@(world) find(world > 6, 1, 'first'), trialData.trial_world);
    
    for itrial = 1:n_trials
        idx = corridor_start_idx(itrial);
        if isempty(idx)
            % If no corridor marker is found, treat the entire trial as dark.
            darkData.trial_position{itrial} = trialData.trial_position{itrial};
            darkData.trial_licks{itrial} = trialData.trial_licks{itrial};
            darkData.trial_reward{itrial} = trialData.trial_reward{itrial};
            darkData.trial_times_zeroed{itrial} = trialData.trial_times_zeroed{itrial};
            corridorData.trial_position{itrial} = [];
            corridorData.trial_licks{itrial} = [];
            corridorData.trial_reward{itrial} = [];
            corridorData.trial_times_zeroed{itrial} = [];
        else
            darkData.trial_position{itrial} = trialData.trial_position{itrial}(1:idx-1);
            darkData.trial_licks{itrial} = trialData.trial_licks{itrial}(1:idx-1);
            darkData.trial_reward{itrial} = trialData.trial_reward{itrial}(1:idx-1);
            darkData.trial_times_zeroed{itrial} = trialData.trial_times_zeroed{itrial}(1:idx-1);
            
            corridorData.trial_position{itrial} = trialData.trial_position{itrial}(idx:end);
            corridorData.trial_licks{itrial} = trialData.trial_licks{itrial}(idx:end);
            corridorData.trial_reward{itrial} = trialData.trial_reward{itrial}(idx:end);
            corridorData.trial_times_zeroed{itrial} = trialData.trial_times_zeroed{itrial}(idx:end);
        end
    end
end