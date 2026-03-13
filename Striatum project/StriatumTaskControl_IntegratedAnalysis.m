%% Integrated Cortico-Striatal Spatiotemporal Analysis Pipeline
% No toolboxes required for decoders (implemented via analytical Ridge Regression).
clearvars; clc; close all;

%% 1. Configuration & Data Loading
% Define learning point threshold criteria
cfg.lp_window = 10; % Window of trials to evaluate
cfg.lp_thresh_count = 7; % User-defined: number of trials within window that must pass threshold
cfg.lp_z_thresh = -2; % Z-score threshold for lick error

fprintf('Loading task and control data...\n');
% Ensure these files exist in your path
load('preprocessed_data.mat', 'preprocessed_data');
task_data = preprocessed_data;

load('preprocessed_data_control.mat', 'preprocessed_data'); 
control_data = preprocessed_data;
% Assuming control_data has the same structure but represents the blank VR corridor habituation

n_animals_task = numel(task_data);
n_animals_ctrl = numel(control_data);
n_bins = size(task_data(1).spatial_binned_data.licks, 2);
bin_size = 4; % Spatial bin multiplier

%% 2. Define Learning (LP) and Disengagement Points (DP)
learning_points_task = nan(1, n_animals_task);
diseng_points_task = nan(1, n_animals_task);

for i = 1:n_animals_task
    z_err = task_data(i).zscored_lick_errors;
    n_trials = length(z_err);
    
    % Find Learning Point
    for tr = 1:(n_trials - cfg.lp_window + 1)
        window_err = z_err(tr : tr + cfg.lp_window - 1);
        if sum(window_err <= cfg.lp_z_thresh) >= cfg.lp_thresh_count
            learning_points_task(i) = tr;
            break;
        end
    end
    
    % Find Disengagement Point (if exists)
    if isfield(task_data(i), 'change_point_mean') && ~isempty(task_data(i).change_point_mean)
        diseng_points_task(i) = task_data(i).change_point_mean;
    end
end

% Averages for control animals
avg_lp = round(mean(learning_points_task, 'omitnan'));
avg_dp = round(mean(diseng_points_task, 'omitnan'));
if isnan(avg_dp), avg_dp = avg_lp + 30; end % Fallback if no DP exists

fprintf('Average Task Learning Point: Trial %d\n', avg_lp);
fprintf('Average Task Disengagement Point: Trial %d\n', avg_dp);

%% Plot Behavioral Evolution: Lick Heatmaps and Z-Scored Errors
fprintf('--- Generating Behavioral Plots for All Mice ---\n');

% --- Configuration ---
% Zone parameters
visual_zone_bin = 20;
reward_zone_bin = 25;

% Smoothing parameter for lick errors
mov_avg_window = 10; 

% Toggle whether to cap Task plots at their respective disengagement points
cap_at_disengagement = true; 
% ---------------------

% =========================================================================
% TASK GROUP PLOTS
% =========================================================================
% 1. Task Group - Lick Rate Heatmaps
figure('Name', 'Task Group - Lick Rate Heatmaps', 'Position', [50, 100, 1400, 800], 'Color', 'w');
t_task_hm = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');

if cap_at_disengagement
    title(t_task_hm, 'Task Group - Lick Rate Heatmaps (up to Disengagement)', 'FontWeight', 'bold', 'FontSize', 14);
else
    title(t_task_hm, 'Task Group - Lick Rate Heatmaps (All Trials)', 'FontWeight', 'bold', 'FontSize', 14);
end

for ianimal = 1:n_animals_task
    licks = task_data(ianimal).spatial_binned_data.licks;
    durations = task_data(ianimal).spatial_binned_data.durations;
    lick_rate = licks ./ durations;
    
    % Cap extreme outliers
    cap_val = quantile(lick_rate(:), 0.99);
    lick_rate(lick_rate > cap_val) = cap_val;
    
    % Determine trials to plot based on toggle
    dp = diseng_points_task(ianimal);
    if cap_at_disengagement && ~isnan(dp) && dp > 0 && dp <= size(lick_rate, 1)
        plot_trials = 1:dp;
    else
        plot_trials = 1:size(lick_rate, 1);
    end
    
    lick_rate_plot = lick_rate(plot_trials, :);
    
    ax = nexttile(t_task_hm);
    imagesc(ax, lick_rate_plot);
    
    % Add Landmark/Reward Zone lines (Task only)
    xline(visual_zone_bin, 'w--', 'LineWidth', 1.5);
    xline(reward_zone_bin, 'w-', 'LineWidth', 1.5);
    
    title(sprintf('Mouse %d', ianimal));
    xlabel('Spatial Bin');
    ylabel('Trial');
end
cb = colorbar(ax);
cb.Label.String = 'Lick Rate (Hz)';
cb.Layout.Tile = 'east';
save_to_svg('Task_Lick_Rate_Heatmaps');

% 2. Task Group - Z-Scored Lick Errors
figure('Name', 'Task Group - Z-Scored Lick Errors (Smoothed)', 'Position', [100, 150, 1400, 800], 'Color', 'w');
t_task_z = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');

if cap_at_disengagement
    title(t_task_z, sprintf('Task Group - Z-Scored Lick Errors (%d-trial moving average, up to Disengagement)', mov_avg_window), 'FontWeight', 'bold', 'FontSize', 14);
else
    title(t_task_z, sprintf('Task Group - Z-Scored Lick Errors (%d-trial moving average, All Trials)', mov_avg_window), 'FontWeight', 'bold', 'FontSize', 14);
end

for ianimal = 1:n_animals_task
    z_errors = task_data(ianimal).zscored_lick_errors;
    
    % Clean Infs 
    z_errors(isinf(z_errors)) = nan;
    
    % Apply moving average and rolling standard error for the shaded bounds
    z_errors_smoothed = movmean(z_errors, mov_avg_window, 'omitnan');
    z_errors_se = movstd(z_errors, mov_avg_window, 'omitnan') ./ sqrt(mov_avg_window);
    
    % Explicitly set any smoothed value > 10 to NaN
    invalid_idx = z_errors_smoothed > 10;
    z_errors_smoothed(invalid_idx) = nan;
    z_errors_se(invalid_idx) = nan;
    
    % Determine trials to plot based on toggle
    dp = diseng_points_task(ianimal);
    if cap_at_disengagement && ~isnan(dp) && dp > 0 && dp <= length(z_errors_smoothed)
        plot_trials = 1:dp;
    else
        plot_trials = 1:length(z_errors_smoothed);
    end
    
    z_err_plot = z_errors_smoothed(plot_trials);
    z_err_se_plot = z_errors_se(plot_trials);
    
    nexttile(t_task_z);
    hold on;
    
    % Thresholds (plotted before the data so they sit behind)
    yline(0, 'r--');
    yline(cfg.lp_z_thresh, 'b--');
    
    % Plot the smoothed data with error bounds
    shadedErrorBar(plot_trials, z_err_plot, z_err_se_plot, 'lineProps', {'Color', 'k', 'LineWidth', 1.5});
    
    % Mark specific Learning Point if available
    lp = learning_points_task(ianimal);
    if ~isnan(lp) && lp <= plot_trials(end)
        xline(lp, 'g-', 'LP', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
    end
    
    title(sprintf('Mouse %d', ianimal));
    xlim([1 plot_trials(end)]);
    xlabel('Trial Number');
    ylabel('Z-Score (Smoothed)');
    grid on; box on;
end
save_to_svg('Task_ZScored_Lick_Errors');

% =========================================================================
% CONTROL GROUP PLOTS
% =========================================================================
if n_animals_ctrl > 0
    % 3. Control Group - Lick Rate Heatmaps
    figure('Name', 'Control Group - Lick Rate Heatmaps', 'Position', [150, 200, 1400, 800], 'Color', 'w');
    t_ctrl_hm = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_ctrl_hm, 'Control Group - Lick Rate Heatmaps (All Trials)', 'FontWeight', 'bold', 'FontSize', 14);

    for ianimal = 1:n_animals_ctrl
        licks = control_data(ianimal).spatial_binned_data.licks;
        durations = control_data(ianimal).spatial_binned_data.durations;
        lick_rate = licks ./ durations;
        
        cap_val = quantile(lick_rate(:), 0.99);
        lick_rate(lick_rate > cap_val) = cap_val;
        
        ax = nexttile(t_ctrl_hm);
        imagesc(ax, lick_rate);
        
        % NOTE: Visual and Reward xlines removed here since control is a blank corridor!
        
        title(sprintf('Ctrl Mouse %d', ianimal));
        xlabel('Spatial Bin');
        ylabel('Trial');
    end
    cb2 = colorbar(ax);
    cb2.Label.String = 'Lick Rate (Hz)';
    cb2.Layout.Tile = 'east';
    save_to_svg('Control_Lick_Rate_Heatmaps');

    % 4. Control Group - Z-Scored Lick Errors
    figure('Name', 'Control Group - Z-Scored Lick Errors (Smoothed)', 'Position', [200, 250, 1400, 800], 'Color', 'w');
    t_ctrl_z = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_ctrl_z, sprintf('Control Group - Z-Scored Lick Errors (%d-trial moving average)', mov_avg_window), 'FontWeight', 'bold', 'FontSize', 14);

    for ianimal = 1:n_animals_ctrl
        z_errors = control_data(ianimal).zscored_lick_errors;
        
        % Clean Infs
        z_errors(isinf(z_errors)) = nan;
        
        % Apply moving average and rolling standard error
        z_errors_smoothed = movmean(z_errors, mov_avg_window, 'omitnan');
        z_errors_se = movstd(z_errors, mov_avg_window, 'omitnan') ./ sqrt(mov_avg_window);
        
        % Explicitly set any smoothed value > 10 to NaN
        invalid_idx = z_errors_smoothed > 10;
        z_errors_smoothed(invalid_idx) = nan;
        z_errors_se(invalid_idx) = nan;
        
        n_trials = length(z_errors_smoothed);
        
        nexttile(t_ctrl_z);
        hold on;
        
        yline(0, 'r--');
        yline(cfg.lp_z_thresh, 'b--');
        
        shadedErrorBar(1:n_trials, z_errors_smoothed, z_errors_se, 'lineProps', {'Color', 'k', 'LineWidth', 1.5});
        
        title(sprintf('Ctrl Mouse %d', ianimal));
        xlim([1 n_trials]);
        xlabel('Trial Number');
        ylabel('Z-Score (Smoothed)');
        grid on; box on;
    end
    save_to_svg('Control_ZScored_Lick_Errors');
end

fprintf('--- Plotting Complete ---\n');

%% 3. Extract Epoch Data (Behavior)
fprintf('Extracting behavioral data across epochs...\n');
epochs = {'Naive', 'Intermediate', 'Expert'};
n_epochs = length(epochs);

% Preallocate: [Animals x Bins x Epochs x Group(Task=1, Ctrl=2)]
avg_licks = nan(max(n_animals_task, n_animals_ctrl), n_bins, n_epochs, 2);
avg_vel = nan(max(n_animals_task, n_animals_ctrl), n_bins, n_epochs, 2);
groups = {task_data, control_data};

for g = 1:2
    data_group = groups{g};
    for i = 1:numel(data_group)
        if g == 1
            lp = learning_points_task(i);
            % Only include task animals that actually learned
            if isnan(lp)
                continue; 
            end
        else
            lp = avg_lp;
        end
        
        licks = data_group(i).spatial_binned_data.licks;
        durs = data_group(i).spatial_binned_data.durations;
        
        vel = (bin_size * 1.25) ./ durs;
        lick_rate = licks ./ durs;
        
        vel(isinf(vel)) = nan;
        lick_rate(isinf(lick_rate)) = nan;
        
        n_trials = size(lick_rate, 1);
        
        % Epoch 1: Naive (1:10)
        if n_trials >= 10
            avg_licks(i, :, 1, g) = mean(lick_rate(1:10, :), 1, 'omitnan');
            avg_vel(i, :, 1, g) = mean(vel(1:10, :), 1, 'omitnan');
        end
        
        % Epoch 2: Intermediate (LP-10 to LP-1)
        if ~isnan(lp) && lp > 10 && lp <= n_trials
            avg_licks(i, :, 2, g) = mean(lick_rate(lp-10:lp-1, :), 1, 'omitnan');
            avg_vel(i, :, 2, g) = mean(vel(lp-10:lp-1, :), 1, 'omitnan');
        end
        
        % Epoch 3: Expert (LP to LP+9)
        if ~isnan(lp) && (lp + 9) <= n_trials
            avg_licks(i, :, 3, g) = mean(lick_rate(lp:lp+9, :), 1, 'omitnan');
            avg_vel(i, :, 3, g) = mean(vel(lp:lp+9, :), 1, 'omitnan');
        end
    end
end

% --- Plotting Behaviour (Task Left, Control Right) ---
figure('Name', 'Behavioural Evolution across Epochs', 'Position', [100, 100, 1000, 600], 'Color', 'w');
colors = lines(n_epochs);
groups_name = {'Task', 'Control'};
ax_beh_licks = gobjects(1, 2);
ax_beh_vel = gobjects(1, 2);

for g = 1:2
    % Lick Rate (Top Row: Task = 1, Ctrl = 2)
    ax_beh_licks(g) = subplot(2, 2, g); hold on; 
    title(sprintf('%s - Lick Rate', groups_name{g}));
    h_lines = gobjects(1, n_epochs);
    
    for e = 1:n_epochs
        valid_data = squeeze(avg_licks(:, :, e, g));
        if all(isnan(valid_data(:))), continue; end
        
        N = sum(~isnan(valid_data), 1);
        mu = mean(valid_data, 1, 'omitnan');
        se = std(valid_data, 0, 1, 'omitnan') ./ sqrt(N);
        
        h = shadedErrorBar(1:n_bins, mu, se, 'lineProps', {'Color', colors(e,:), 'LineWidth', 2});
        if isfield(h, 'mainLine'), h_lines(e) = h.mainLine; end
    end
    
    if g == 1 && any(isgraphics(h_lines))
        xline(20, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom'); 
        xline(25, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom');
        legend(h_lines(isgraphics(h_lines)), epochs, 'Location', 'best'); 
    end
    
    ylabel('Lick Rate (Hz)');
    
    % Velocity (Bottom Row: Task = 3, Ctrl = 4)
    ax_beh_vel(g) = subplot(2, 2, g+2); hold on; 
    title(sprintf('%s - Velocity', groups_name{g}));
    
    for e = 1:n_epochs
        valid_data = squeeze(avg_vel(:, :, e, g));
        if all(isnan(valid_data(:))), continue; end
        
        N = sum(~isnan(valid_data), 1);
        mu = mean(valid_data, 1, 'omitnan');
        se = std(valid_data, 0, 1, 'omitnan') ./ sqrt(N);
        
        shadedErrorBar(1:n_bins, mu, se, 'lineProps', {'Color', colors(e,:), 'LineWidth', 2});
    end
    
    xlabel('Spatial Bin'); 
    ylabel('Velocity (cm/s)');
    if g == 1
        xline(20, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom'); 
        xline(25, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom');
    end
end
linkaxes(ax_beh_licks, 'xy');
linkaxes(ax_beh_vel, 'xy');
save_to_svg('Behavioural_Evolution_Epochs');

%% 4 & 5. Trial-to-Trial Correlation (Single-Neuron Reliability)
fprintf('Computing continuous single-neuron reliability for task and control groups...\n');
epochs = {'Naive', 'Intermediate', 'Expert'};
n_epochs = length(epochs);
areas = {'dms', 'dls', 'acc', 'all'}; % Added 'all' for whole population
n_areas = length(areas);
window_size = 5; 
half_win = floor(window_size/2);
trials_per_epoch = 10;
min_units = 5;
max_animals = max(n_animals_task, n_animals_ctrl);

% --- Preallocate Hierarchical Trackers ---
% Dimensions: [Animals x Epochs x Trials x Group x Area]
hier_raw       = nan(max_animals, n_epochs, trials_per_epoch, 2, n_areas);
hier_z         = nan(max_animals, n_epochs, trials_per_epoch, 2, n_areas);
hier_raw_shuff = nan(max_animals, n_epochs, trials_per_epoch, 2, n_areas);
hier_z_shuff   = nan(max_animals, n_epochs, trials_per_epoch, 2, n_areas);

% --- Preallocate Pooled Trackers ---
% Cell Array Dimensions: {Group, Area} -> holds [N_units x Epochs x Trials]
pooled_raw       = cell(2, n_areas);
pooled_z         = cell(2, n_areas);
pooled_raw_shuff = cell(2, n_areas);
pooled_z_shuff   = cell(2, n_areas);

groups = {task_data, control_data};
for g = 1:2
    curr_data = groups{g};
    for i = 1:numel(curr_data)
        if g == 1
            lp = learning_points_task(i);
            % Only include task animals that actually learned
            if isnan(lp)
                continue;
            end
        else
            lp = avg_lp;
        end
        
        activity_raw = curr_data(i).spatial_binned_fr_all; % [neurons x bins x trials]
        masks = {curr_data(i).is_dms, curr_data(i).is_dls, curr_data(i).is_acc};
        n_trials = size(activity_raw, 3);
        n_cells_total = size(activity_raw, 1);
        
        % Pre-compute Z-scored activity
        activity_z = nan(size(activity_raw));
        for c = 1:n_cells_total
            unit_data = squeeze(activity_raw(c, :, :));
            mu = mean(unit_data(:), 'omitnan');
            sig = std(unit_data(:), 'omitnan');
            if sig > 0
                activity_z(c, :, :) = (unit_data - mu) / sig;
            else
                activity_z(c, :, :) = 0; % Handle silent neurons
            end
        end
        
        % Create Trial-Shuffled control datasets
        shuff_idx = randperm(n_trials);
        activity_raw_shuff = activity_raw(:, :, shuff_idx);
        activity_z_shuff   = activity_z(:, :, shuff_idx);
        
        % Compute continuous rolling stability for all cells
        stab_raw       = nan(n_cells_total, n_trials);
        stab_z         = nan(n_cells_total, n_trials);
        stab_raw_shuff = nan(n_cells_total, n_trials);
        stab_z_shuff   = nan(n_cells_total, n_trials);
        
        for t = 1:n_trials
            win_idx = max(1, t - half_win) : min(n_trials, t + half_win);
            if length(win_idx) < 2, continue; end
            
            for c = 1:n_cells_total
                stab_raw(c, t)       = calc_triu_corr(squeeze(activity_raw(c, :, win_idx)));
                stab_z(c, t)         = calc_triu_corr(squeeze(activity_z(c, :, win_idx)));
                stab_raw_shuff(c, t) = calc_triu_corr(squeeze(activity_raw_shuff(c, :, win_idx)));
                stab_z_shuff(c, t)   = calc_triu_corr(squeeze(activity_z_shuff(c, :, win_idx)));
            end
        end
        
        % Post-hoc extraction of epochs
        epoch_idx = cell(1, n_epochs);
        if n_trials >= trials_per_epoch, epoch_idx{1} = 1:trials_per_epoch; end
        if ~isnan(lp) && lp > trials_per_epoch && lp <= n_trials
            epoch_idx{2} = (lp - trials_per_epoch) : (lp - 1); 
        end
        if ~isnan(lp) && (lp + trials_per_epoch - 1) <= n_trials
            epoch_idx{3} = lp : (lp + trials_per_epoch - 1); 
        end
        
        % Unit-level epoch storage
        ep_raw       = nan(n_cells_total, n_epochs, trials_per_epoch);
        ep_z         = nan(n_cells_total, n_epochs, trials_per_epoch);
        ep_raw_shuff = nan(n_cells_total, n_epochs, trials_per_epoch);
        ep_z_shuff   = nan(n_cells_total, n_epochs, trials_per_epoch);
        
        for e = 1:n_epochs
            idx = epoch_idx{e};
            if isempty(idx), continue; end
            take_n = min(length(idx), trials_per_epoch);
            
            ep_raw(:, e, 1:take_n)       = stab_raw(:, idx(1:take_n));
            ep_z(:, e, 1:take_n)         = stab_z(:, idx(1:take_n));
            ep_raw_shuff(:, e, 1:take_n) = stab_raw_shuff(:, idx(1:take_n));
            ep_z_shuff(:, e, 1:take_n)   = stab_z_shuff(:, idx(1:take_n));
        end
        
        % Distribute into Hierarchical and Pooled arrays
        for a = 1:n_areas
            if a == 4
                area_mask = true(n_cells_total, 1); % 'all'
            else
                area_mask = masks{a};
            end
            
            if sum(area_mask) < min_units, continue; end
            
            % Hierarchical (Average across cells per animal)
            hier_raw(i, :, :, g, a)       = mean(ep_raw(area_mask, :, :), 1, 'omitnan');
            hier_z(i, :, :, g, a)         = mean(ep_z(area_mask, :, :), 1, 'omitnan');
            hier_raw_shuff(i, :, :, g, a) = mean(ep_raw_shuff(area_mask, :, :), 1, 'omitnan');
            hier_z_shuff(i, :, :, g, a)   = mean(ep_z_shuff(area_mask, :, :), 1, 'omitnan');
            
            % Pooled (Concat cells)
            pooled_raw{g, a}       = cat(1, pooled_raw{g, a}, ep_raw(area_mask, :, :));
            pooled_z{g, a}         = cat(1, pooled_z{g, a}, ep_z(area_mask, :, :));
            pooled_raw_shuff{g, a} = cat(1, pooled_raw_shuff{g, a}, ep_raw_shuff(area_mask, :, :));
            pooled_z_shuff{g, a}   = cat(1, pooled_z_shuff{g, a}, ep_z_shuff(area_mask, :, :));
        end
    end
end

% =========================================================================
% PLOTTING: Dynamic Engine for 8 Neural Stability Figures
% =========================================================================
x_bases = {1:10, 12:21, 23:32};
x_ticks_centers = [5.5, 16.5, 27.5];
group_titles = {'Task', 'Control'};
area_colors = [0 0.4470 0.7410; 0.4660 0.6740 0.1880; 0.8500 0.3250 0.0980]; % DMS, DLS, ACC

agg_modes = {'Hierarchical', 'Pooled'};
met_modes = {'Raw', 'ZScored'};
scope_modes = {'WholePopulation', 'ByArea'};

for agg = 1:2
    for met = 1:2
        for scp = 1:2
            % Setup Figure Name
            fig_name = sprintf('Stability_%s_%s_%s', agg_modes{agg}, met_modes{met}, scope_modes{scp});
            figure('Name', fig_name, 'Position', [150, 100, 1000, 450], 'Color', 'w');
            
            % Determine Scope (Which areas to loop)
            if strcmp(scope_modes{scp}, 'WholePopulation')
                areas_to_plot = 4; % Just 'all'
                plot_colors = [0 0 0]; % Black
            else
                areas_to_plot = 1:3; % DMS, DLS, ACC
                plot_colors = area_colors;
            end
            
            ax = gobjects(1, 2);
            for g = 1:2
                ax(g) = subplot(1, 2, g); hold on;
                h_lines = gobjects(1, length(areas_to_plot));
                N_counts = nan(1, length(areas_to_plot));
                
                for a_idx = 1:length(areas_to_plot)
                    a = areas_to_plot(a_idx);
                    c_color = plot_colors(a_idx, :);
                    
                    % Get correct data matrix
                    if strcmp(agg_modes{agg}, 'Hierarchical')
                        if met == 1
                            data = squeeze(hier_raw(:, :, :, g, a));
                            data_shf = squeeze(hier_raw_shuff(:, :, :, g, a));
                        else
                            data = squeeze(hier_z(:, :, :, g, a));
                            data_shf = squeeze(hier_z_shuff(:, :, :, g, a));
                        end
                        % N is number of valid animals
                        valid_N = sum(~isnan(data(:, 1, 1)));
                    else
                        if met == 1
                            data = pooled_raw{g, a};
                            data_shf = pooled_raw_shuff{g, a};
                        else
                            data = pooled_z{g, a};
                            data_shf = pooled_z_shuff{g, a};
                        end
                        % N is number of valid units
                        valid_N = size(data, 1);
                    end
                    
                    if isempty(data) || valid_N == 0, continue; end
                    N_counts(a_idx) = valid_N;
                    
                    % Plot Epochs
                    first_plotted = false;
                    for e = 1:n_epochs
                        % Real Data (Solid)
                        ep_data = squeeze(data(:, e, :));
                        mu = mean(ep_data, 1, 'omitnan');
                        se = std(ep_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(ep_data), 1));
                        if ~all(isnan(mu))
                            h = shadedErrorBar(x_bases{e}, mu, se, 'lineProps', {'Color', c_color, 'LineWidth', 2});
                            if ~first_plotted && isfield(h, 'mainLine')
                                h_lines(a_idx) = h.mainLine;
                                first_plotted = true;
                            end
                        end
                        
                        % Shuffled Control Data (Dashed)
                        ep_shf = squeeze(data_shf(:, e, :));
                        mu_shf = mean(ep_shf, 1, 'omitnan');
                        se_shf = std(ep_shf, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(ep_shf), 1));
                        if ~all(isnan(mu_shf))
                            % If whole pop, use gray for shuffle. If by area, use same color but dashed.
                            if strcmp(scope_modes{scp}, 'WholePopulation')
                                shf_color = [0.6 0.6 0.6]; 
                            else
                                shf_color = c_color; 
                            end
                            shadedErrorBar(x_bases{e}, mu_shf, se_shf, 'lineProps', {'Color', shf_color, 'LineStyle', '--', 'LineWidth', 1.5});
                        end
                    end
                end
                
                xline([11, 22], 'k:');
                xticks(x_ticks_centers); xticklabels(epochs);
                
                % Format Title with N depending on scope and aggregation
                if strcmp(scope_modes{scp}, 'WholePopulation')
                    title(sprintf('%s - %s (N=%d)', group_titles{g}, met_modes{met}, N_counts(1)));
                else
                    title(sprintf('%s - %s', group_titles{g}, met_modes{met}));
                end
                ylabel(sprintf('Pearson r (%s)', agg_modes{agg}));
                
                % Generate Legends
                if g == 1 && any(isgraphics(h_lines))
                    if strcmp(scope_modes{scp}, 'WholePopulation')
                        legend(h_lines(isgraphics(h_lines)), 'All Units', 'Location', 'best');
                    else
                        leg_labels = arrayfun(@(x, n) sprintf('%s (N=%d)', upper(areas{x}), n), areas_to_plot, N_counts, 'UniformOutput', false);
                        legend(h_lines(isgraphics(h_lines)), leg_labels(isgraphics(h_lines)), 'Location', 'best');
                    end
                end
            end
            linkaxes(ax, 'xy');
            save_to_svg(fig_name);
        end
    end
end
fprintf('--- Stability Plotting Complete ---\n');


%% 6. Decoding of Position (Poisson ML) & Spatial Lick Patterns (Bin-by-Bin Log-Link)
fprintf('Running continuous decoding with ablation and shuffle controls...\n');

epochs = {'Naive', 'Intermediate', 'Expert'};
n_epochs = length(epochs);
max_trials = 300;
min_units = 3; % Enforce minimum units per combined network
lambda = 1.0; % Ridge regularization
trials_per_epoch = 10;
max_animals = max(n_animals_task, n_animals_ctrl);
cond_names = {'All', 'No-DMS', 'No-DLS', 'No-ACC', 'Shuffle'};
n_conds = length(cond_names);

% --- Trackers ---
bin_pos_err     = nan(max_animals, n_bins, 2, n_conds);
bin_pos_entropy = nan(max_animals, n_bins, 2, n_conds); 
ep_pos_err      = nan(max_animals, n_epochs, trials_per_epoch, 2, n_conds);
ep_pos_entropy  = nan(max_animals, n_epochs, trials_per_epoch, 2, n_conds); 
ep_lick_corr    = nan(max_animals, n_epochs, trials_per_epoch, 2, n_conds);

groups = {task_data, control_data};
for g = 1:2
    curr_data = groups{g};
    for i = 1:numel(curr_data)
        if g == 1
            lp = learning_points_task(i);
            
            % STRICT FILTER: Skip task animals that never learned
            if isnan(lp)
                continue;
            end
            
            dp = diseng_points_task(i);
            % Truncate Task up to disengagement point
            if ~isnan(dp) && dp > 0
                n_tr = min(size(curr_data(i).spatial_binned_fr_all, 3), dp);
            else
                n_tr = min(size(curr_data(i).spatial_binned_fr_all, 3), max_trials);
            end
        else
            lp = avg_lp;
            dp = avg_dp;
            n_tr = min(size(curr_data(i).spatial_binned_fr_all, 3), max_trials);
        end
        
        activity = curr_data(i).spatial_binned_fr_all(:, :, 1:n_tr); % [n_cells x n_bins x n_tr]
        n_cells_total = size(activity, 1);
        
        masks = struct('DMS', curr_data(i).is_dms, ...
                       'DLS', curr_data(i).is_dls, ...
                       'ACC', curr_data(i).is_acc);
        
        % Get spatial licking pattern (Lick Rate) [n_tr x n_bins]
        licks = curr_data(i).spatial_binned_data.licks(1:n_tr, :);
        durs = curr_data(i).spatial_binned_data.durations(1:n_tr, :);
        lick_pattern = licks ./ durs;
        
        % Clean behavioral data
        lick_pattern(isinf(lick_pattern)) = nan;
        cap_val = quantile(lick_pattern(:), 0.99);
        lick_pattern(lick_pattern > cap_val) = cap_val;
        
        % --- Define extraction indices for this animal (Only 3 Epochs) ---
        epoch_idx = cell(1, n_epochs);
        if n_tr >= trials_per_epoch, epoch_idx{1} = 1:trials_per_epoch; end
        if ~isnan(lp) && lp > trials_per_epoch && lp <= n_tr
            epoch_idx{2} = (lp - trials_per_epoch) : (lp - 1); 
        end
        if ~isnan(lp) && (lp + trials_per_epoch - 1) <= n_tr
            epoch_idx{3} = lp : (lp + trials_per_epoch - 1); 
        end
        
        % --- Run through conditions (Ablations & Shuffle) ---
        for c = 1:n_conds
            c_name = cond_names{c};
            is_shuffle = strcmp(c_name, 'Shuffle');
            
            % Determine active cell mask for this condition
            active_mask = true(n_cells_total, 1);
            if strcmp(c_name, 'No-DMS'), active_mask(masks.DMS) = false; end
            if strcmp(c_name, 'No-DLS'), active_mask(masks.DLS) = false; end
            if strcmp(c_name, 'No-ACC'), active_mask(masks.ACC) = false; end
            
            if sum(active_mask) < min_units, continue; end
            
            cond_data = activity(active_mask, :, :); % [cells x bins x trials]
            n_k_cells = sum(active_mask);
            
            % =========================================================
            % A. Spatial Decoding (Poisson Naive Bayes ML + Entropy)
            % =========================================================
            
            trial_pos_error     = nan(1, n_tr);
            trial_pos_entropy   = nan(1, n_tr); 
            trial_bin_errors    = nan(n_tr, n_bins);
            trial_bin_entropies = nan(n_tr, n_bins); 
            
            for t_test = 1:n_tr
                tr_train = setdiff(1:n_tr, t_test);
                
                lambda_x = mean(cond_data(:, :, tr_train), 3, 'omitnan') + 1e-6; 
                lambda_x(isnan(lambda_x)) = 1e-6; 
                
                % SHUFFLE: Randomize the spatial bins for each cell independently
                if is_shuffle
                    for cell_idx = 1:n_k_cells
                        lambda_x(cell_idx, :) = lambda_x(cell_idx, randperm(n_bins));
                    end
                end
                
                test_trial_data = cond_data(:, :, t_test); 
                pred_bins = nan(1, n_bins);
                bin_entropies = nan(1, n_bins); 
                
                for b = 1:n_bins
                    r = test_trial_data(:, b); 
                    r(isnan(r)) = 0; 
                    
                    % 1. Calculate Log-Likelihoods
                    LL = r' * log(lambda_x) - sum(lambda_x, 1);
                    
                    % 2. Convert LL to Posterior Probability (Log-Sum-Exp Trick)
                    LL_shifted = LL - max(LL); 
                    posterior = exp(LL_shifted) / sum(exp(LL_shifted));
                    
                    % 3. Calculate Shannon Entropy (bits)
                    bin_entropies(b) = -sum(posterior .* log2(posterior + eps));
                    
                    % 4. Find Maximum Likelihood Estimate
                    [~, pred_bins(b)] = max(LL);
                end
                
                % Errors & Entropy
                sq_errors = (pred_bins - (1:n_bins)).^2;
                trial_pos_error(t_test)     = sqrt(mean(sq_errors, 'omitnan')); 
                trial_bin_errors(t_test, :) = sqrt(sq_errors); 
                trial_pos_entropy(t_test)   = mean(bin_entropies, 'omitnan'); 
                trial_bin_entropies(t_test, :) = bin_entropies; 
            end
            
            % Store average bin error & entropy for this animal/condition
            bin_pos_err(i, :, g, c)     = mean(trial_bin_errors, 1, 'omitnan');
            bin_pos_entropy(i, :, g, c) = mean(trial_bin_entropies, 1, 'omitnan'); 
            
            % =========================================================
            % B. Spatial Lick Pattern Decoding (Bin-by-Bin Log-Link Ridge)
            % =========================================================
            
            trial_lick_corr = nan(1, n_tr);
            
            Y_lick = lick_pattern(1:n_tr, :);
            if is_shuffle
                % Shuffle trials to break relationship with neural data
                Y_lick = Y_lick(randperm(n_tr), :);
            end
            
            % Apply pseudo-Poisson link function
            Y_lick_log = log(Y_lick + 1); 
            Y_pred_full = nan(n_tr, n_bins);
            
            for b = 1:n_bins
                X_b = squeeze(cond_data(:, b, 1:n_tr))'; % [trials x cells]
                
                for t_test = 1:n_tr
                    tr_train = setdiff(1:n_tr, t_test);
                    
                    X_tr = X_b(tr_train, :);
                    Y_tr = Y_lick_log(tr_train, b);
                    
                    % Drop NaNs
                    valid_idx = ~any(isnan(X_tr), 2) & ~isnan(Y_tr);
                    if sum(valid_idx) < 5, continue; end
                    
                    X_tr = X_tr(valid_idx, :);
                    Y_tr = Y_tr(valid_idx);
                    
                    % Standardize Features
                    mu_X = mean(X_tr, 1, 'omitnan');
                    sig_X = std(X_tr, 0, 1, 'omitnan') + 1e-6;
                    X_tr_sc = (X_tr - mu_X) ./ sig_X;
                    
                    % Analytical Ridge 
                    W_b = (X_tr_sc' * X_tr_sc + lambda * eye(n_k_cells)) \ (X_tr_sc' * Y_tr);
                    b0_b = mean(Y_tr) - mean(X_tr_sc * W_b);
                    
                    % Predict Test Trial
                    X_te = X_b(t_test, :);
                    if any(isnan(X_te)), continue; end
                    X_te_sc = (X_te - mu_X) ./ sig_X;
                    
                    Y_pred_log = X_te_sc * W_b + b0_b;
                    
                    % Apply Inverse Link function
                    Y_pred_full(t_test, b) = exp(Y_pred_log) - 1; 
                end
            end
            
            % Correlate the full predicted spatial vector with the actual spatial vector per trial
            for t_test = 1:n_tr
                actual_licks = lick_pattern(t_test, :);
                pred_licks = Y_pred_full(t_test, :);
                
                % Enforce biological bound (no negative licks)
                pred_licks(pred_licks < 0) = 0;
                
                valid_b = ~isnan(pred_licks) & ~isnan(actual_licks);
                if sum(valid_b) > 3 && var(pred_licks(valid_b)) > 1e-6 && var(actual_licks(valid_b)) > 1e-6
                    trial_lick_corr(t_test) = corr(pred_licks(valid_b)', actual_licks(valid_b)');
                end
            end
            
            % =========================================================
            % C. Extract Epochs
            % =========================================================
            for e = 1:n_epochs
                idx = epoch_idx{e};
                if isempty(idx), continue; end
                take_n = min(length(idx), trials_per_epoch);
                
                ep_pos_err(i, e, 1:take_n, g, c)     = trial_pos_error(idx(1:take_n));
                ep_pos_entropy(i, e, 1:take_n, g, c) = trial_pos_entropy(idx(1:take_n)); 
                ep_lick_corr(i, e, 1:take_n, g, c)   = trial_lick_corr(idx(1:take_n));
            end
        end
    end
end

% --- Plotting Configurations ---
x_bases = {1:10, 12:21, 23:32};
x_ticks_centers = [5.5, 16.5, 27.5];
group_titles = {'Task (Up to Disengagement)', 'Control (All Trials)'};
% Colors for Conditions: All, No-DMS, No-DLS, No-ACC, Shuffle
cond_colors = [
    0 0 0;                  % Black
    0.8500 0.3250 0.0980;   % Orange (Missing DMS)
    0.9290 0.6940 0.1250;   % Yellow (Missing DLS)
    0.4940 0.1840 0.5560;   % Purple (Missing ACC)
    0.5 0.5 0.5             % Grey (Shuffle)
];

% =========================================================================
% FIGURE 1: Spatial Decoding Trial Evolution (RMSE vs Epochs)
% =========================================================================
figure('Name', 'Spatial Decoding Trial Evolution', 'Position', [100, 100, 1200, 500], 'Color', 'w');
ax_pos = gobjects(1, 2);
for g = 1:2
    ax_pos(g) = subplot(1, 2, g); hold on;
    h_lines = gobjects(1, n_conds);
    
    for c = 1:n_conds
        first_epoch_plotted = false;
        for e = 1:n_epochs
            data_matrix = squeeze(ep_pos_err(:, e, :, g, c)); % [Animals x 10]
            mu = mean(data_matrix, 1, 'omitnan');
            se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
            
            if all(isnan(mu)), continue; end
            
            h = shadedErrorBar(x_bases{e}, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
            if ~first_epoch_plotted && isfield(h, 'mainLine')
                h_lines(c) = h.mainLine;
                first_epoch_plotted = true;
            end
        end
    end
    xline([11, 22], 'k:');
    xticks(x_ticks_centers); xticklabels(epochs);
    title(sprintf('%s - Spatial ML Decoder', group_titles{g})); ylabel('RMSE (Bins)');
    set(gca, 'YDir', 'reverse'); % Downward means lower error / improvement
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_pos, 'xy'); 
save_to_svg('Decoding_Spatial_Trial_Evolution');

% =========================================================================
% FIGURE 1B: Spatial Certainty Trial Evolution (Entropy vs Epochs)
% =========================================================================
figure('Name', 'Spatial Certainty (Entropy) Evolution', 'Position', [100, 50, 1200, 500], 'Color', 'w');
ax_ent_pos = gobjects(1, 2);
for g = 1:2
    ax_ent_pos(g) = subplot(1, 2, g); hold on;
    h_lines = gobjects(1, n_conds);
    
    for c = 1:n_conds
        first_epoch_plotted = false;
        for e = 1:n_epochs
            data_matrix = squeeze(ep_pos_entropy(:, e, :, g, c)); % [Animals x 10]
            mu = mean(data_matrix, 1, 'omitnan');
            se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
            
            if all(isnan(mu)), continue; end
            
            h = shadedErrorBar(x_bases{e}, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
            if ~first_epoch_plotted && isfield(h, 'mainLine')
                h_lines(c) = h.mainLine;
                first_epoch_plotted = true;
            end
        end
    end
    xline([11, 22], 'k:');
    xticks(x_ticks_centers); xticklabels(epochs);
    title(sprintf('%s - Spatial Certainty', group_titles{g})); 
    ylabel('Shannon Entropy (bits)');
    set(gca, 'YDir', 'reverse'); % Downward means lower entropy / higher certainty
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_ent_pos, 'xy'); 
save_to_svg('Decoding_Spatial_Entropy_Evolution');

% =========================================================================
% FIGURE 2: Spatial Decoding Bin Error (RMSE vs Spatial Bins)
% =========================================================================
figure('Name', 'Spatial Decoding Error Profile across Corridor', 'Position', [150, 150, 1200, 500], 'Color', 'w');
ax_bin = gobjects(1, 2);
for g = 1:2
    ax_bin(g) = subplot(1, 2, g); hold on;
    h_lines = gobjects(1, n_conds);
    
    for c = 1:n_conds
        data_matrix = squeeze(bin_pos_err(:, :, g, c)); % [Animals x Bins]
        mu = mean(data_matrix, 1, 'omitnan');
        se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
        
        if all(isnan(mu)), continue; end
        
        h = shadedErrorBar(1:n_bins, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
        if isfield(h, 'mainLine')
            h_lines(c) = h.mainLine;
        end
    end
    
    if g == 1
        % ONLY plot landmarks for the Task group
        xline(20, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
        xline(25, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
    end
    
    xlabel('Spatial Bin'); ylabel('Average RMSE (Bins)');
    title(sprintf('%s - Spatial Error Profile', group_titles{g}));
    set(gca, 'YDir', 'reverse');
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_bin, 'xy');
save_to_svg('Decoding_Spatial_Bin_Error');

% =========================================================================
% FIGURE 2B: Spatial Certainty Bin Profile (Entropy vs Spatial Bins)
% =========================================================================
figure('Name', 'Spatial Certainty Profile across Corridor', 'Position', [150, 100, 1200, 500], 'Color', 'w');
ax_ent_bin = gobjects(1, 2);
for g = 1:2
    ax_ent_bin(g) = subplot(1, 2, g); hold on;
    h_lines = gobjects(1, n_conds);
    
    for c = 1:n_conds
        data_matrix = squeeze(bin_pos_entropy(:, :, g, c)); % [Animals x Bins]
        mu = mean(data_matrix, 1, 'omitnan');
        se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
        
        if all(isnan(mu)), continue; end
        
        h = shadedErrorBar(1:n_bins, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
        if isfield(h, 'mainLine')
            h_lines(c) = h.mainLine;
        end
    end
    
    if g == 1
        % ONLY plot landmarks for the Task group
        xline(20, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
        xline(25, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
    end
    
    xlabel('Spatial Bin'); ylabel('Average Shannon Entropy (bits)');
    title(sprintf('%s - Spatial Certainty Profile', group_titles{g}));
    set(gca, 'YDir', 'reverse'); % Downward means lower entropy / higher certainty
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_ent_bin, 'xy');
save_to_svg('Decoding_Spatial_Entropy_Bin_Profile');

% =========================================================================
% FIGURE 3: Lick Pattern Decoding (Bin-by-Bin Log-Link Ridge)
% =========================================================================
figure('Name', 'Lick Spatial Pattern Decoding Evolution', 'Position', [200, 200, 1200, 500], 'Color', 'w');
ax_lick = gobjects(1, 2);
for g = 1:2
    ax_lick(g) = subplot(1, 2, g); hold on;
    h_lines = gobjects(1, n_conds);
    
    for c = 1:n_conds
        first_epoch_plotted = false;
        for e = 1:n_epochs
            data_matrix = squeeze(ep_lick_corr(:, e, :, g, c)); % [Animals x 10]
            mu = mean(data_matrix, 1, 'omitnan');
            se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
            
            if all(isnan(mu)), continue; end
            
            h = shadedErrorBar(x_bases{e}, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
            if ~first_epoch_plotted && isfield(h, 'mainLine')
                h_lines(c) = h.mainLine;
                first_epoch_plotted = true;
            end
        end
    end
    xline([11, 22], 'k:');
    xticks(x_ticks_centers); xticklabels(epochs);
    title(sprintf('%s - Spatial Lick Pattern', group_titles{g})); ylabel('Pearson r (Predicted vs Actual Vector)');
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_lick, 'xy');
save_to_svg('Decoding_Lick_Spatial_Pattern');
fprintf('--- Decoding Plotting Complete ---\n');

%% Local Helper Functions
function r = calc_triu_corr(mat)
    % Calculates mean of upper triangle of correlation matrix
    if size(mat, 2) < 2
        r = nan; return;
    end
    rho = corrcoef(mat);
    idx = triu(true(size(rho)), 1);
    r = mean(rho(idx), 'omitnan');
end

function c = calc_triu_cos(mat)
    % Calculates mean of upper triangle of cosine similarity matrix
    if size(mat, 2) < 2
        c = nan; return;
    end
    norms = sqrt(sum(mat.^2, 1));
    norms(norms == 0) = 1e-10; % Avoid division by zero
    mat_norm = mat ./ norms;
    cos_sim = mat_norm' * mat_norm;
    idx = triu(true(size(cos_sim)), 1);
    c = mean(cos_sim(idx), 'omitnan');
end
