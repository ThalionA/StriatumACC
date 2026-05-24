%% Integrated Cortico-Striatal Spatiotemporal & Temporal Analysis Pipeline
% Unified pipeline for Task (g=1), Control 1 (g=2, Spatial), and Control 2 (g=3, Temporal)
% ALL groups are aligned to the Task Group's Average Learning Point (avg_lp)
% Y-axes are strictly linked across comparative subplots.
clearvars; clc; close all;

%% 1. Configuration & Data Loading
% Project-wide constants (paths, LP parameters, areas, colours, ...)
cfg = project_cfg();

% Local aliases for legacy variable names used downstream in this script.
cfg.lp_thresh_count = cfg.lp_min_consecutive;
cfg.lp_z_thresh     = cfg.lp_z_threshold;


mov_avg_window      = 10;
max_trials          = 300;
trials_per_epoch    = cfg.trials_per_epoch;

cfg.task_data_file     = 'processed_data/preprocessed_data2p5cm.mat';
cfg.control_data_file  = 'processed_data/preprocessed_data_control2p5cm.mat';

cfg.bin_size_au        = 2;
n_bins              = cfg.corridor_cm/(cfg.bin_size_au*1.25);

fprintf('Loading task, control 1 (spatial), and control 2 (temporal) data...\n');
S = load(cfg.task_data_file, 'preprocessed_data');
task_data = S.preprocessed_data;

S = load(cfg.control_data_file, 'preprocessed_data');
control1_data = S.preprocessed_data;

S = load(cfg.control2_data_file, 'preprocessed_data');
control2_data = S.preprocessed_data;
clear S

n_animals_task  = numel(task_data);
n_animals_ctrl1 = numel(control1_data);
n_animals_ctrl2 = numel(control2_data);
max_animals = max([n_animals_task, n_animals_ctrl1, n_animals_ctrl2]);

%% 2. Define Task Learning Points (LP) and Disengagement Points (DP)
% Refactored 2026-05-07 to use the shared find_learning_points helper.
% LP convention here is "start of qualifying window" (matches legacy
% processTaskData and the legacy learning_points_task script).
lp_cfg = struct('lp_z_threshold', cfg.lp_z_thresh, ...
                'lp_window',      cfg.lp_window, ...
                'lp_min_consecutive', cfg.lp_thresh_count);
[learning_points_task, ~] = find_learning_points(task_data, lp_cfg);

diseng_points_task = nan(1, n_animals_task);
for i = 1:n_animals_task
    if isfield(task_data(i), 'change_point_mean') && ~isempty(task_data(i).change_point_mean)
        diseng_points_task(i) = task_data(i).change_point_mean;
    end
end

avg_lp = round(mean(learning_points_task, 'omitnan'));
avg_dp = round(mean(diseng_points_task, 'omitnan'));
if isnan(avg_dp), avg_dp = avg_lp + 30; end

fprintf('Average Task Learning Point: Trial %d\n', avg_lp);

%% 3. Compute Temporal Binned Licks for Control 2 (On-the-fly)
fprintf('Computing temporal binned licks for Control 2 (dark habituation)...\n');
for i = 1:n_animals_ctrl2
    trialData = control2_data(i).trialData;
    n_trials = trialData.n_trials;
    % Match lick binning to the neural bin count of this Control 2
    % mouse (controls use coarser bins than the task — 2026-05-24 patch).
    n_bins_ctrl2 = size(control2_data(i).firing_rates_per_bin, 2);
    temp_licks = nan(n_trials, n_bins_ctrl2);
    
    for itrial = 1:n_trials
        times_in_trial = trialData.trial_times_zeroed{itrial};
        licks_in_trial = trialData.trial_licks{itrial}; 
        trial_duration = max(times_in_trial);
        
        if isempty(trial_duration) || trial_duration == 0, continue; end
        bin_edges = linspace(0, trial_duration, n_bins_ctrl2 + 1);
        
        for ibin = 1:n_bins_ctrl2
            bin_start = bin_edges(ibin);
            bin_end = bin_edges(ibin + 1);
            bin_duration = (bin_end - bin_start) / 1000; % Convert ms to s
            idx_in_bin = (times_in_trial >= bin_start) & (times_in_trial < bin_end);
            
            if any(idx_in_bin)
                if length(licks_in_trial) == length(times_in_trial) 
                    lick_count = sum(licks_in_trial(idx_in_bin) > 0);
                else 
                    lick_count = sum((licks_in_trial >= bin_start) & (licks_in_trial < bin_end));
                end
                temp_licks(itrial, ibin) = lick_count / bin_duration; 
            else
                temp_licks(itrial, ibin) = 0;
            end
        end
    end
    control2_data(i).temporal_binned_licks = temp_licks;
end

groups = {task_data, control1_data, control2_data};
group_names = {'Task (Spatial)', 'Control 1 (Spatial)', 'Control 2 (Temporal)'};

% --- Per-group bin counts (task is binned finer than the controls) ---
% The task uses 100 spatial bins, the controls 50. Every bin-indexed
% array and plot below takes its width from group_n_bins rather than a
% single global n_bins, so the pipeline is robust to the mismatch.
% (2026-05-24 patch.)
group_n_bins = nan(1, 3);
for g = 1:3
    for i = 1:numel(groups{g})
        if g < 3
            fr = groups{g}(i).spatial_binned_fr_all;
        else
            fr = groups{g}(i).firing_rates_per_bin;
        end
        group_n_bins(g) = max(group_n_bins(g), size(fr, 2));
    end
end
max_n_bins = max(group_n_bins);

%% 4. Plot Behavioral Evolution: Heatmaps
fprintf('--- Generating Behavioral Lick Heatmaps ---\n');

for g = 1:3
    curr_data = groups{g};
    fig_name_hm = sprintf('Behavioral_Lick_Heatmaps_Group%d', g);
    figure('Name', sprintf('%s Lick Rate Heatmaps', group_names{g}), 'Position', [50, 50, 1400, 800], 'Color', 'w');
    t_hm = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_hm, sprintf('%s - Lick Rate Heatmaps', group_names{g}), 'FontWeight', 'bold', 'FontSize', 14);
    
    for ianimal = 1:numel(curr_data)
        if g < 3
            licks = curr_data(ianimal).spatial_binned_data.licks;
            durations = curr_data(ianimal).spatial_binned_data.durations;
            lick_rate = licks ./ durations;
        else
            lick_rate = curr_data(ianimal).temporal_binned_licks;
        end
        
        cap_val = quantile(lick_rate(:), 0.99);
        lick_rate(lick_rate > cap_val) = cap_val;
        
        ax = nexttile(t_hm);
        imagesc(ax, lick_rate);
        
        if g < 3
            xline(cfg.visual_zone_cm/cfg.bin_size_au * group_n_bins(g)/n_bins, 'w--', 'LineWidth', 1.5); 
            xline(cfg.reward_zone_cm/cfg.bin_size_au * group_n_bins(g)/n_bins, 'w-', 'LineWidth', 1.5);  
            xlabel('Spatial Bin');
        else
            xlabel('Temporal Bin');
        end
        
        title(sprintf('Mouse %d', ianimal));
        ylabel('Trial');
    end
    cb = colorbar(ax); cb.Label.String = 'Lick Rate (Hz)'; cb.Layout.Tile = 'east';
    save_to_svg(fig_name_hm);
end

%% 4.1. Task Mice: All Z-Scored Lick Errors
fprintf('--- Generating Z-Scored Lick Errors (Task Group) ---\n');
figure('Name', 'Task Group - Z-Scored Lick Errors', 'Position', [100, 100, 1400, 800], 'Color', 'w');
t_err = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
title(t_err, 'Task Group - Z-Scored Lick Errors (Smoothed)', 'FontWeight', 'bold', 'FontSize', 14);

for ianimal = 1:n_animals_task
    z_errors = task_data(ianimal).zscored_lick_errors;
    z_errors(isinf(z_errors)) = nan;
    
    z_smoothed = movmean(z_errors, mov_avg_window, 'omitnan');
    z_se = movstd(z_errors, mov_avg_window, 'omitnan') ./ sqrt(mov_avg_window);
    
    invalid = z_smoothed > 10;
    z_smoothed(invalid) = nan; z_se(invalid) = nan;
    
    nexttile(t_err); hold on;
    yline(0, 'r--'); yline(cfg.lp_z_thresh, 'b--');
    
    shadedErrorBar(1:length(z_smoothed), z_smoothed, z_se, 'lineProps', {'Color', 'k', 'LineWidth', 1.5});
    
    lp = learning_points_task(ianimal);
    if ~isnan(lp), xline(lp, 'g-', 'LP', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom'); end
    
    title(sprintf('Mouse %d', ianimal)); xlabel('Trial'); ylabel('Z-Score Error');
    grid on; box on; xlim([1 length(z_smoothed)]);
end
save_to_svg('Behavioral_ZError_AllTask');

%% 4.2. Task Mouse 3: Heatmap + Vertical Z-Error
fprintf('--- Generating Mouse 3 Aligned Heatmap & Vertical Error ---\n');
if n_animals_task >= 3
    figure('Name', 'Task Mouse 3 - Vertical Error Alignment', 'Position', [150, 150, 800, 600], 'Color', 'w');
    
    % Heatmap
    ax1 = subplot(1, 4, 1:3);
    licks = task_data(3).spatial_binned_data.licks;
    durs = task_data(3).spatial_binned_data.durations;
    lr = licks ./ durs;
    lr(isinf(lr)) = nan;
    cap_val = quantile(lr(:), 0.99); lr(lr > cap_val) = cap_val;
    
    imagesc(ax1, lr);
    xline(ax1, cfg.visual_zone_cm/cfg.bin_size_au, 'w--', 'Visual', 'LabelVerticalAlignment', 'bottom'); 
    xline(ax1, cfg.reward_zone_cm/cfg.bin_size_au, 'w-', 'Reward', 'LabelVerticalAlignment', 'bottom');
    xlabel('Spatial Bin'); ylabel('Trial'); title('Mouse 3 Lick Rate');
    colorbar
    
    % Vertical Z-Error
    ax2 = subplot(1, 4, 4); hold on;
    z_errors = task_data(3).zscored_lick_errors;
    z_errors(isinf(z_errors)) = nan;
    z_smoothed = movmean(z_errors, mov_avg_window, 'omitnan');
    
    plot(ax2, z_smoothed, 1:length(z_smoothed), 'k', 'LineWidth', 1.5);
    xline(ax2, cfg.lp_z_thresh, 'b--', 'Threshold');
    
    lp3 = learning_points_task(3);
    if ~isnan(lp3), yline(ax2, lp3, 'g-', 'LP', 'LineWidth', 2); end
    
    set(ax2, 'YDir', 'reverse'); % Match imagesc Y-direction
    ylim(ax2, [1 size(lr, 1)]); 
    xlabel('Z-Score Error'); title('Smoothed Error');
    linkaxes([ax1, ax2], 'y'); % Crucial for alignment
    save_to_svg('Behavioral_Mouse3_VerticalAlign');
end

%% 4.3. Z-Scored Errors across Epochs (Learners Avg & Mouse 3)
fprintf('--- Generating Z-Scored Errors across Epochs ---\n');
epochs = {'Naive', 'Intermediate', 'Expert'};
n_epochs = length(epochs);
trials_per_epoch = 10;

ep_z_errors = nan(n_animals_task, n_epochs, trials_per_epoch);

for i = 1:n_animals_task
    lp = learning_points_task(i);
    if isnan(lp), continue; end % Only include successful learners
    
    z = task_data(i).zscored_lick_errors;
    z(isinf(z)) = nan;
    n_tr = length(z);
    
    % Define standard epochs
    epoch_idx = cell(1, n_epochs);
    if n_tr >= trials_per_epoch, epoch_idx{1} = 1:trials_per_epoch; end
    if ~isnan(lp) && lp > trials_per_epoch && lp <= n_tr, epoch_idx{2} = (lp - trials_per_epoch) : (lp - 1); end
    if ~isnan(lp) && (lp + trials_per_epoch - 1) <= n_tr, epoch_idx{3} = lp : (lp + trials_per_epoch - 1); end
    
    for e = 1:n_epochs
        idx = epoch_idx{e};
        if isempty(idx), continue; end
        take_n = min(length(idx), trials_per_epoch);
        ep_z_errors(i, e, 1:take_n) = z(idx(1:take_n));
    end
end

x_bases_z = {1:10, 12:21, 23:32};
x_ticks_centers_z = [5.5, 16.5, 27.5];

figure('Name', 'Z-Scored Errors by Epoch', 'Position', [200, 200, 1000, 400], 'Color', 'w');
ax_z = gobjects(1, 2);

% Subplot 1: Learners Average
ax_z(1) = subplot(1, 2, 1); hold on;
for e = 1:n_epochs
    data_matrix = squeeze(ep_z_errors(:, e, :));
    mu_z = mean(data_matrix, 1, 'omitnan');
    se_z = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
    if all(isnan(mu_z)), continue; end
    shadedErrorBar(x_bases_z{e}, mu_z, se_z, 'lineProps', {'Color', 'k', 'LineWidth', 2});
end
xline([11, 22], 'k:'); xticks(x_ticks_centers_z); xticklabels(epochs);
yline(0, 'r--'); 
yline(cfg.lp_z_thresh, 'b--', 'Threshold', 'LabelHorizontalAlignment', 'left');
title('Learners Average (Epochs)'); ylabel('Z-Score Error');

% Subplot 2: Mouse 3
if n_animals_task >= 3 && ~isnan(learning_points_task(3))
    ax_z(2) = subplot(1, 2, 2); hold on;
    for e = 1:n_epochs
        data_matrix = squeeze(ep_z_errors(3, e, :));
        if all(isnan(data_matrix)), continue; end
        plot(x_bases_z{e}, data_matrix', 'k', 'LineWidth', 1.5);
    end
    xline([11, 22], 'k:'); xticks(x_ticks_centers_z); xticklabels(epochs);
    yline(0, 'r--'); 
    yline(cfg.lp_z_thresh, 'b--', 'Threshold', 'LabelHorizontalAlignment', 'left');
    title('Task Mouse 3 (Epochs)'); ylabel('Z-Score Error');
end

linkaxes(ax_z, 'y'); % Link Y-axes for direct visual comparison
save_to_svg('Behavioral_Epoch_ZError');

%% 4.4. Behavioral Stability Evolution (3 Epochs)
fprintf('--- Generating Behavioral Stability Evolution (Epochs) ---\n');
window_size_beh = 5; 
half_win_beh = floor(window_size_beh/2);

ep_stab_licks = nan(max_animals, n_epochs, trials_per_epoch, 3);
ep_stab_vel   = nan(max_animals, n_epochs, trials_per_epoch, 3);

for g = 1:3
    curr_data = groups{g};
    for i = 1:numel(curr_data)
        
        % 1. Get LP for alignment and filter learners
        if g == 1
            lp = learning_points_task(i);
            if isnan(lp)
                continue; % Skip non-learners in task group
            end
        else
            lp = avg_lp; % Use average Task LP for all controls
        end
        
        % 2. Extract Data
        if g < 3
            licks = curr_data(i).spatial_binned_data.licks;
            durs  = curr_data(i).spatial_binned_data.durations;
            vel   = (cfg.bin_size_au * 1.25) ./ durs;
            vel(isinf(vel)) = nan;
        else
            licks = curr_data(i).temporal_binned_licks;
            durs  = ones(size(licks));
            vel   = nan(size(licks));
        end
        
        lr = licks ./ durs;
        lr(isinf(lr)) = nan;
        n_tr = size(lr, 1);
        
        % 3. Calculate continuous stability
        stab_licks_raw = nan(1, n_tr);
        stab_vel_raw   = nan(1, n_tr);
        
        for t = 1:n_tr
            win_idx = max(1, t - half_win_beh) : min(n_tr, t + half_win_beh);
            if length(win_idx) < 2, continue; end
            
            stab_licks_raw(t) = calc_triu_corr(lr(win_idx, :)'); 
            if g < 3
                stab_vel_raw(t) = calc_triu_corr(vel(win_idx, :)');
            end
        end
        
        % 4. Extract into Epochs
        epoch_idx = cell(1, n_epochs);
        if n_tr >= trials_per_epoch, epoch_idx{1} = 1:trials_per_epoch; end
        if ~isnan(lp) && lp > trials_per_epoch && lp <= n_tr, epoch_idx{2} = (lp - trials_per_epoch) : (lp - 1); end
        if ~isnan(lp) && (lp + trials_per_epoch - 1) <= n_tr, epoch_idx{3} = lp : (lp + trials_per_epoch - 1); end
        
        for e = 1:n_epochs
            idx = epoch_idx{e};
            if isempty(idx), continue; end
            take_n = min(length(idx), trials_per_epoch);
            
            ep_stab_licks(i, e, 1:take_n, g) = stab_licks_raw(idx(1:take_n));
            if g < 3
                ep_stab_vel(i, e, 1:take_n, g) = stab_vel_raw(idx(1:take_n));
            end
        end
    end
end

% 5. Plotting
x_bases_beh = {1:10, 12:21, 23:32};
x_ticks_centers_beh = [5.5, 16.5, 27.5];

figure('Name', 'Behavioral Stability Evolution (Epochs)', 'Position', [100, 100, 1500, 600], 'Color', 'w');
ax_stab_licks = gobjects(1, 3);
ax_stab_vel   = gobjects(1, 2);

for g = 1:3
    % --- Lick Pattern Stability ---
    ax_stab_licks(g) = subplot(2, 3, g); hold on;
    for e = 1:n_epochs
        data_matrix = squeeze(ep_stab_licks(:, e, :, g));
        mu = mean(data_matrix, 1, 'omitnan');
        se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
        if all(isnan(mu)), continue; end
        shadedErrorBar(x_bases_beh{e}, mu, se, 'lineProps', {'Color', [0.85 0.32 0.09], 'LineWidth', 2});
    end
    
    xline([11, 22], 'k:'); xticks(x_ticks_centers_beh); xticklabels(epochs);
    if g == 1
        title(sprintf('%s (Learners) - Lick Stability', group_names{g}));
    else
        title(sprintf('%s - Lick Stability', group_names{g}));
    end
    ylabel('Pearson r');
    
    % --- Velocity Pattern Stability ---
    if g < 3
        ax_stab_vel(g) = subplot(2, 3, g+3); hold on;
        for e = 1:n_epochs
            data_matrix = squeeze(ep_stab_vel(:, e, :, g));
            mu = mean(data_matrix, 1, 'omitnan');
            se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
            if all(isnan(mu)), continue; end
            shadedErrorBar(x_bases_beh{e}, mu, se, 'lineProps', {'Color', [0 0.4470 0.7410], 'LineWidth', 2});
        end
        
        xline([11, 22], 'k:'); xticks(x_ticks_centers_beh); xticklabels(epochs);
        if g == 1
            title(sprintf('%s (Learners) - Velocity Stability', group_names{g}));
        else
            title(sprintf('%s - Velocity Stability', group_names{g}));
        end
        ylabel('Pearson r');
    else
        subplot(2, 3, g+3); axis off; title('No Velocity (Wheel Blocked)');
    end
end

% Enforce shared Y-axes for fair magnitude comparisons
linkaxes(ax_stab_licks, 'y');
linkaxes(ax_stab_vel, 'y');
save_to_svg('Behavioral_Stability_AllGroups_Epochs');

%% 5. Extract Epoch Data (Behavior - Yoked to LP)
fprintf('Extracting behavioral data across yoked epochs...\n');
epochs = {'Naive', 'Intermediate', 'Expert'};
n_epochs = length(epochs);

% Bin-indexed trackers sized to the largest group (see group_n_bins); each
% group fills only its own 1:nb range (2026-05-24 bin-size patch).
avg_licks = nan(max_animals, max_n_bins, n_epochs, 3);
avg_vel   = nan(max_animals, max_n_bins, n_epochs, 3);

for g = 1:3
    curr_data = groups{g};
    for i = 1:numel(curr_data)
        if g < 3
            n_tr = size(curr_data(i).spatial_binned_fr_all, 3);
            licks = curr_data(i).spatial_binned_data.licks(1:n_tr, :);
            durs  = curr_data(i).spatial_binned_data.durations(1:n_tr, :);
            vel   = (cfg.bin_size_au * 1.25) ./ durs;
            vel(isinf(vel)) = nan;
        else
            n_tr = size(curr_data(i).firing_rates_per_bin, 3);
            licks = curr_data(i).temporal_binned_licks(1:n_tr, :);
            durs  = ones(size(licks)); 
            vel   = nan(size(licks));  
        end
        
        lick_rate = licks ./ durs;
        lick_rate(isinf(lick_rate)) = nan;
        nb = size(lick_rate, 2);   % this group's bin count
        
        if g == 1
            lp = learning_points_task(i);
        else
            lp = avg_lp; 
        end
        
        epoch_idx = cell(1, n_epochs);
        if n_tr >= trials_per_epoch, epoch_idx{1} = 1:trials_per_epoch; end
        if ~isnan(lp) && lp > trials_per_epoch && lp <= n_tr, epoch_idx{2} = (lp - trials_per_epoch) : (lp - 1); end
        if ~isnan(lp) && (lp + trials_per_epoch - 1) <= n_tr, epoch_idx{3} = lp : (lp + trials_per_epoch - 1); end
        
        for e = 1:n_epochs
            idx = epoch_idx{e};
            if ~isempty(idx) && max(idx) <= size(lick_rate, 1)
                avg_licks(i, 1:nb, e, g) = mean(lick_rate(idx, :), 1, 'omitnan');
                avg_vel(i, 1:nb, e, g)   = mean(vel(idx, :), 1, 'omitnan');
            end
        end
    end
end

% --- Plotting Behaviour (3 Columns with Linked Axes) ---
% Each group is plotted on its own bin axis; landmarks scale per group.
figure('Name', 'Behavioural Evolution across Yoked Epochs', 'Position', [100, 100, 1500, 600], 'Color', 'w');
colors = lines(n_epochs);

ax_lick = gobjects(1, 3);
ax_vel = gobjects(1, 2);

for g = 1:3
    n_bins_g = group_n_bins(g);   % this group's bin count

    % Lick Rate (Top Row)
    ax_lick(g) = subplot(2, 3, g); hold on; 
    title(sprintf('%s - Lick Rate', group_names{g}));
    h_lines = gobjects(1, n_epochs);
    
    for e = 1:n_epochs
        valid_data = squeeze(avg_licks(:, 1:n_bins_g, e, g));
        if all(isnan(valid_data(:))), continue; end
        mu = mean(valid_data, 1, 'omitnan');
        se = std(valid_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(valid_data), 1));
        h = shadedErrorBar(1:n_bins_g, mu, se, 'lineProps', {'Color', colors(e,:), 'LineWidth', 2});
        if isfield(h, 'mainLine'), h_lines(e) = h.mainLine; end
    end
    
    if g < 3
        xline(cfg.visual_zone_cm/cfg.bin_size_au * n_bins_g/n_bins, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom'); 
        xline(cfg.reward_zone_cm/cfg.bin_size_au * n_bins_g/n_bins, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom');
        xlabel('Spatial Bin');
    else
        xlabel('Temporal Bin');
    end
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), epochs, 'Location', 'best'); end
    ylabel('Lick Rate (Hz)');
    
    % Velocity (Bottom Row)
    if g < 3
        ax_vel(g) = subplot(2, 3, g+3); hold on; 
        title(sprintf('%s - Velocity', group_names{g}));
        for e = 1:n_epochs
            valid_data = squeeze(avg_vel(:, 1:n_bins_g, e, g));
            if all(isnan(valid_data(:))), continue; end
            mu = mean(valid_data, 1, 'omitnan');
            se = std(valid_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(valid_data), 1));
            shadedErrorBar(1:n_bins_g, mu, se, 'lineProps', {'Color', colors(e,:), 'LineWidth', 2});
        end
        xlabel('Spatial Bin'); ylabel('Velocity (cm/s)');
        xline(cfg.visual_zone_cm/cfg.bin_size_au * n_bins_g/n_bins, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom'); 
        xline(cfg.reward_zone_cm/cfg.bin_size_au * n_bins_g/n_bins, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom');
    else
        subplot(2, 3, g+3);
        axis off; title('No Velocity (Wheel Blocked)');
    end
end
% Link the Y-axes for behavioral plots
linkaxes(ax_lick, 'y');
linkaxes(ax_vel, 'y');
save_to_svg('Behavioural_Evolution_3Groups_Yoked');

%% 6. Trial-to-Trial Correlation (Neural Single-Neuron Reliability)
fprintf('Computing continuous single-neuron reliability...\n');
% V1/CA1 added (2026-05-07/08); DG dropped 2026-05-24 (excluded from all
% figures/analyses). 'all' stays last so the "All Units" semantics are
% preserved as the final element regardless of how many real areas exist.
% To add another area: append to this list, append a colour to
% area_colors, and extend the masks struct in section 7.
% Only the z-scored reliability metric is computed/plotted (raw dropped
% 2026-05-24).
areas = {'dms', 'dls', 'acc', 'v1', 'ca1', 'all'};
n_areas = length(areas);
all_idx = n_areas; % index of the synthetic "all units" pseudo-area
window_size = 5;
half_win = floor(window_size/2);
min_units = 5;

hier_z       = nan(max_animals, n_epochs, trials_per_epoch, 3, n_areas);
hier_z_shuff = nan(max_animals, n_epochs, trials_per_epoch, 3, n_areas);

pooled_z       = cell(3, n_areas);
pooled_z_shuff = cell(3, n_areas);

for g = 1:3
    curr_data = groups{g};
    for i = 1:numel(curr_data)

        if g == 1
            lp = learning_points_task(i);
            if isnan(lp), continue; end
            activity_raw = curr_data(i).spatial_binned_fr_all;
        elseif g == 2
            lp = avg_lp;
            activity_raw = curr_data(i).spatial_binned_fr_all;
        else
            lp = avg_lp;
            activity_raw = curr_data(i).firing_rates_per_bin;
        end

        activity_raw(isnan(activity_raw)) = 0;
        % Drop DG units — excluded from all figures/analyses (2026-05-24).
        keep_cells = ~is_area_safe(curr_data(i), 'DG');
        activity_raw = activity_raw(keep_cells, :, :);
        n_trials = size(activity_raw, 3);
        n_cells_total = size(activity_raw, 1);
        v1_mask  = is_area_safe(curr_data(i), 'V1');
        ca1_mask = is_area_safe(curr_data(i), 'CA1');
        masks = {curr_data(i).is_dms(keep_cells), ...
                 curr_data(i).is_dls(keep_cells), ...
                 curr_data(i).is_acc(keep_cells), ...
                 v1_mask(keep_cells), ...
                 ca1_mask(keep_cells)};

        activity_z = nan(size(activity_raw));
        for c = 1:n_cells_total
            unit_data = squeeze(activity_raw(c, :, :));
            mu = mean(unit_data(:), 'omitnan');
            sig = std(unit_data(:), 'omitnan');
            if sig > 0
                activity_z(c, :, :) = (unit_data - mu) / sig;
            else
                activity_z(c, :, :) = 0;
            end
        end

        shuff_idx = randperm(n_trials);
        activity_z_shuff = activity_z(:, :, shuff_idx);

        stab_z       = nan(n_cells_total, n_trials);
        stab_z_shuff = nan(n_cells_total, n_trials);

        % Vectorised per-cell triu-corr-mean (2026-05-07). ~50-100x faster
        % than the previous nested for-loop. See batch_triu_corr_mean.m.
        for t = 1:n_trials
            win_idx = max(1, t - half_win) : min(n_trials, t + half_win);
            if length(win_idx) < 2, continue; end
            stab_z(:, t)       = batch_triu_corr_mean(activity_z(:, :, win_idx));
            stab_z_shuff(:, t) = batch_triu_corr_mean(activity_z_shuff(:, :, win_idx));
        end

        epoch_idx = cell(1, n_epochs);
        if n_trials >= trials_per_epoch, epoch_idx{1} = 1:trials_per_epoch; end
        if ~isnan(lp) && lp > trials_per_epoch && lp <= n_trials, epoch_idx{2} = (lp - trials_per_epoch) : (lp - 1); end
        if ~isnan(lp) && (lp + trials_per_epoch - 1) <= n_trials, epoch_idx{3} = lp : (lp + trials_per_epoch - 1); end

        ep_z       = nan(n_cells_total, n_epochs, trials_per_epoch);
        ep_z_shuff = nan(n_cells_total, n_epochs, trials_per_epoch);

        for e = 1:n_epochs
            idx = epoch_idx{e};
            if isempty(idx), continue; end
            take_n = min(length(idx), trials_per_epoch);
            ep_z(:, e, 1:take_n)       = stab_z(:, idx(1:take_n));
            ep_z_shuff(:, e, 1:take_n) = stab_z_shuff(:, idx(1:take_n));
        end

        for a = 1:n_areas
            if a == all_idx, area_mask = true(n_cells_total, 1); else, area_mask = masks{a}; end
            if sum(area_mask) < min_units, continue; end

            hier_z(i, :, :, g, a)       = mean(ep_z(area_mask, :, :), 1, 'omitnan');
            hier_z_shuff(i, :, :, g, a) = mean(ep_z_shuff(area_mask, :, :), 1, 'omitnan');

            pooled_z{g, a}       = cat(1, pooled_z{g, a}, ep_z(area_mask, :, :));
            pooled_z_shuff{g, a} = cat(1, pooled_z_shuff{g, a}, ep_z_shuff(area_mask, :, :));
        end
    end
end

% --- Plot Stability Engines: one panel per area (rows) x group (columns) ---
% Z-scored metric only (raw dropped 2026-05-24). Each area — including the
% 'all units' pseudo-area as the bottom row — gets its own row; the three
% task/control groups are the columns. Y-axes are linked across the whole
% grid so areas and groups stay directly comparable.
agg_modes = {'Hierarchical', 'Pooled'};
x_bases_stab = {1:10, 12:21, 23:32};
x_ticks_centers_stab = [5.5, 16.5, 27.5];
stab_ylim_top = 0.5;   % cap reliability y-axis — the full auto-range (~1) squishes the data
area_colors = [0 0.4470 0.7410; 0.4660 0.6740 0.1880; 0.8500 0.3250 0.0980; ...
               0.4940 0.1840 0.5560; 0.8 0.1 0.2];   % DMS DLS ACC V1 CA1
all_color = [0 0 0];                                 % 'all units' pseudo-area

for agg = 1:2
    fig_name = sprintf('Stability_AllGroups_%s_ZScored', agg_modes{agg});
    figure('Name', fig_name, 'Position', [120, 60, 1250, 200*n_areas], 'Color', 'w');
    t_stab = tiledlayout(n_areas, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_stab, sprintf('Single-Neuron Reliability — Z-Scored (%s)', agg_modes{agg}), ...
          'FontWeight', 'bold', 'FontSize', 14);
    xlabel(t_stab, 'Epoch');

    ax_stab = gobjects(n_areas, 3);
    legend_done = false;

    for a = 1:n_areas
        if a == all_idx, c_color = all_color; else, c_color = area_colors(a, :); end

        for g = 1:3
            ax_stab(a, g) = nexttile; hold on;

            if strcmp(agg_modes{agg}, 'Hierarchical')
                data     = squeeze(hier_z(:, :, :, g, a));
                data_shf = squeeze(hier_z_shuff(:, :, :, g, a));
                valid_N  = sum(~isnan(data(:, 1, 1)));
            else
                data     = pooled_z{g, a};
                data_shf = pooled_z_shuff{g, a};
                valid_N  = size(data, 1);
            end

            if ~isempty(data) && valid_N > 0
                h_obs = gobjects(1);
                h_shf = gobjects(1);
                for e = 1:n_epochs
                    ep_data = squeeze(data(:, e, :));
                    mu = mean(ep_data, 1, 'omitnan');
                    se = std(ep_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(ep_data), 1));
                    if ~all(isnan(mu))
                        h = shadedErrorBar(x_bases_stab{e}, mu, se, 'lineProps', {'Color', c_color, 'LineWidth', 2});
                        if isfield(h, 'mainLine'), h_obs = h.mainLine; end
                    end

                    ep_shf = squeeze(data_shf(:, e, :));
                    mu_shf = mean(ep_shf, 1, 'omitnan');
                    se_shf = std(ep_shf, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(ep_shf), 1));
                    if ~all(isnan(mu_shf))
                        h = shadedErrorBar(x_bases_stab{e}, mu_shf, se_shf, 'lineProps', {'Color', [0.6 0.6 0.6], 'LineStyle', '--', 'LineWidth', 1.5});
                        if isfield(h, 'mainLine'), h_shf = h.mainLine; end
                    end
                end
                text(0.03, 0.92, sprintf('N=%d', valid_N), 'Units', 'normalized', ...
                     'FontSize', 8, 'Color', [0.3 0.3 0.3]);

                xline([11, 22], 'k:'); xticks(x_ticks_centers_stab); xticklabels(epochs);
                box on;

                if ~legend_done && isgraphics(h_obs) && isgraphics(h_shf)
                    legend([h_obs, h_shf], {'Observed', 'Shuffle'}, 'Location', 'best');
                    legend_done = true;
                end
            else
                text(0.5, 0.5, 'No data', 'Units', 'normalized', ...
                     'HorizontalAlignment', 'center', 'Color', [0.6 0.6 0.6]);
            end

            

            if a == 1, title(group_names{g}); end
            if g == 1, ylabel(sprintf('%s\nPearson r', upper(areas{a})), 'FontWeight', 'bold'); end
        end
    end

    linkaxes(ax_stab, 'y');   % Link Y-axes across the whole grid
    yl = ylim(ax_stab(1));    % shared auto-range after linking
    ylim(ax_stab(1), [yl(1), stab_ylim_top]);   % cap the top; propagates via linkaxes
    save_to_svg(fig_name);
end

%% 7. Decoding Space/Time & Lick Patterns (Poisson ML + Ridge Log-Link)
fprintf('Running Spatial/Temporal ML Decoding and Lick Ridge... \n');
lambda = 1.0; 
cond_names = {'All', 'No-DMS', 'No-DLS', 'No-ACC', 'No-V1', 'No-CA1', 'Shuffle'};
n_conds = length(cond_names);

% Bin counts differ per group (the task uses finer spatial bins than the
% controls — 2026-05-24 patch). n_bins_i below is derived per mouse and
% drives every bin loop/array; bin-indexed trackers are sized to the
% largest count and each group fills only its own range. Decoder metrics
% are normalised (RMSE by n_bins, entropy by log2(n_bins)) so the groups
% stay comparable despite the different binning.

% --- Trackers ---
bin_pos_err     = nan(max_animals, max_n_bins, 3, n_conds);
bin_pos_entropy = nan(max_animals, max_n_bins, 3, n_conds); 
ep_pos_err      = nan(max_animals, n_epochs, trials_per_epoch, 3, n_conds);
ep_pos_entropy  = nan(max_animals, n_epochs, trials_per_epoch, 3, n_conds); 
ep_lick_corr    = nan(max_animals, n_epochs, trials_per_epoch, 3, n_conds);

for g = 1:3
    curr_data = groups{g};
    for i = 1:numel(curr_data)
        
        if g == 1
            lp = learning_points_task(i);
            if isnan(lp), continue; end
            dp = diseng_points_task(i);
            n_tr = min(size(curr_data(i).spatial_binned_fr_all, 3), dp);
            activity = curr_data(i).spatial_binned_fr_all(:, :, 1:n_tr);
            Y_lick = curr_data(i).spatial_binned_data.licks(1:n_tr,:) ./ curr_data(i).spatial_binned_data.durations(1:n_tr,:);
        elseif g == 2
            lp = avg_lp;
            n_tr = min(size(curr_data(i).spatial_binned_fr_all, 3), max_trials);
            activity = curr_data(i).spatial_binned_fr_all(:, :, 1:n_tr);
            Y_lick = curr_data(i).spatial_binned_data.licks(1:n_tr,:) ./ curr_data(i).spatial_binned_data.durations(1:n_tr,:);
        else
            lp = avg_lp;
            n_tr = min(size(curr_data(i).firing_rates_per_bin, 3), max_trials);
            activity = curr_data(i).firing_rates_per_bin(:, :, 1:n_tr);
            Y_lick = curr_data(i).temporal_binned_licks(1:n_tr, :);
        end
        
        activity(isnan(activity)) = 0;
        Y_lick(isinf(Y_lick)) = nan;
        Y_lick(Y_lick > quantile(Y_lick(:), 0.99)) = quantile(Y_lick(:), 0.99);

        % Drop DG units — excluded from all figures/analyses (2026-05-24).
        keep_cells = ~is_area_safe(curr_data(i), 'DG');
        activity = activity(keep_cells, :, :);
        n_cells_total = size(activity, 1);

        % Bin count for THIS mouse — drives every bin loop below, so the
        % section is robust to task (100 bins) vs controls (50 bins).
        n_bins_i = size(activity, 2);

        % The lick ridge regresses lick-in-bin-b on neural-in-bin-b, so the
        % two must share a binning. Skip lick decoding (and warn) for any
        % mouse where they do not, rather than silently mis-pairing bins.
        lick_bins_ok = (size(Y_lick, 2) == n_bins_i);
        if ~lick_bins_ok
            warning('IntegratedAll:LickBinMismatch', ...
                    'Group %d mouse %d: %d lick bins vs %d neural bins — skipping lick decoding.', ...
                    g, i, size(Y_lick, 2), n_bins_i);
        end

        v1_mask  = is_area_safe(curr_data(i), 'V1');
        ca1_mask = is_area_safe(curr_data(i), 'CA1');
        masks = struct('DMS', curr_data(i).is_dms(keep_cells), ...
                       'DLS', curr_data(i).is_dls(keep_cells), ...
                       'ACC', curr_data(i).is_acc(keep_cells), ...
                       'V1',  v1_mask(keep_cells), ...
                       'CA1', ca1_mask(keep_cells));
        
        epoch_idx = cell(1, n_epochs);
        if n_tr >= trials_per_epoch, epoch_idx{1} = 1:trials_per_epoch; end
        if ~isnan(lp) && lp > trials_per_epoch && lp <= n_tr, epoch_idx{2} = (lp - trials_per_epoch) : (lp - 1); end
        if ~isnan(lp) && (lp + trials_per_epoch - 1) <= n_tr, epoch_idx{3} = lp : (lp + trials_per_epoch - 1); end
        
        for c = 1:n_conds
            c_name = cond_names{c};
            is_shuffle = strcmp(c_name, 'Shuffle');
            
            active_mask = true(n_cells_total, 1);
            if strcmp(c_name, 'No-DMS'), active_mask(masks.DMS) = false; end
            if strcmp(c_name, 'No-DLS'), active_mask(masks.DLS) = false; end
            if strcmp(c_name, 'No-ACC'), active_mask(masks.ACC) = false; end
            if strcmp(c_name, 'No-V1'),  active_mask(masks.V1)  = false; end
            if strcmp(c_name, 'No-CA1'), active_mask(masks.CA1) = false; end

            if sum(active_mask) < min_units, continue; end
            
            cond_data = activity(active_mask, :, :); 
            n_k_cells = sum(active_mask);
            
            % --- A. Spatial/Temporal Decoding (Poisson ML) ---
            % RMSE is normalised by n_bins_i (fraction of the decoded axis)
            % and entropy by log2(n_bins_i), so task/controls stay comparable
            % despite their different bin counts.
            trial_pos_error     = nan(1, n_tr);
            trial_pos_entropy   = nan(1, n_tr); 
            trial_bin_errors    = nan(n_tr, n_bins_i);
            trial_bin_entropies = nan(n_tr, n_bins_i); 
            
            for t_test = 1:n_tr
                tr_train = setdiff(1:n_tr, t_test);
                lambda_x = mean(cond_data(:, :, tr_train), 3, 'omitnan') + 1e-6; 
                lambda_x(isnan(lambda_x)) = 1e-6; 
                
                if is_shuffle
                    for cell_idx = 1:n_k_cells
                        lambda_x(cell_idx, :) = lambda_x(cell_idx, randperm(n_bins_i));
                    end
                end
                
                test_trial_data = cond_data(:, :, t_test); 
                pred_bins = nan(1, n_bins_i);
                bin_entropies = nan(1, n_bins_i); 
                
                for b = 1:n_bins_i
                    r = test_trial_data(:, b); 
                    r(isnan(r)) = 0; 
                    
                    LL = r' * log(lambda_x) - sum(lambda_x, 1);
                    LL_shifted = LL - max(LL); 
                    posterior = exp(LL_shifted) / sum(exp(LL_shifted));
                    
                    bin_entropies(b) = -sum(posterior .* log2(posterior + eps));
                    [~, pred_bins(b)] = max(LL);
                end
                
                sq_errors = (pred_bins - (1:n_bins_i)).^2;
                trial_pos_error(t_test)        = sqrt(mean(sq_errors, 'omitnan')) / n_bins_i; 
                trial_bin_errors(t_test, :)    = sqrt(sq_errors) / n_bins_i; 
                trial_pos_entropy(t_test)      = mean(bin_entropies, 'omitnan') / log2(n_bins_i); 
                trial_bin_entropies(t_test, :) = bin_entropies / log2(n_bins_i); 
            end
            
            bin_pos_err(i, 1:n_bins_i, g, c)     = mean(trial_bin_errors, 1, 'omitnan');
            bin_pos_entropy(i, 1:n_bins_i, g, c) = mean(trial_bin_entropies, 1, 'omitnan'); 
            
            % --- B. Lick Pattern Decoding (Bin-by-Bin Log-Link Ridge) ---
            trial_lick_corr = nan(1, n_tr);
            if lick_bins_ok
                Y_lick_local = Y_lick;
                if is_shuffle, Y_lick_local = Y_lick_local(randperm(n_tr), :); end
                
                Y_lick_log = log(Y_lick_local + 1); 
                Y_pred_full = nan(n_tr, n_bins_i);
                
                for b = 1:n_bins_i
                    X_b = squeeze(cond_data(:, b, 1:n_tr))'; % [Trials x Cells]
                    Y_b = Y_lick_log(:, b);                  % [Trials x 1]

                    % Closed-form PRESS LOO ridge with intercept and per-fit
                    % standardisation. Replaces the per-trial setdiff loop.
                    % (2026-05-07; ~10x speedup, behaviour-equivalent within
                    % O(1/n) standardisation drift.)
                    Y_pred_log = loo_ridge_press(X_b, Y_b, lambda);
                    Y_pred_full(:, b) = exp(Y_pred_log) - 1;
                end
                
                for t_test = 1:n_tr
                    actual_licks = Y_lick_local(t_test, :);
                    pred_licks = Y_pred_full(t_test, :);
                    pred_licks(pred_licks < 0) = 0;
                    
                    valid_b = ~isnan(pred_licks) & ~isnan(actual_licks);
                    if sum(valid_b) > 3 && var(pred_licks(valid_b)) > 1e-6 && var(actual_licks(valid_b)) > 1e-6
                        trial_lick_corr(t_test) = corr(pred_licks(valid_b)', actual_licks(valid_b)');
                    end
                end
            end
            
            % --- C. Extract Epochs ---
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

%% 8. Plotting Decoder Engines (Linked Axes)

x_bases_dec = {1:10, 12:21, 23:32};
x_ticks_centers_dec = [5.5, 16.5, 27.5];
% cond_colors: All, No-DMS, No-DLS, No-ACC, No-V1, No-CA1, Shuffle
cond_colors = [0      0      0;        % All
               0.8500 0.3250 0.0980;   % No-DMS (orange)
               0.9290 0.6940 0.1250;   % No-DLS (yellow)
               0.4940 0.1840 0.5560;   % No-ACC (purple)
               0      0.6    0.6;      % No-V1  (teal)
               0.8    0.1    0.2;      % No-CA1 (crimson)
               0.5    0.5    0.5];     % Shuffle

% =========================================================================
% FIGURE A: ML Decoding Error Evolution (3 Groups)
% =========================================================================
figure('Name', 'ML Decoding Evolution (Yoked)', 'Position', [100, 100, 1500, 500], 'Color', 'w');
ax_dec_err = gobjects(1, 3);
for g = 1:3
    ax_dec_err(g) = subplot(1, 3, g); hold on;
    h_lines = gobjects(1, n_conds);
    for c = 1:n_conds
        first_epoch = false;
        for e = 1:n_epochs
            data_matrix = squeeze(ep_pos_err(:, e, :, g, c));
            mu = mean(data_matrix, 1, 'omitnan');
            se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
            if all(isnan(mu)), continue; end
            h = shadedErrorBar(x_bases_dec{e}, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
            if ~first_epoch && isfield(h, 'mainLine'), h_lines(c) = h.mainLine; first_epoch = true; end
        end
    end
    xline([11, 22], 'k:'); xticks(x_ticks_centers_dec); xticklabels(epochs);
    title(sprintf('%s - ML Decoder Error', group_names{g})); 
    ylabel('Decoding error (fraction of axis)'); set(gca, 'YDir', 'reverse');
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_dec_err, 'y');
save_to_svg('Decoding_Evolution_3Groups_Yoked');

% =========================================================================
% FIGURE B: Spatial Decoding Bin Error (Error vs Bins, per-group axis)
% =========================================================================
figure('Name', 'Spatial Decoding Error Profile across Corridor', 'Position', [150, 150, 1500, 500], 'Color', 'w');
ax_bin = gobjects(1, 3);
for g = 1:3
    ax_bin(g) = subplot(1, 3, g); hold on;
    n_bins_g = group_n_bins(g);   % this group's bin count (see section 7)
    if isnan(n_bins_g)
        title(sprintf('%s - no data', group_names{g})); continue;
    end
    h_lines = gobjects(1, n_conds);
    
    for c = 1:n_conds
        data_matrix = squeeze(bin_pos_err(:, 1:n_bins_g, g, c)); % [Animals x Bins]
        mu = mean(data_matrix, 1, 'omitnan');
        se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
        
        if all(isnan(mu)), continue; end
        
        h = shadedErrorBar(1:n_bins_g, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
        if isfield(h, 'mainLine')
            h_lines(c) = h.mainLine;
        end
    end
    
    if g < 3
        % Landmarks scaled to this group's bin count.
        xline((cfg.visual_zone_cm/cfg.bin_size_au) * n_bins_g/n_bins, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
        xline((cfg.reward_zone_cm/cfg.bin_size_au) * n_bins_g/n_bins, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
        xlabel('Spatial Bin');
    else
        xlabel('Temporal Bin');
    end
    
    ylabel('Decoding error (fraction of axis)');
    title(sprintf('%s - Error Profile', group_names{g}));
    set(gca, 'YDir', 'reverse');
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_bin, 'y');
save_to_svg('Decoding_Spatial_Bin_Error_3Groups');

% =========================================================================
% FIGURE C: Spatial Certainty Bin Profile (Normalized Entropy vs Bins)
% =========================================================================
figure('Name', 'Spatial Certainty Profile across Corridor', 'Position', [150, 100, 1500, 500], 'Color', 'w');
ax_ent_bin = gobjects(1, 3);
for g = 1:3
    ax_ent_bin(g) = subplot(1, 3, g); hold on;
    n_bins_g = group_n_bins(g);
    if isnan(n_bins_g)
        title(sprintf('%s - no data', group_names{g})); continue;
    end
    h_lines = gobjects(1, n_conds);
    
    for c = 1:n_conds
        data_matrix = squeeze(bin_pos_entropy(:, 1:n_bins_g, g, c)); % [Animals x Bins]
        mu = mean(data_matrix, 1, 'omitnan');
        se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
        
        if all(isnan(mu)), continue; end
        
        h = shadedErrorBar(1:n_bins_g, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
        if isfield(h, 'mainLine')
            h_lines(c) = h.mainLine;
        end
    end
    
    if g < 3
        xline((cfg.visual_zone_cm/cfg.bin_size_au) * n_bins_g/n_bins, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
        xline((cfg.reward_zone_cm/cfg.bin_size_au) * n_bins_g/n_bins, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
        xlabel('Spatial Bin');
    else
        xlabel('Temporal Bin');
    end
    
    ylabel('Normalized entropy (H / log_2 N)');
    title(sprintf('%s - Certainty Profile', group_names{g}));
    set(gca, 'YDir', 'reverse'); % Downward means lower entropy / higher certainty
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_ent_bin, 'y');
save_to_svg('Decoding_Spatial_Entropy_Bin_Profile_3Groups');

% =========================================================================
% FIGURE D: Lick Pattern Prediction Correlation Evolution (3 Groups)
% =========================================================================
figure('Name', 'Lick Spatial/Temporal Pattern Prediction', 'Position', [100, 150, 1500, 500], 'Color', 'w');
ax_dec_lick = gobjects(1, 3);
for g = 1:3
    ax_dec_lick(g) = subplot(1, 3, g); hold on;
    h_lines = gobjects(1, n_conds);
    for c = 1:n_conds
        first_epoch = false;
        for e = 1:n_epochs
            data_matrix = squeeze(ep_lick_corr(:, e, :, g, c));
            mu = mean(data_matrix, 1, 'omitnan');
            se = std(data_matrix, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_matrix), 1));
            if all(isnan(mu)), continue; end
            h = shadedErrorBar(x_bases_dec{e}, mu, se, 'lineProps', {'Color', cond_colors(c,:), 'LineWidth', 2});
            if ~first_epoch && isfield(h, 'mainLine'), h_lines(c) = h.mainLine; first_epoch = true; end
        end
    end
    xline([11, 22], 'k:'); xticks(x_ticks_centers_dec); xticklabels(epochs);
    title(sprintf('%s - Lick Prediction', group_names{g})); 
    ylabel('Pearson r (Predicted vs Actual)');
    if g == 1 && any(isgraphics(h_lines)), legend(h_lines(isgraphics(h_lines)), cond_names, 'Location', 'best'); end
end
linkaxes(ax_dec_lick, 'y');
save_to_svg('Decoding_Lick_Prediction_3Groups');

fprintf('--- Full Integrated Pipeline Execution Complete ---\n');

%% 9. Comprehensive Spatiotemporal Activity by AREA
fprintf('--- Generating Spatiotemporal Plots by AREA (All Groups) ---\n');

areas = {'DMS', 'DLS', 'ACC', 'V1', 'CA1'};
n_areas = length(areas);
epochs = {'Naive', 'Intermediate', 'Expert'};
n_epochs = length(epochs);

% Alignment parameters for temporal evolution
pre_lp = 20; post_lp = 40;
aligned_x = -pre_lp:post_lp;
n_aligned = length(aligned_x);

% Initialize nested data structures
% pop_data(g).area(a) will hold matrices for pooled/hierarchical processing
pop_data = struct();
for g = 1:3
    for a = 1:n_areas
        pop_data(g).area(a).epoch_raw = cell(1, n_epochs);
        pop_data(g).area(a).epoch_z   = cell(1, n_epochs);
        pop_data(g).area(a).align_raw = [];
        pop_data(g).area(a).align_z   = [];
        pop_data(g).area(a).mouse_id  = [];
    end
end

% Extract Data
for g = 1:3
    curr_data = groups{g};
    for i = 1:numel(curr_data)
        if g == 1
            lp = learning_points_task(i);
            if isnan(lp), continue; end
        else
            lp = avg_lp;
        end
        
        if g < 3
            act = curr_data(i).spatial_binned_fr_all;
        else
            act = curr_data(i).firing_rates_per_bin;
        end
        act(isnan(act)) = 0;
        % Drop DG units — excluded from all figures/analyses (2026-05-24).
        keep_cells = ~is_area_safe(curr_data(i), 'DG');
        act = act(keep_cells, :, :);
        n_cells = size(act, 1);
        n_tr = size(act, 3);
        
        % Z-score calculation
        act_z = nan(size(act));
        for c = 1:n_cells
            mu = mean(act(c,:,:), 'all', 'omitnan');
            sig = std(act(c,:,:), 0, 'all', 'omitnan');
            if sig > 0, act_z(c,:,:) = (act(c,:,:) - mu) / sig; else, act_z(c,:,:) = 0; end
        end
        
        % Average across spatial/temporal bins for trial-by-trial temporal evolution
        act_time_raw = squeeze(mean(act, 2, 'omitnan')); % [Cells x Trials]
        act_time_z   = squeeze(mean(act_z, 2, 'omitnan'));
        
        % Epoch definitions
        epoch_idx = cell(1, n_epochs);
        if n_tr >= trials_per_epoch, epoch_idx{1} = 1:trials_per_epoch; end
        if ~isnan(lp) && lp > trials_per_epoch && lp <= n_tr, epoch_idx{2} = (lp - trials_per_epoch) : (lp - 1); end
        if ~isnan(lp) && (lp + trials_per_epoch - 1) <= n_tr, epoch_idx{3} = lp : (lp + trials_per_epoch - 1); end
        
        % Masks
        v1_mask  = is_area_safe(curr_data(i), 'V1');
        ca1_mask = is_area_safe(curr_data(i), 'CA1');
        masks = {curr_data(i).is_dms(keep_cells), ...
                 curr_data(i).is_dls(keep_cells), ...
                 curr_data(i).is_acc(keep_cells), ...
                 v1_mask(keep_cells), ...
                 ca1_mask(keep_cells)};

        % Populate Structure
        for a = 1:n_areas
            mask = masks{a};
            if sum(mask) == 0, continue; end
            
            % 1. Spatial Epochs
            for e = 1:n_epochs
                idx = epoch_idx{e};
                if isempty(idx), continue; end
                take_n = min(length(idx), trials_per_epoch);
                ep_raw = mean(act(mask, :, idx(1:take_n)), 3, 'omitnan');
                ep_z   = mean(act_z(mask, :, idx(1:take_n)), 3, 'omitnan');
                
                pop_data(g).area(a).epoch_raw{e} = [pop_data(g).area(a).epoch_raw{e}; ep_raw];
                pop_data(g).area(a).epoch_z{e}   = [pop_data(g).area(a).epoch_z{e}; ep_z];
            end
            
            % 2. Aligned Temporal Evolution
            s_idx = lp - pre_lp; e_idx = lp + post_lp;
            z_start = max(1, s_idx); z_end = min(n_tr, e_idx);
            
            align_raw_tmp = nan(sum(mask), n_aligned);
            align_z_tmp   = nan(sum(mask), n_aligned);
            
            if z_start <= z_end
                a_start = 1 + (z_start - s_idx);
                a_end   = n_aligned - (e_idx - z_end);
                align_raw_tmp(:, a_start:a_end) = act_time_raw(mask, z_start:z_end);
                align_z_tmp(:, a_start:a_end)   = act_time_z(mask, z_start:z_end);
            end
            
            pop_data(g).area(a).align_raw = [pop_data(g).area(a).align_raw; align_raw_tmp];
            pop_data(g).area(a).align_z   = [pop_data(g).area(a).align_z; align_z_tmp];
            pop_data(g).area(a).mouse_id  = [pop_data(g).area(a).mouse_id; repmat(i, sum(mask), 1)];
        end
    end
end

% --- PLOTTING LOOPS ---
metrics = {'raw', 'Raw FR'; 'z', 'Z-Scored'};
avg_methods = {'Pooled', 'Hierarchical'};
epoch_colors = lines(n_epochs);
area_colors = [0      0.4470 0.7410;   % DMS
               0.4660 0.6740 0.1880;   % DLS
               0.8500 0.3250 0.0980;   % ACC
               0.4940 0.1840 0.5560;   % V1
               0.8000 0.1000 0.2000];  % CA1

for met_idx = 1:size(metrics, 1)
    met_field = metrics{met_idx, 1};
    met_name  = metrics{met_idx, 2};
    
    for avg_idx = 1:length(avg_methods)
        avg_mode = avg_methods{avg_idx};
        fig_prefix = sprintf('Area_Activity_%s_%s', met_field, avg_mode);
        
        % -----------------------------------------------------------------
        % FIGURE A: SPATIAL/TEMPORAL TUNING BY AREA AND EPOCH
        % -----------------------------------------------------------------
        figure('Name', sprintf('%s: Spatial', fig_prefix), 'Position', [50, 50, 1500, 900], 'Color', 'w');
        t_spatial = tiledlayout(3, n_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
        ax_spatial = gobjects(3, n_areas);

        for g = 1:3
            n_bins_g = group_n_bins(g);
            for a = 1:n_areas
                ax_spatial(g, a) = nexttile((g-1)*n_areas + a); hold on;
                h_lines = gobjects(1, n_epochs);
                mouse_ids = pop_data(g).area(a).mouse_id;
                unique_mice = unique(mouse_ids);
                
                for e = 1:n_epochs
                    data_mat = pop_data(g).area(a).(['epoch_' met_field]){e};
                    if isempty(data_mat), continue; end
                    
                    if strcmp(avg_mode, 'Pooled')
                        mu = mean(data_mat, 1, 'omitnan');
                        se = std(data_mat, 0, 1, 'omitnan') / sqrt(size(data_mat, 1));
                        n_label = size(data_mat, 1);
                    else
                        n_m = length(unique_mice);
                        mouse_means = nan(n_m, size(data_mat, 2));
                        for m = 1:n_m
                            m_idx = (mouse_ids == unique_mice(m));
                            mouse_means(m, :) = mean(data_mat(m_idx, :), 1, 'omitnan');
                        end
                        mu = mean(mouse_means, 1, 'omitnan');
                        se = std(mouse_means, 0, 1, 'omitnan') / sqrt(n_m);
                        n_label = n_m;
                    end
                    
                    if all(isnan(mu)), continue; end
                    h = shadedErrorBar(1:n_bins_g, mu, se, 'lineProps', {'Color', epoch_colors(e,:), 'LineWidth', 2});
                    if isfield(h, 'mainLine'), h_lines(e) = h.mainLine; end
                end
                
                if g < 3
                    xline(cfg.visual_zone_cm/cfg.bin_size_au * n_bins_g/n_bins, 'k--', 'Visual', 'LabelVerticalAlignment', 'bottom'); 
                    xline(cfg.reward_zone_cm/cfg.bin_size_au * n_bins_g/n_bins, 'k-', 'Reward', 'LabelVerticalAlignment', 'bottom');
                end
                
                if strcmp(avg_mode, 'Pooled')
                    title(sprintf('%s - %s (n=%d)', group_names{g}, areas{a}, n_label));
                else
                    title(sprintf('%s - %s (N=%d mice)', group_names{g}, areas{a}, n_label));
                end
                
                if g == 3 && a == 1
                    xlabel('Bin'); ylabel(met_name);
                end
                if g == 1 && a == 3 && any(isgraphics(h_lines))
                    legend(h_lines(isgraphics(h_lines)), epochs, 'Location', 'best');
                end
            end
        end
        linkaxes(ax_spatial, 'y'); % Link Y across all tiles
        title(t_spatial, sprintf('Spatial/Temporal Profile by Area (%s - %s)', met_name, avg_mode), 'FontSize', 16);
        save_to_svg(fig_prefix);
        
        % -----------------------------------------------------------------
        % FIGURE B: TEMPORAL EVOLUTION (TRIAL-BY-TRIAL) BY AREA
        % -----------------------------------------------------------------
        figure('Name', sprintf('%s: Temporal Evolution', fig_prefix), 'Position', [100, 150, 1500, 450], 'Color', 'w');
        ax_temp = gobjects(1, 3);
        
        for g = 1:3
            ax_temp(g) = subplot(1, 3, g); hold on;
            h_lines = gobjects(1, n_areas);
            
            for a = 1:n_areas
                data_mat = pop_data(g).area(a).(['align_' met_field]);
                if isempty(data_mat), continue; end
                mouse_ids = pop_data(g).area(a).mouse_id;
                unique_mice = unique(mouse_ids);
                
                if strcmp(avg_mode, 'Pooled')
                    mu = mean(data_mat, 1, 'omitnan');
                    se = std(data_mat, 0, 1, 'omitnan') / sqrt(size(data_mat, 1));
                else
                    n_m = length(unique_mice);
                    mouse_means = nan(n_m, size(data_mat, 2));
                    for m = 1:n_m
                        m_idx = (mouse_ids == unique_mice(m));
                        mouse_means(m, :) = mean(data_mat(m_idx, :), 1, 'omitnan');
                    end
                    mu = mean(mouse_means, 1, 'omitnan');
                    se = std(mouse_means, 0, 1, 'omitnan') / sqrt(n_m);
                end
                
                if all(isnan(mu)), continue; end
                h = shadedErrorBar(aligned_x, mu, se, 'lineProps', {'Color', area_colors(a,:), 'LineWidth', 2});
                if isfield(h, 'mainLine'), h_lines(a) = h.mainLine; end
            end
            
            xline(0, 'k-', 'LP', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            title(sprintf('%s', group_names{g}));
            xlabel('Trials relative to LP');
            ylabel(sprintf('Mean %s (Averaged over Space)', met_name));
            if g == 1 && any(isgraphics(h_lines))
                legend(h_lines(isgraphics(h_lines)), areas, 'Location', 'best');
            end
        end
        linkaxes(ax_temp, 'y');
        sgtitle(sprintf('Trial-by-Trial Evolution by Area (%s - %s)', met_name, avg_mode));
        save_to_svg(sprintf('%s_TrialEvo', fig_prefix));
        
    end
end
fprintf('--- Area Activity Plotting Complete ---\n\n');

%% 11. Pooled Trial-Wise Correlation Pipeline (Behavior on X-Axis)
% Aggregates all trials from Naive (Epoch 1) and Expert (Epoch 3) across all mice.
% Plots are separated by experimental group.
fprintf('--- Running Pooled Trial-Wise Correlations (Behavior on X-Axis) ---\n');

% Configuration
corr_epochs = [1, 3]; 
epoch_colors = [0, 0.4470, 0.7410;  % Naive: Blue
                0.8500, 0.3250, 0.0980]; % Expert: Red/Orange
epoch_labels = {'Naive', 'Expert'};
area_names = {'DMS', 'DLS', 'ACC', 'V1', 'CA1', 'All Units'};
n_area_cols = length(area_names);  % DMS, DLS, ACC, V1, CA1, All
all_col = n_area_cols;              % index of the "All Units" column

for g = 1:3
    % 11.1. Data Aggregation for Group G
    % Initialize containers for trial-wise pooling
    p_lick_stab = [];
    p_vel_stab  = [];
    p_neur_stab = nan(0, n_area_cols); % columns = DMS, DLS, ACC, V1, CA1, All
    p_dec_space = [];
    p_dec_lick  = [];
    p_epoch_idx = []; 

    for e_idx = 1:length(corr_epochs)
        e = corr_epochs(e_idx);
        for i = 1:max_animals
            % Find trials where we have both behavior and neural data
            % Using 'All Units' (a=4) stability as a proxy for neural data presence
            valid_tr = find(~isnan(squeeze(ep_stab_licks(i, e, :, g))) & ...
                            ~isnan(squeeze(hier_z(i, e, :, g, all_col))));
            if isempty(valid_tr), continue; end
            
            % Behavior (X-axis candidates)
            p_lick_stab = [p_lick_stab; squeeze(ep_stab_licks(i, e, valid_tr, g))];
            if g < 3
                p_vel_stab = [p_vel_stab; squeeze(ep_stab_vel(i, e, valid_tr, g))];
            else
                p_vel_stab = [p_vel_stab; nan(length(valid_tr), 1)];
            end
            
            % Neural Stability (Y-axis candidates)
            area_tmp = nan(length(valid_tr), n_area_cols);
            for a = 1:n_area_cols
                area_tmp(:, a) = squeeze(hier_z(i, e, valid_tr, g, a));
            end
            p_neur_stab = [p_neur_stab; area_tmp];
            
            % Decoding (Y or X candidates)
            % Using Cond 1 (All units) for decoding metrics
            p_dec_space = [p_dec_space; squeeze(ep_pos_err(i, e, valid_tr, g, 1))];
            p_dec_lick  = [p_dec_lick; squeeze(ep_lick_corr(i, e, valid_tr, g, 1))];
            
            % Epoch tracking
            p_epoch_idx = [p_epoch_idx; repmat(e_idx, length(valid_tr), 1)];
        end
    end

    % 11.2. Plotting per Group
    % Layout expanded 2026-05-07 from 3x4 to 3xn_area_cols so V1 fits.
    fig_name = sprintf('Group_%d_Cross_Modal_Correlations', g);
    figure('Name', sprintf('%s: Pooled Trial Correlations', group_names{g}), ...
           'Position', [50, 50, 300 * n_area_cols, 900], 'Color', 'w');
    t = tiledlayout(3, n_area_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Pooled Analysis: %s (Trials from all mice)', group_names{g}), 'FontSize', 16);

    n_per_area = n_area_cols - 1; % brain areas without "all"

    % --- ROW 1: Neural Stability (Y) vs Lick Stability (X), one panel per column ---
    for a = 1:n_area_cols
        nexttile(a); hold on;
        local_plot_scatter(p_lick_stab, p_neur_stab(:, a), p_epoch_idx, ...
            'Lick Stability (r)', sprintf('Neural Stab: %s (Z)', area_names{a}), epoch_colors);
    end

    % --- ROW 2: Decoding Performance (Y) vs Lick Stability (X) ---
    base = n_area_cols;
    % Column 1: Lick Decoding
    nexttile(base + 1); hold on;
    local_plot_scatter(p_lick_stab, p_dec_lick, p_epoch_idx, ...
        'Lick Stability (r)', 'Lick Decoding (r)', epoch_colors);

    % Column 2: Spatial Decoding (RMSE)
    nexttile(base + 2); hold on;
    local_plot_scatter(p_lick_stab, p_dec_space, p_epoch_idx, ...
        'Lick Stability (r)', 'Space Decoding (RMSE)', epoch_colors);
    set(gca, 'YDir', 'reverse'); % Downward = lower error (better)

    % Column 3: Velocity Stability vs All-Units Neural Stability
    nexttile(base + 3); hold on;
    if g < 3
        local_plot_scatter(p_vel_stab, p_neur_stab(:, all_col), p_epoch_idx, ...
            'Vel Stability (r)', 'Neural Stab: All (Z)', epoch_colors);
    else
        axis off; text(0.1, 0.5, 'No Wheel Data (Ctrl 2)');
    end

    % Column 4: Lick Decoding (Y) vs Velocity Stability (X)
    nexttile(base + 4); hold on;
    if g < 3
        local_plot_scatter(p_vel_stab, p_dec_lick, p_epoch_idx, ...
            'Vel Stability (r)', 'Lick Decoding (r)', epoch_colors);
    else
        axis off;
    end
    % Any remaining tiles in row 2 stay empty (when n_area_cols > 4)
    for extra = 5:n_area_cols
        nexttile(base + extra); axis off;
    end

    % --- ROW 3: Neural Stability per area (Y) vs Space Decoding (X) ---
    base = 2 * n_area_cols;
    for a = 1:n_per_area
        nexttile(base + a); hold on;
        local_plot_scatter(p_dec_space, p_neur_stab(:, a), p_epoch_idx, ...
            'Space Decoding (RMSE)', sprintf('Neural Stab: %s (Z)', area_names{a}), epoch_colors);
        set(gca, 'XDir', 'reverse'); % Rightward = lower error (better)
    end

    % Final tile of row 3: All-Units Neural vs Lick Decoding
    nexttile(base + n_area_cols); hold on;
    local_plot_scatter(p_dec_lick, p_neur_stab(:, all_col), p_epoch_idx, ...
        'Lick Decoding (r)', 'Neural Stab: All (Z)', epoch_colors);

    save_to_svg(fig_name);
end

%% 12. Summary Metrics for Presentation (Unit Counts & Firing Rates)
fprintf('\n=====================================================================\n');
fprintf('                 DATA SUMMARY FOR PRESENTATION                       \n');
fprintf('=====================================================================\n');

for g = 1:3
    curr_data = groups{g};
    if isempty(curr_data), continue; end
    
    n_animals = numel(curr_data);
    
    % Initialize storage arrays for unit counts
    total_units   = zeros(n_animals, 1);
    units_dms     = zeros(n_animals, 1);
    units_dls     = zeros(n_animals, 1);
    units_acc     = zeros(n_animals, 1);
    units_v1      = zeros(n_animals, 1);
    units_ca1     = zeros(n_animals, 1);

    units_msn     = zeros(n_animals, 1);
    units_fsn     = zeros(n_animals, 1);
    units_tan     = zeros(n_animals, 1);
    units_unclass = zeros(n_animals, 1);

    % Initialize arrays for behavior and firing rates
    total_trials = zeros(n_animals, 1);
    fr_msn = []; fr_fsn = []; fr_tan = [];
    fr_dms = []; fr_dls = []; fr_acc = []; fr_v1 = []; fr_ca1 = [];

    for i = 1:n_animals
        % --- Area counts ---
        % Drop DG units — excluded from all figures/analyses (2026-05-24).
        keep_cells = ~is_area_safe(curr_data(i), 'DG');
        is_dms = curr_data(i).is_dms(keep_cells);
        is_dls = curr_data(i).is_dls(keep_cells);
        is_acc = curr_data(i).is_acc(keep_cells);
        v1_mask  = is_area_safe(curr_data(i), 'V1');
        ca1_mask = is_area_safe(curr_data(i), 'CA1');
        is_v1  = v1_mask(keep_cells);
        is_ca1 = ca1_mask(keep_cells);

        total_units(i) = length(is_dms);
        units_dms(i)   = sum(is_dms);
        units_dls(i)   = sum(is_dls);
        units_acc(i)   = sum(is_acc);
        units_v1(i)    = sum(is_v1);
        units_ca1(i)   = sum(is_ca1);
        
        % --- Neuron type counts ---
        if isfield(curr_data(i), 'final_neurontypes') && ~isempty(curr_data(i).final_neurontypes)
            ntypes_raw = curr_data(i).final_neurontypes;
            [~, cols] = size(ntypes_raw);
            
            if cols >= 5
                ntypes = ntypes_raw(:, 5);
            elseif cols == 1
                ntypes = ntypes_raw(:, 1);
            else
                ntypes = nan(size(ntypes_raw, 1), 1);
            end
            ntypes = ntypes(keep_cells);  % DG excluded

            units_msn(i) = sum(ntypes == 1);
            units_fsn(i) = sum(ntypes == 2);
            units_tan(i) = sum(ntypes == 3);
            units_unclass(i) = sum(isnan(ntypes) | (ntypes ~= 1 & ntypes ~= 2 & ntypes ~= 3));
        else
            units_unclass(i) = total_units(i);
            ntypes = nan(total_units(i), 1); % Dummy array for the FR calculation below
        end
        
        % --- Trial counts & Firing Rates ---
        if g < 3 && isfield(curr_data(i), 'spatial_binned_fr_all')
            fr_tensor = curr_data(i).spatial_binned_fr_all;
        elseif g == 3 && isfield(curr_data(i), 'firing_rates_per_bin')
            fr_tensor = curr_data(i).firing_rates_per_bin;
        else
            fr_tensor = [];
        end
        
        if ~isempty(fr_tensor)
            total_trials(i) = size(fr_tensor, 3);
            
            % Average FR across bins and trials for each neuron
            mean_fr_per_neuron = mean(fr_tensor, [2, 3], 'omitnan');
            mean_fr_per_neuron = mean_fr_per_neuron(keep_cells);  % DG excluded

            % Area Firing Rates
            if sum(is_dms) > 0, fr_dms = [fr_dms; mean_fr_per_neuron(is_dms)]; end
            if sum(is_dls) > 0, fr_dls = [fr_dls; mean_fr_per_neuron(is_dls)]; end
            if sum(is_acc) > 0, fr_acc = [fr_acc; mean_fr_per_neuron(is_acc)]; end
            if sum(is_v1)  > 0, fr_v1  = [fr_v1;  mean_fr_per_neuron(is_v1)];  end
            if sum(is_ca1) > 0, fr_ca1 = [fr_ca1; mean_fr_per_neuron(is_ca1)]; end

            % Cell Type Firing Rates
            if sum(ntypes == 1) > 0, fr_msn = [fr_msn; mean_fr_per_neuron(ntypes == 1)]; end
            if sum(ntypes == 2) > 0, fr_fsn = [fr_fsn; mean_fr_per_neuron(ntypes == 2)]; end
            if sum(ntypes == 3) > 0, fr_tan = [fr_tan; mean_fr_per_neuron(ntypes == 3)]; end
        end
    end
    
    % Anonymous functions for quick formatting (Mean +/- SEM)
    fmt_stats = @(x) sprintf('%7.1f +/- %5.1f', mean(x), std(x)/sqrt(n_animals));
    fmt_fr    = @(x) sprintf('%6.2f +/- %5.2f', mean(x), std(x)/sqrt(max(1,length(x))));
    
    fprintf('\n>>> %s <<<\n', upper(group_names{g}));
    fprintf('Total Animals: %d\n\n', n_animals);
    
    fprintf('--- Units per Animal (Mean +/- SEM) [Total across all animals] ---\n');
    fprintf('Total Units : %s  [Sum: %d]\n', fmt_stats(total_units), sum(total_units));
    fprintf('DMS Units   : %s  [Sum: %d]\n', fmt_stats(units_dms), sum(units_dms));
    fprintf('DLS Units   : %s  [Sum: %d]\n', fmt_stats(units_dls), sum(units_dls));
    fprintf('ACC Units   : %s  [Sum: %d]\n', fmt_stats(units_acc), sum(units_acc));
    n_with_v1  = sum(units_v1  > 0);
    n_with_ca1 = sum(units_ca1 > 0);
    fprintf('V1  Units   : %s  [Sum: %d]   (%d/%d animals have V1 probe)\n', ...
        fmt_stats(units_v1),  sum(units_v1),  n_with_v1,  n_animals);
    fprintf('CA1 Units   : %s  [Sum: %d]   (%d/%d animals have CA1 in probe)\n', ...
        fmt_stats(units_ca1), sum(units_ca1), n_with_ca1, n_animals);

    fprintf('\n--- Putative Cell Types per Animal (Mean +/- SEM) [Total] ---\n');
    fprintf('(MSN/FSN/TAN classification applies to striatum only — V1/CA1 not classified)\n');
    fprintf('MSN (Type 1): %s  [Sum: %d]\n', fmt_stats(units_msn), sum(units_msn));
    fprintf('FSN (Type 2): %s  [Sum: %d]\n', fmt_stats(units_fsn), sum(units_fsn));
    fprintf('TAN (Type 3): %s  [Sum: %d]\n', fmt_stats(units_tan), sum(units_tan));
    fprintf('Unclassified: %s  [Sum: %d]\n', fmt_stats(units_unclass), sum(units_unclass));
    
    fprintf('\n--- Behavior & Global Firing Rates ---\n');
    fprintf('Total Trials per Session : %s\n\n', fmt_stats(total_trials));
    
    fprintf('--- Firing Rates by Area (Hz, Mean +/- SEM across neurons) ---\n');
    fprintf('DMS Units : %s\n', fmt_fr(fr_dms));
    fprintf('DLS Units : %s\n', fmt_fr(fr_dls));
    fprintf('ACC Units : %s\n', fmt_fr(fr_acc));
    if ~isempty(fr_v1)
        fprintf('V1  Units : %s  (n=%d units)\n', fmt_fr(fr_v1),  numel(fr_v1));
    end
    if ~isempty(fr_ca1)
        fprintf('CA1 Units : %s  (n=%d units)\n', fmt_fr(fr_ca1), numel(fr_ca1));
    end
    fprintf('\n');
    
    fprintf('--- Firing Rates by Putative Cell Type (Hz, Mean +/- SEM) ---\n');
    fprintf('MSN (Type 1): %s\n', fmt_fr(fr_msn));
    fprintf('FSN (Type 2): %s\n', fmt_fr(fr_fsn));
    fprintf('TAN (Type 3): %s\n', fmt_fr(fr_tan));
    fprintf('---------------------------------------------------------------------\n');
end
fprintf('\n======================= END OF SUMMARY =======================\n\n');

%% Local Helper for Clean Scatter Plotting
function local_plot_scatter(x, y, e_idx, xlab, ylab, colors)
    valid = ~isnan(x) & ~isnan(y) & ~isinf(x) & ~isinf(y);
    if sum(valid) < 5, return; end
    
    x = x(valid); y = y(valid); e_idx = e_idx(valid);
    
    % Plot epoch-specific points
    u_eps = unique(e_idx);
    for i_ep = 1:length(u_eps)
        curr_mask = (e_idx == u_eps(i_ep));
        scatter(x(curr_mask), y(curr_mask), 15, colors(u_eps(i_ep), :), 'filled', ...
            'MarkerFaceAlpha', 0.25);
    end
    
    % Global Regression
    p = polyfit(x, y, 1);
    xf = linspace(min(x), max(x), 100);
    plot(xf, polyval(p, xf), 'k-', 'LineWidth', 1.5);
    
    [r, pval] = corr(x, y, 'Type', 'Pearson');
    
    % Aesthetics
    grid on; box on;
    xlabel(xlab); ylabel(ylab);
    title(sprintf('r=%.2f, p=%.1e', r, pval), 'FontSize', 9);
    
    % Highlight significant subplots
    if pval < 0.05, set(gca, 'LineWidth', 1.5, 'XColor', 'k', 'YColor', 'k'); end
end



%% Local Helper Functions
function r = calc_triu_corr(mat)
    if size(mat, 2) < 2
        r = nan; return;
    end
    rho = corrcoef(mat);
    idx = triu(true(size(rho)), 1);
    r = mean(rho(idx), 'omitnan');
end

function save_to_svg(fig_name)
    try
        saveas(gcf, sprintf('%s.svg', fig_name));
        fprintf('Saved %s.svg\n', fig_name);
    catch
        fprintf('Warning: Could not save %s as SVG.\n', fig_name);
    end
end

% is_v1_safe(s) is now a standalone file in Striatum project/ — see
% is_v1_safe.m. Removed from this file 2026-05-07.