%% Build the Super-Mouse (Task + Control)

% --- Load Data ---
load("preprocessed_data.mat")
task_data_raw = preprocessed_data; % Keep raw task data separate initially
load("preprocessed_data_control.mat")
control_data_raw = preprocessed_data; % Keep raw control data separate

areas_to_include = {'DMS', 'DLS'};  % OMITTING ACC

% TASK
for i = 1:length(task_data_raw)
    keep_mask = ...
        (task_data_raw(i).is_dms) | ...
        (task_data_raw(i).is_dls);
    
    task_data_raw(i).spatial_binned_fr_all = task_data_raw(i).spatial_binned_fr_all(keep_mask, :, :);
    task_data_raw(i).is_dms = task_data_raw(i).is_dms(keep_mask);
    task_data_raw(i).is_dls = task_data_raw(i).is_dls(keep_mask);
    task_data_raw(i).is_acc = task_data_raw(i).is_acc(keep_mask);
end

% CONTROL
for i = 1:length(control_data_raw)
    keep_mask = ...
        (control_data_raw(i).is_dms) | ...
        (control_data_raw(i).is_dls);
    
    control_data_raw(i).spatial_binned_fr_all = control_data_raw(i).spatial_binned_fr_all(keep_mask, :, :);
    control_data_raw(i).is_dms = control_data_raw(i).is_dms(keep_mask);
    control_data_raw(i).is_dls = control_data_raw(i).is_dls(keep_mask);
    control_data_raw(i).is_acc = control_data_raw(i).is_acc(keep_mask);
end

% --- Process Task Mice ---
fprintf('Processing Task Mice...\n');
% Calculate learning point for ALL task mice first
zscored_lick_errors_task = {task_data_raw(:).zscored_lick_errors};
learning_point_task_all = cellfun(@(x) find(movsum(x <= -2, [0,9]) >= 8, 1), zscored_lick_errors_task, 'UniformOutput', false);
has_learning_point = ~cellfun(@isempty, learning_point_task_all);

% Filter task data to include only those with a learning point
task_data = task_data_raw(has_learning_point);
learning_point_task = learning_point_task_all(has_learning_point); % Keep only corresponding learning points
n_animals_task = length(task_data);

fprintf('  Found %d task mice with a defined learning point.\n', n_animals_task);

% Extract neural data for valid task mice
X = {task_data(:).spatial_binned_fr_all}; % Cell array of [n_i x B x T_i]

% MOD: Calculate Average Learning Point from Task Mice (use only those WITH a learning point)
avg_learning_point = round(mean(cell2mat(learning_point_task)));
fprintf('  Average learning point from task mice: Trial %d\n', avg_learning_point);

% --- Process Control Mice ---
fprintf('Processing Control Mice...\n');
n_animals_control_raw = length(control_data_raw);
Y_raw = {control_data_raw(:).spatial_binned_fr_all}; % Extract neural data for all control mice

% MOD: Define trial indices for control mice based on AVERAGE task learning point
control_indices_early = 1:10;
control_indices_pre_avg = (avg_learning_point - 10) : (avg_learning_point - 1);
control_indices_post_avg = (avg_learning_point + 1) : (avg_learning_point + 10);

% MOD: Check indices validity (must be positive)
if any(control_indices_pre_avg <= 0)
    error('Calculated pre-average-learning-point indices (%d:%d) are not valid based on average learning point %d.', ...
          min(control_indices_pre_avg), max(control_indices_pre_avg), avg_learning_point);
    % Or, adapt: e.g., clamp to 1, but this changes the window size/meaning.
end
control_indices_all_needed = [control_indices_early, control_indices_pre_avg, control_indices_post_avg];
max_trial_needed_control = max(control_indices_all_needed);

% MOD: Filter control mice based on trial availability
control_data = []; % Initialize filtered control data structure
Y = {}; % Initialize filtered control neural data cell array
valid_control_indices_mask = false(1, n_animals_control_raw); % Keep track of which controls are used

for i_control = 1:n_animals_control_raw
    num_trials_control = size(Y_raw{i_control}, 3); % Get total trials for this control mouse
    if num_trials_control >= max_trial_needed_control
        valid_control_indices_mask(i_control) = true;
        control_data = [control_data; control_data_raw(i_control)]; % Append struct
        Y{end+1} = Y_raw{i_control}; % Append neural data cell
    else
        fprintf('  Excluding control mouse %d: Has %d trials, needs at least %d trials.\n', ...
                i_control, num_trials_control, max_trial_needed_control);
    end
end
n_animals_control = length(control_data); % Number of VALID control animals
fprintf('  Found %d control mice with sufficient trials.\n', n_animals_control);

% --- Combine Task and Control Unit Information ---
fprintf('Combining Task and Control data...\n');
n_animals_total = n_animals_task + n_animals_control;
% MOD: Combine neural data cell arrays
XY = [X, Y]; % Combined cell array [Task..., Control...]

% MOD: Calculate unit counts and indices for the COMBINED dataset
all_units_total = sum(cellfun(@(xy) size(xy, 1), XY));
mouse_units_cumsum_total = cumsum(cellfun(@(xy) size(xy, 1), XY));
mouse_units_starts_total = [0 mouse_units_cumsum_total(1:end-1)] + 1; % Start index for each mouse's units
mouse_units_ends_total = mouse_units_cumsum_total; % End index for each mouse's units

bins = size(XY{1}, 2); % Assuming all mice have the same number of bins

% --- Build the Combined Supermouse Tensor ---
% MOD: Initialize the combined tensor
supermouse_big_tensor = nan(all_units_total, bins, 30); % 30 trials: 10 early, 10 pre-LP/avg, 10 post-LP/avg

% MOD: Loop through TASK mice
fprintf('  Populating tensor with Task mice data...\n');
for ianimal = 1:n_animals_task
    start_idx = mouse_units_starts_total(ianimal);
    end_idx = mouse_units_ends_total(ianimal);
    n_units_mouse = end_idx - start_idx + 1;

    % Check if task mouse has enough trials for its OWN learning point windows
    task_lp = learning_point_task{ianimal};
    task_trials_needed_pre = task_lp - 1; % Needs trials up to LP-1
    task_trials_needed_post = task_lp + 10; % Needs trials up to LP+10
    task_total_trials = size(X{ianimal}, 3);

    if (task_lp - 10 > 0) && (task_trials_needed_post <= task_total_trials)
        % Populate using individual learning point
        supermouse_big_tensor(start_idx:end_idx, :, 1:10)  = X{ianimal}(:, :, 1:10);
        supermouse_big_tensor(start_idx:end_idx, :, 11:20) = X{ianimal}(:, :, task_lp-10 : task_lp-1);
        supermouse_big_tensor(start_idx:end_idx, :, 21:30) = X{ianimal}(:, :, task_lp+1 : task_lp+10);
    else
        fprintf('  Warning: Task mouse %d does not have sufficient trials around its learning point (%d). Leaving its units as NaN.\n', ianimal, task_lp);
        % Corresponding rows in supermouse_big_tensor will remain NaN and likely get filtered out later
    end
end

% MOD: Loop through VALID CONTROL mice
fprintf('  Populating tensor with Control mice data...\n');
for i_control_valid = 1:n_animals_control
    ianimal = n_animals_task + i_control_valid; % Overall index in the combined list
    start_idx = mouse_units_starts_total(ianimal);
    end_idx = mouse_units_ends_total(ianimal);
    n_units_mouse = end_idx - start_idx + 1;

    % Populate using defined indices based on AVERAGE task learning point
    % We already checked for trial availability when filtering controls
    supermouse_big_tensor(start_idx:end_idx, :, 1:10)  = Y{i_control_valid}(:, :, control_indices_early);
    supermouse_big_tensor(start_idx:end_idx, :, 11:20) = Y{i_control_valid}(:, :, control_indices_pre_avg);
    supermouse_big_tensor(start_idx:end_idx, :, 21:30) = Y{i_control_valid}(:, :, control_indices_post_avg);
end

% --- NaN Handling (on the combined tensor) ---
% MOD: Use the combined tensor 'supermouse_big_tensor'
fprintf('Checking for NaN values in the combined tensor...\n');
neuron_has_nan = squeeze(any(isnan(supermouse_big_tensor), [2, 3]));
valid_neuron_indices = find(~neuron_has_nan); % Indices relative to the combined tensor
invalid_neuron_indices = find(neuron_has_nan);

num_total_neurons = size(supermouse_big_tensor, 1);
num_invalid_neurons = numel(invalid_neuron_indices);
num_valid_neurons = numel(valid_neuron_indices);

fprintf('Found %d neurons with NaN values (%.1f%%) in the combined tensor.\n', ...
    num_invalid_neurons, (num_invalid_neurons/num_total_neurons)*100);
fprintf('%d neurons are valid (NaN-free).\n', num_valid_neurons);

% --- Prepare data for NaN-sensitive analysis (like standard TCA) ---
% MOD: Create the final valid tensor from the combined one
supermouse_combined_valid = supermouse_big_tensor(valid_neuron_indices, :, :);

% --- Create Labels for the Combined Tensor ---
fprintf('Creating mouse and group labels...\n');
% MOD: Build labels for the combined set of mice
mouse_labels_all = [];
group_labels_all = []; % 1 for Task, 2 for Control

% Task mice labels
for ianimal = 1:n_animals_task
    n_i = size(X{ianimal}, 1);
    mouse_labels_all = [mouse_labels_all; ianimal * ones(n_i, 1)]; % Use original task index 1:n_animals_task
    group_labels_all = [group_labels_all; 1 * ones(n_i, 1)]; % Task group = 1
end

% Control mice labels
for i_control_valid = 1:n_animals_control
    ianimal_overall = n_animals_task + i_control_valid; % Overall index
    n_i = size(Y{i_control_valid}, 1);
    mouse_labels_all = [mouse_labels_all; ianimal_overall * ones(n_i, 1)]; % Use overall index
    group_labels_all = [group_labels_all; 2 * ones(n_i, 1)]; % Control group = 2
end

% MOD: Filter labels based on valid neurons found in the combined tensor
mouse_labels_valid = mouse_labels_all(valid_neuron_indices);
group_labels_valid = group_labels_all(valid_neuron_indices);

fprintf('Combined tensor construction complete.\n');
fprintf('  Dimensions of valid tensor: [%d neurons x %d bins x %d trials]\n', size(supermouse_combined_valid, 1), size(supermouse_combined_valid, 2), size(supermouse_combined_valid, 3));
fprintf('  Total valid mice included: %d Task, %d Control\n', n_animals_task, n_animals_control); % Report actual included counts

% --- Define Colors (Keep as before, or adjust if needed) ---
color_dms = [0, 0.4470, 0.7410];       % Deep Blue for DMS
color_dls =  [0.4660, 0.6740, 0.1880];  % Forest Green for DLS
color_acc = [0.8500, 0.3250, 0.0980];  % Crimson Red for ACC
colors = [color_dms; color_dls; color_acc]; % For area plots later

%% Cross-Validation to Choose Number of Factors
% Set up parameters for CP_NMU
maxNumFactors = 8;
max_iterations = 100;
num_initialisations = 25;
normalisation = 'min-max';  % adjust if needed
tca_method = 'cp_nmu';

% Use a cross-validation function to test different factor numbers.
% The function should run CP_NMU repeatedly on training/test splits and return:
%   best_mdl: the best model (a ktensor) fitted on the full super-mouse data
%   variance_explained: vector of variance explained per component
%   mean_cv_errors, sem_cv_errors: CV reconstruction errors
[best_mdl, variance_explained, bic_values, best_nFactors, all_best_models, recon_errors, init_similarity_scores] =...
    tca_with_bic_extended(supermouse_combined_valid, tca_method, normalisation, maxNumFactors, max_iterations, num_initialisations);


% Manual inspection and selection
figure
plot(bic_values, 'o-')
hold on
yyaxis('right')
plot(recon_errors/max(recon_errors), '-*')
plot(init_similarity_scores)
legend({'BIC', 'reconstruction error', 'average factor similarity'})
box off
xlabel('factor #')

best_n_factors = 4;
best_mdl = all_best_models{best_n_factors};

% best_mdl.U{1}: Neuron factors (for super-mouse; size = (sum_i n_i) x R)
% best_mdl.U{2}: Common spatial factors (size = B x R)
% best_mdl.U{3}: Trial factors (size = T_common x R)


%% Visualize Common Spatial Factors (Mode 2)

% --- Define Zone Parameters (Ideally outside the loop) ---
visual_zone_start_au = 80;
reward_zone_start_au = 100;
reward_zone_end_au = 135;
corridor_end_au = 200;
bin_size = 4;
bin_edges = 0:bin_size:corridor_end_au;
bin_edges(end) = corridor_end_au + bin_size; % Makes last bin centered at 202 if end=200
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres); % Number of bins corresponds to rows in U{2}
visual_start_idx = find(bin_centres >= visual_zone_start_au, 1, 'first');
reward_start_idx = find(bin_centres >= reward_zone_start_au, 1, 'first');
reward_end_idx   = find(bin_centres <= reward_zone_end_au, 1, 'last');
visual_end_idx = reward_start_idx - 1;

factor_data = best_mdl.U{2};
y_min_all = min(factor_data(:));
y_max_all = max(factor_data(:));

figure;
nFactors = best_n_factors;
t = tiledlayout('flow', 'TileSpacing', 'compact');
for r = 1:nFactors
    nexttile
    plot(best_mdl.U{2}(:,r), 'LineWidth', 2);
    hold on

    % Define common y vertices for patches
    y_patch_vertices = [y_min_all, y_min_all, y_max_all, y_max_all];

    % Define x vertices and draw Visual Zone patch (if valid)
    if ~isempty(visual_start_idx)
        x_visual_vertices = [visual_start_idx - 0.5, visual_end_idx + 0.5, visual_end_idx + 0.5, visual_start_idx - 0.5];
        patch(x_visual_vertices, y_patch_vertices, 'b', ... % Blue for visual
            'EdgeColor', 'none', 'FaceAlpha', 0.15);
    end

    % Define x vertices and draw Reward Zone patch (if valid)
    if ~isempty(reward_start_idx)
        x_reward_vertices = [reward_start_idx - 0.5, reward_end_idx + 0.5, reward_end_idx + 0.5, reward_start_idx - 0.5];
        patch(x_reward_vertices, y_patch_vertices, 'r', ... % Red for reward
            'EdgeColor', 'none', 'FaceAlpha', 0.15);
    end

    % Ensure line plot remains visible (redundant if plotted first, but safe)
    uistack(findobj(gca, 'Type', 'line'), 'top');


    title(sprintf('Spatial Factor %d', r));
    
end
xlabel(t, 'Spatial Bin');
ylabel(t, 'Loading');
title(t, 'Spatial Factors');

%% Visualize Neuron Factors (Mode 1) color-coded by mouse identity
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');
for r = 1:nFactors
    nexttile
    scatter(1:size(best_mdl.U{1},1), best_mdl.U{1}(:,r), 10, mouse_labels_valid, 'filled');
    colormap(parula(n_animals_task));
    colorbar;
    title(sprintf('Neuron Factor %d', r));
    
    axis tight
end
xlabel(t, 'Neuron Index');
ylabel(t, 'Loading');
title(t, 'Neuron Factors (Color-coded by Mouse)');

%% Visualize Trial Factors (Mode 3)
% Here, trials 1-10 correspond to the first 10 trials,
% 11-20 to the 10 trials immediately before the learning point,
% and 21-30 to the 10 trials immediately after the learning point.
figure
t = tiledlayout('flow', 'TileSpacing', 'compact');
for r = 1:nFactors
    nexttile
    plot(best_mdl.U{3}(:,r), 'LineWidth', 2);
    hold on;
    % Mark boundaries between segments
    xline(10, 'k');  % Boundary after trial 10
    xline(20, 'k');  % Boundary after trial 20
    title(sprintf('Trial Factor %d', r));

    hold off;
end
xlabel(t, 'Trial (Aligned)');
ylabel(t, 'Loading');
title(t, 'Trial Factors (Aligned Across Segments)');

%% Analyze and Visualize Trial Factors by Epoch
% This section analyzes the loadings of each trial factor (U{3}) across
% the three defined learning epochs: Early (trials 1-10), Middle (trials 11-20),
% and Late (trials 21-30). It uses error bar plots, ANOVA, and multiple
% comparisons to identify significant differences.

% Define colors for three learning epochs: early, middle, expert
early_color = [0.298, 0.447, 0.690];   % Blue-ish
middle_color = [0.867, 0.518, 0.322];    % Orange-ish
expert_color = [0.333, 0.776, 0.333];    % Green-ish

epoch_colors = [early_color; middle_color; expert_color];

% --- Define Epochs and Colors ---
epoch_indices = {1:10, 11:20, 21:30}; % Indices within the 30 trials of U{3}
epoch_names = {'Naive', 'Intermediate', 'Expert'}; % Labels for the epochs

% --- Setup Figure ---
figure; % Create a new figure for this analysis
t_epoch = tiledlayout('flow', 'TileSpacing', 'compact'); % Use tiled layout for subplots
nFactors = best_n_factors; % Use the selected number of factors

% --- Loop Through Factors ---
fprintf('Analyzing Trial Factors across epochs...\n');
for r = 1:nFactors
    nexttile(t_epoch); % Move to the next tile for the current factor

    factor_loadings = best_mdl.U{3}(:, r); % Loadings for factor r (30x1 vector)

    % --- Extract data for each epoch ---
    data_for_plot = cell(1, length(epoch_names));
    data_vec = []; % Vector for all data points for ANOVA
    group_vec = []; % Grouping variable for ANOVA
    for e = 1:length(epoch_names)
        epoch_data = factor_loadings(epoch_indices{e});
        data_for_plot{e} = epoch_data; % Store data per epoch for plotting

        % Append data and group labels for ANOVA
        data_vec = [data_vec; epoch_data]; %#ok<AGROW>
        group_vec = [group_vec; repmat(epoch_names(e), numel(epoch_data), 1)]; %#ok<AGROW>
    end

    % --- Plotting with my_simple_errorbar_plot ---
    hold on; % Hold on to add significance stars later
    try
        % Assumes my_simple_errorbar_plot takes cell array data and color matrix
        my_simple_errorbar_plot(data_for_plot, epoch_colors);
        set(gca, 'XTick', 1:length(epoch_names), 'XTickLabel', epoch_names); % Label x-axis ticks
        xlim([0.5 length(epoch_names) + 0.5]); % Adjust x-limits for clarity
    catch ME_plot
        warning('Error during my_simple_errorbar_plot for factor %d: %s\nTrying boxplot instead.', r, ME_plot.message);
        % Fallback basic plot if custom function fails
        boxplot(data_vec, group_vec, 'Colors', epoch_colors, 'Symbol', 'o', 'Widths', 0.6);
        set(gca, 'XTickLabel', epoch_names);
    end
    title(sprintf('Factor %d', r));
    % ylabel('Loading'); % Y-label per subplot (optional, can use overall ylabel)

    % --- Statistics (ANOVA and Multiple Comparisons) ---
    % Check if there are multiple groups and enough data points
    unique_groups = unique(group_vec);
    if numel(unique_groups) > 1 && numel(data_vec) > numel(unique_groups)
        try
            % Create categorical array to ensure group order matches plot order for stats
            % This assumes my_simple_errorbar_plot plots in the cell order 1, 2, 3
            group_cat = categorical(group_vec, epoch_names, 'Ordinal', false);

            % Run one-way ANOVA (suppress figure)
            [p_anova, tbl, stats] = anova1(data_vec, group_cat, 'off');

            % Only proceed with multiple comparisons if ANOVA is significant (optional but common)
            if p_anova < 0.05
                 % Run multiple comparisons using Tukey-Kramer (suitable for all pairs)
                mc = multcompare(stats, 'Display', 'off', 'Alpha', 0.05, 'CType', 'tukey-kramer');
                % mc columns: [group1_idx, group2_idx, lower_CI, mean_diff, upper_CI, p_value]
                % Indices correspond to the order in stats.gnames (which matches epoch_names)

                % --- Identify significant pairs for sigstar ---
                sigPairs = {};
                pvals = [];
                for i = 1:size(mc, 1)
                    % Check if the p-value (column 6) is significant
                    if mc(i, 6) < 0.05
                        % Get the indices of the groups being compared (these match plot positions 1, 2, 3)
                        pair = sort([mc(i, 1), mc(i, 2)]);
                        sigPairs{end+1} = pair; %#ok<AGROW>
                        pvals(end+1) = mc(i, 6); %#ok<AGROW>
                    end
                end

                % Ensure pairs are unique (mainly relevant if >3 groups, but good practice)
                if ~isempty(sigPairs)
                    [sigPairs_str, unique_idx] = unique(cellfun(@mat2str, sigPairs, 'UniformOutput', false));
                    pvals = pvals(unique_idx);
                    sigPairs = cellfun(@str2num, sigPairs_str, 'UniformOutput', false); % Convert back to numeric pairs

                    % --- Add sigstar (check if function exists) ---
                    if ~isempty(which('sigstar'))
                        sigstar(sigPairs, pvals);
                    else
                        fprintf('sigstar function not found. Skipping significance stars for factor %d.\n', r);
                        % Optionally add text indication for significant pairs
                        % for p_idx = 1:numel(sigPairs)
                        %     text(mean(sigPairs{p_idx}), max(ylim)*0.9, '*', 'HorizontalAlignment', 'center');
                        % end
                    end
                end % if ~isempty(sigPairs)
            % else % Optional: Indicate if ANOVA was not significant
            %     text(mean(xlim), mean(ylim), sprintf('ANOVA p=%.2f', p_anova), 'HorizontalAlignment', 'center');
            end % if p_anova < 0.05

        catch ME_stats
             warning('Error during ANOVA/multcompare for factor %d: %s', r, ME_stats.message);
        end % try-catch for stats
    else
        fprintf('Skipping stats for factor %d: Not enough groups or data.\n', r);
    end % if enough groups/data

    hold off; % Release hold for the next subplot

end % End factor loop

% --- Add Overall Labels ---
xlabel(t_epoch, 'Learning Epoch');
ylabel(t_epoch, 'Loading'); % Add overall Y label
title(t_epoch, 'Trial Factors by Learning Epoch');
%% Visualize aligned z-scored lick errors
zscored_lick_errors = {task_data(:).zscored_lick_errors};
aligned_lick_errors = nan(n_animals_task, 30);

for ianimal = 1:n_animals_task
    aligned_lick_errors(ianimal, 1:10) = zscored_lick_errors{ianimal}(1:10);
    aligned_lick_errors(ianimal, 11:20) = zscored_lick_errors{ianimal}(learning_point{ianimal}-10:learning_point{ianimal}-1);
    aligned_lick_errors(ianimal, 21:30) = zscored_lick_errors{ianimal}(learning_point{ianimal}+1:learning_point{ianimal}+10);

end

% Plot lick error
figure
shadedErrorBar(1:30, mean(aligned_lick_errors, 'omitmissing'), sem(aligned_lick_errors))
xline([10, 20], '--')
xlabel('trial')
ylabel('z-scored lick error')
axis tight
yline(-2, 'r')


%% Visualize Neuron Factors by Area

% This assumes that for each animal in task_data,
% task_data(ianimal).spatial_binned_fr_all is [n_i x B x T_i]
% and task_data(ianimal).is_dms, is_dls, is_acc are logical vectors of length n_i.
area_labels = cell(num_valid_neurons, 1);
current_idx = 1;
for ianimal = 1:n_animals_total
    if ianimal <= n_animals_task
        n_i = size(task_data(ianimal).spatial_binned_fr_all, 1);
        dms = task_data(ianimal).is_dms;
        dls = task_data(ianimal).is_dls;
        acc = task_data(ianimal).is_acc;
    else
        n_i = size(control_data(ianimal - n_animals_task).spatial_binned_fr_all, 1);
        dms = control_data(ianimal - n_animals_task).is_dms;
        dls = control_data(ianimal - n_animals_task).is_dls;
        acc = control_data(ianimal - n_animals_task).is_acc;
    end

    
    for j = 1:n_i
        if dms(j)
            area_labels{current_idx} = 'DMS';
        elseif dls(j)
            area_labels{current_idx} = 'DLS';
        elseif acc(j)
            area_labels{current_idx} = 'ACC';
        else
            area_labels{current_idx} = 'Unknown';
        end
        current_idx = current_idx + 1;
    end
end

area_labels = area_labels(valid_neuron_indices);

% Run ANOVA and Multiple Comparisons for Each Factor and Visualize with Sigstar
% This code assumes that 'area_labels' is a cell array of strings indicating
% the area for each neuron (e.g. 'DMS', 'DLS', or 'ACC'), and that for each factor
% the neuron loadings are in best_mdl.U{1}(:,r).
unique_areas = {'DMS','DLS','ACC'};
% colors is defined as:
% colors = [0,0.4470,0.7410; 0.4660,0.6740,0.1880; 0.8500,0.3250,0.0980];

figure;
t = tiledlayout('flow', 'TileSpacing', 'compact');
for r = 1:nFactors
    nexttile;
    % Gather data by area for factor r.
    data = cell(1, length(unique_areas));
    for a = 1:length(unique_areas)
        idx = strcmp(area_labels, unique_areas{a});
        data{a} = best_mdl.U{1}(idx, r);
    end

    % Plot using your custom errorbar plot (assumes it accepts a cell array and a color matrix)
    my_simple_errorbar_plot(data, colors)
    xticklabels(unique_areas)
    title(sprintf('Factor %d', r));
    ylabel('Loading');

    % Combine data for ANOVA
    data_vec = [];
    group_vec = [];
    for a = 1:length(unique_areas)
        curData = data{a};
        data_vec = [data_vec; curData];
        group_vec = [group_vec; repmat(unique_areas(a), length(curData), 1)];
    end

    % Run one-way ANOVA (suppress the figure output)
    [p, tbl, stats] = anova1(data_vec, group_vec, 'off');
    % Run multiple comparisons test
    mc = multcompare(stats, 'Display', 'off');

    % mc is an M-by-6 matrix with columns:
    % [group1, group2, lower limit, estimate, upper limit, p-value]
    % Build a cell array of pairs for which p < 0.05.
    sigPairs = {};
    pvals = [];
    for i = 1:size(mc,1)
        if mc(i,6) < 0.05
            % The group numbers correspond to the order in stats.gnames (should be DMS=1, DLS=2, ACC=3)
            sigPairs{end+1} = [mc(i,1) mc(i,2)];  %#ok<AGROW>
            pvals(end+1) = mc(i,6);  %#ok<AGROW>
        end
    end

    if ~isempty(sigPairs)
        sigstar(sigPairs, pvals);
    end
end
xlabel(t, 'Area');
ylabel(t, 'Neuron Factor Loading');
title(t, 'ANOVA of Neuron Factors by Area with Significant Differences');

%% Downsample to common number across areas

% Get indices for each area
idx_DMS = find(strcmp(area_labels, 'DMS'));
idx_DLS = find(strcmp(area_labels, 'DLS'));
idx_ACC = find(strcmp(area_labels, 'ACC'));

% Count the number of units in each area
n_DMS = numel(idx_DMS);
n_DLS = numel(idx_DLS);
n_ACC = numel(idx_ACC);

% Determine the lowest common number
target_units = min([n_DMS, n_DLS, n_ACC]);
fprintf('Downsampling to %d units per area\n', target_units);

% Randomly sample the target number from each area
ds_idx_DMS = randsample(idx_DMS, target_units);
ds_idx_DLS = randsample(idx_DLS, target_units);
ds_idx_ACC = randsample(idx_ACC, target_units);

% Store the downsampled indices in a structure for convenience
downsampled_indices_all.DMS = ds_idx_DMS;
downsampled_indices_all.DLS = ds_idx_DLS;
downsampled_indices_all.ACC = ds_idx_ACC;

% For each factor, we combine the downsampled neuron loadings by area, run a one-way ANOVA,
% perform multiple comparisons, and add significance markers using sigstar.

unique_areas = {'DMS','DLS','ACC'};
colors = [0,0.4470,0.7410; 0.4660,0.6740,0.1880; 0.8500,0.3250,0.0980];

figure;
t = tiledlayout('flow', 'TileSpacing', 'compact');
for r = 1:nFactors
    nexttile;
    % Gather data by area for factor r using the downsampled indices
    data = cell(1, length(unique_areas));
    for a = 1:length(unique_areas)
        switch unique_areas{a}
            case 'DMS'
                idx = downsampled_indices_all.DMS;
            case 'DLS'
                idx = downsampled_indices_all.DLS;
            case 'ACC'
                idx = downsampled_indices_all.ACC;
        end
        data{a} = best_mdl.U{1}(idx, r);
    end

    % Plot the errorbars
    my_simple_errorbar_plot(data, colors)
    xticklabels(unique_areas)
    title(sprintf('Factor %d', r));

    % Combine data for ANOVA
    data_vec = [];
    group_vec = [];
    for a = 1:length(unique_areas)
        curData = data{a};
        data_vec = [data_vec; curData];
        group_vec = [group_vec; repmat(unique_areas(a), length(curData), 1)];
    end

    % Run one-way ANOVA without displaying a figure
    [p, tbl, stats] = anova1(data_vec, group_vec, 'off');
    % Run multiple comparisons
    mc = multcompare(stats, 'Display', 'off');

    % mc columns: [group1, group2, lower limit, estimate, upper limit, p-value]
    sigPairs = {};
    pvals = [];
    for i = 1:size(mc,1)
        if mc(i,6) < 0.05
            % Group numbers correspond to the order in stats.gnames (DMS=1, DLS=2, ACC=3)
            sigPairs{end+1} = [mc(i,1) mc(i,2)];  %#ok<SAGROW>
            pvals(end+1) = mc(i,6);  %#ok<SAGROW>
        end
    end

    if ~isempty(sigPairs)
        sigstar(sigPairs, pvals);
    end
end
xlabel(t, 'Area');
ylabel(t, 'Neuron Factor Loading');
title(t, 'ANOVA of Neuron Factors by Area (Downsampled) with Sigstar');

%% Trial factors vs lick error

figure
t = tiledlayout('flow', 'TileSpacing', 'compact');
for r = 1:nFactors
    nexttile 

    factor_loadings = best_mdl.U{3}(:, r);

    scatter(factor_loadings, -mean(aligned_lick_errors, 'omitmissing')', 75, 'filled', 'MarkerEdgeColor', 'w')
    lsline
    title(sprintf('factor %d', r))
end
xlabel(t, 'factor loading')
ylabel(t, 'inverse lick error')

%% 2. Factor Contribution by Mouse and Area (Heatmaps)
% This section visualizes the mean neuron loading for each factor,
% broken down by mouse ID and brain area, using separate heatmaps per area.
area_labels_cat = categorical(area_labels);

fprintf('Generating heatmaps of mean factor loadings by mouse and area...\n');

figure;
nAreas = numel(unique_areas);
t_heatmap = tiledlayout(1, nAreas, 'TileSpacing', 'compact', 'Padding', 'compact');

mean_loadings_all = []; % To store all values for consistent color limits

% Calculate mean loadings for each Factor x Mouse x Area combination
mean_loadings_faceted = nan(nFactors, n_animals_total, nAreas); % Factors x Mouse x Area
for r = 1:nFactors
    for m = 1:n_animals_total
        for a = 1:nAreas
            % Find indices for neurons from this mouse AND this area
            idx_combined = find(mouse_labels_valid == m & area_labels_cat == unique_areas{a});

            if ~isempty(idx_combined)
                mean_loadings_faceted(r, m, a) = mean(best_mdl.U{1}(idx_combined, r), 'omitmissing');
            % else, leave as NaN
            end
        end
    end
end
mean_loadings_all = mean_loadings_faceted(:); % Collect all values

% Determine common color limits, ignoring NaNs
clim_min = min(mean_loadings_all, [], 'omitnan');
clim_max = max(mean_loadings_all, [], 'omitnan');
if isempty(clim_min) || isempty(clim_max) || clim_min == clim_max % Handle cases with no data or constant data
    clim_min = 0; clim_max = 1;
end

% Plot heatmap for each area
for a = 1:nAreas
    nexttile(t_heatmap);
    area_data = mean_loadings_faceted(:, :, a); % Factors x Mouse data for this area

    imagesc(area_data);
    set(gca, 'YDir', 'normal'); % Place Factor 1 at the bottom if desired
    colormap("parula"); % Or another suitable colormap
    clim([clim_min, clim_max]); % Apply consistent color limits

    % Add labels and ticks
    title(unique_areas{a});
    if a == 1 % Add Y label only to the first plot
        ylabel('Factor Index');
    end
    xlabel('Mouse ID');
    yticks(1:nFactors);
    xticks(1:n_animals_total);
    xtickangle(45);
end

% Add a single colorbar for the whole figure
cb = colorbar(nexttile(t_heatmap, 1)); % Get handle from one tile
cb.Layout.Tile = 'east'; % Place colorbar to the east of the layout
ylabel(cb, 'Mean Neuron Loading');
title(t_heatmap, 'Mean Neuron Factor Loadings (Factors x Mouse ID) per Area');

%% 3. Low-Dimensional Embedding of Neurons (t-SNE)
% This section applies t-SNE to the neuron factor matrix (U{1}) to visualize
% the relationships between neurons in a 2D space, colored by area or mouse ID.

fprintf('Performing t-SNE on neuron factors and plotting...\n');

neuron_factors = best_mdl.U{1}; % N_valid_neurons x nFactors

% Check if tsne function is available (Statistics Toolbox)
if isempty(which('tsne'))
    warning('tsne function not found (requires Statistics and Machine Learning Toolbox). Skipping t-SNE plots.');
else
    % Run t-SNE (adjust parameters as needed, e.g., Perplexity, Distance metric)
    fprintf('  Running t-SNE (may take a moment)...\n');
    % Standardize features? tsne often works okay without it, but can help.
    % neuron_factors_std = zscore(neuron_factors); % Optional standardization
    embedding_2d = tsne(neuron_factors, 'NumDimensions', 2, 'Perplexity', 30);
    fprintf('  t-SNE finished.\n');

    % Plot t-SNE embedding colored by Brain Area
    figure;
    gscatter(embedding_2d(:,1), embedding_2d(:,2), area_labels_cat, colors, '.', 12); % Use area colors
    legend(unique_areas, 'Location', 'best');
    xlabel('t-SNE Dimension 1');
    ylabel('t-SNE Dimension 2');
    title('t-SNE Embedding of Neuron Factors (Colored by Area)');
    axis tight; % Adjust axis for better visualization

    % Plot t-SNE embedding colored by Mouse ID
    figure;
    % Create a categorical array for mouse IDs for gscatter legend
    mouse_labels_cat = categorical(mouse_labels_valid);
    n_mice = numel(unique(mouse_labels_valid)); % Use actual number of mice present
    mouse_colors = parula(n_mice); % Use parula or another map for mice
    gscatter(embedding_2d(:,1), embedding_2d(:,2), mouse_labels_cat, mouse_colors, '.', 12);
    % legend('Location', 'best'); % Legend might be too crowded for many mice
    xlabel('t-SNE Dimension 1');
    ylabel('t-SNE Dimension 2');
    title('t-SNE Embedding of Neuron Factors (Colored by Mouse ID)');
    axis tight;

end

%% 
early_color = [0.298, 0.447, 0.690];   % Blue-ish
middle_color = [0.867, 0.518, 0.322];    % Orange-ish
expert_color = [0.333, 0.776, 0.333];    % Green-ish

figure
hold on
for itrial = 1:30
    if itrial <= 10
        color = early_color;
    elseif itrial <=20
        color = middle_color;
    else
        color = expert_color;
    end
    scatter3(best_mdl.U{3}(itrial, 1), best_mdl.U{3}(itrial, 2), best_mdl.U{3}(itrial, 3), 75, 'filled', 'MarkerFaceColor', color, 'MarkerEdgeColor', 'k')
    xlabel('Factor 1')
    ylabel('Factor 2')
    zlabel('Factor 3')

    view(-15, 45)
end

%% 4. Correlate Trial Factors with Within-Epoch Behavior
% This section investigates if trial factor loadings (U{3}) correlate with
% trial-by-trial behavioral performance (mean inverse lick error across animals)
% *within* each defined learning epoch (e.g., Naive, Intermediate, Expert).

fprintf('Correlating trial factors with within-epoch mean inverse lick error...\n');

% --- Prepare Data ---
% Use the aligned lick errors (animals x 30 trials) from previous section
if ~exist('aligned_lick_errors', 'var')
    error('Variable aligned_lick_errors not found. Please run the lick error alignment section first.');
end
% Calculate mean inverse lick error across animals for each aligned trial
% Inverse error used so higher values indicate better performance (fewer errors)
mean_inv_lick_error = -mean(aligned_lick_errors, 1, 'omitmissing')'; % Result is a 30x1 column vector

trial_factors = best_mdl.U{3}; % 30 x nFactors matrix

% Define Epochs (use the same names and indices as your epoch ANOVA section)
epoch_indices = {1:10, 11:20, 21:30};
epoch_names = {'Naive', 'Intermediate', 'Expert'}; % Or 'Early', 'Middle', 'Late'

nFactors = best_n_factors; % From your factor selection
nEpochs = numel(epoch_names);

% Check for data consistency
if size(trial_factors, 1) ~= 30 || size(mean_inv_lick_error, 1) ~= 30
    error('Mismatch in trial dimensions between trial factors (%d) and lick error (%d). Expected 30.', ...
          size(trial_factors, 1), size(mean_inv_lick_error, 1));
end

% --- Calculate Correlations ---
correlation_coeffs = nan(nFactors, nEpochs);
p_values = nan(nFactors, nEpochs);

for r = 1:nFactors % Loop through each factor
    for e = 1:nEpochs % Loop through each epoch
        % Get trial indices for the current epoch
        current_trials = epoch_indices{e};

        % Extract factor loadings and behavior data for this epoch
        factor_data_epoch = trial_factors(current_trials, r);
        behavior_data_epoch = mean_inv_lick_error(current_trials);

        % Filter out any NaN pairs (important if mean_inv_lick_error has NaNs)
        valid_pair_idx = ~isnan(factor_data_epoch) & ~isnan(behavior_data_epoch);

        factor_data_filt = factor_data_epoch(valid_pair_idx);
        behavior_data_filt = behavior_data_epoch(valid_pair_idx);

        % Check if enough data points remain for correlation (e.g., need > 2)
        if sum(valid_pair_idx) > 2
            % Calculate Pearson correlation (linear relationship)
            % Consider 'Spearman' if non-linear monotonic relationship is suspected
            [R, P] = corr(factor_data_filt, behavior_data_filt, 'Type', 'Pearson');
            correlation_coeffs(r, e) = R;
            p_values(r, e) = P;
        else
            fprintf('Skipping correlation for Factor %d, Epoch %s: Not enough valid data points (%d).\n', ...
                    r, epoch_names{e}, sum(valid_pair_idx));
            % Leave values as NaN
        end
    end
end

% --- Visualize Correlations as Heatmap ---
figure;
imagesc(correlation_coeffs);
set(gca, 'YDir', 'normal'); % Place Factor 1 at bottom

% Use a diverging colormap centered at zero. RedBlue from File Exchange is good.
% If unavailable, use a built-in and interpret carefully. Example using 'coolwarm' if available,
% otherwise 'parula' is okay but less intuitive for correlations.

colormap(redblue);

cb = colorbar;
ylabel(cb, 'Pearson Correlation (R)');
caxis([-1 1]); % Set limits for correlation coefficients

% Add significance stars (*p<0.05, **p<0.01, ***p<0.001)
hold on;
for r = 1:nFactors
    for e = 1:nEpochs
        p_val = p_values(r, e);
        if ~isnan(p_val)
            star_str = '';
            if p_val < 0.001
                star_str = '***';
            elseif p_val < 0.01
                star_str = '**';
            elseif p_val < 0.05
                star_str = '*';
            end

            % Add text star - choose color for visibility
            current_corr = correlation_coeffs(r, e);
            text_color = 'k'; % Default black
            if ~isnan(current_corr) && abs(current_corr) > 0.6 % Heuristic: If background color is strong
                 text_color = 'w';
            end
            % text(e, r, star_str, 'HorizontalAlignment', 'center', ...
            %      'VerticalAlignment', 'middle', 'FontSize', 14, 'Color', text_color, 'FontWeight', 'bold');
            text(e, r, num2str(current_corr), 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'middle', 'FontSize', 14, 'Color', text_color, 'FontWeight', 'bold');
        end
    end
end
hold off;

% Add labels and title
xticks(1:nEpochs);
xticklabels(epoch_names);
xtickangle(30);
yticks(1:nFactors);
yticklabels(1:nFactors); % Or specific factor names if available
xlabel('Learning Epoch');
ylabel('Factor Index');
title({'Correlation: Trial Factor Loading vs.', 'Mean Inverse Lick Error (within each epoch)'});