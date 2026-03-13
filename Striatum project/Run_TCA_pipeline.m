%% ================= Configuration =======================================================
clear; clc; close all;
% rng(0)
% --- Data Files ---
cfg.task_data_file = "preprocessed_data.mat";
cfg.control_data_file = "preprocessed_data_control.mat";
cfg.control2_data_file = "preprocessed_data_control2.mat";

% --- Analysis Selection ---
cfg.analysis_mode = 'task_only'; % Options: 'task_only', 'control_only', 'task_and_control'
cfg.areas_to_include = {'DMS', 'DLS', 'ACC'}; % List areas to keep (e.g., {'DMS', 'DLS', 'ACC'})
% Define mapping from area names to field names in the data struct
cfg.area_field_map = containers.Map(...
    {'DMS', 'DLS', 'ACC'}, ...
    {'is_dms', 'is_dls', 'is_acc'} ...
    );

% --- Processing Parameters ---
cfg.control_epoch_method = 'avg_task_lp'; % How to align control trials
cfg.control_fixed_ref_trial = 40;
cfg.control_epoch_windows = {1:10, [-10, -1], [1, 10]}; % {early_range, pre_lp_offset, post_lp_offset} relative to reference point
cfg.task_lp_zscore_threshold = -2;
cfg.task_lp_window_length = 10; % movsum uses [0, N-1], so 10 means current + next 9
cfg.task_lp_min_consecutive = 7;

% --- TCA Parameters ---
cfg.tca.method = 'cp_nmu';
cfg.tca.normalization = 'min-max';
cfg.tca.max_factors = 8;
cfg.tca.max_iterations = 200;
cfg.tca.num_initialisations = 25;
cfg.tca.select_factors_method = 'bic'; % or 'manual' or 'fixed'
cfg.tca.fixed_n_factors = 5; % Used if select_factors_method is 'fixed' or as fallback/manual choice

% --- Plotting Parameters ---
cfg.plot.zone_params.visual_zones_au  = [80 100];
cfg.plot.zone_params.reward_zone_au   = [100 135];
cfg.plot.zone_params.corridor_end_au = 200;
cfg.plot.zone_params.bin_size = 4;

cfg.plot.colors.dms = [0, 0.4470, 0.7410];
cfg.plot.colors.dls = [0.4660, 0.6740, 0.1880];
cfg.plot.colors.acc = [0.8500, 0.3250, 0.0980];
cfg.plot.colors.area_map = containers.Map(...
    {'DMS', 'DLS', 'ACC'}, ...
    {cfg.plot.colors.dms, cfg.plot.colors.dls, cfg.plot.colors.acc}...
    );
cfg.plot.colors.epoch_early = [0.298, 0.447, 0.690];
cfg.plot.colors.epoch_middle = [0.867, 0.518, 0.322];
cfg.plot.colors.epoch_expert = [0.333, 0.776, 0.333];
cfg.plot.epoch_names = {'Naive', 'Intermediate', 'Expert'}; % Corresponds to cfg.control_epoch_windows order

% --- Analysis Flags (Control which analyses/plots are generated) ---
cfg.run_tca = true;
cfg.plot.spatial_factors = true;
cfg.plot.neuron_factors_by_mouse = true; % Requires TCA
cfg.plot.neuron_factors_by_area = true; % Requires TCA
cfg.plot.neuron_factors_by_area_downsampled = true; % Requires TCA
cfg.plot.trial_factors = true; % Requires TCA
cfg.plot.trial_factors_by_epoch = true; % Requires TCA
cfg.plot.aligned_behavior = true; % Requires task data
cfg.plot.trial_factors_vs_behavior = true; % Requires TCA and task behavior
cfg.plot.factor_heatmaps_by_mouse_area = true; % Requires TCA
cfg.plot.tsne_embedding = false; % Requires TCA and Statistics Toolbox
cfg.plot.trial_factor_scatter3d = false; % Requires TCA and at least 3 factors
cfg.plot.trial_factors_vs_epoch_behavior = true; % Requires TCA and task behavior

%% ================= Data Loading and Filtering ==========================================
fprintf('--- Loading and Filtering Data ---\n');
task_data_raw = [];
control_data_raw = [];
avg_learning_point = []; % Ensure avg_learning_point is initialized

% --- Load Task Data ONLY if needed ---
% Needed if:
% 1. analysis_mode includes 'task' OR
% 2. analysis_mode is 'control_only' AND control_epoch_method requires it ('avg_task_lp')
needs_task_data_for_processing = strcmp(cfg.analysis_mode, 'task_only') || strcmp(cfg.analysis_mode, 'task_and_control');
needs_task_data_for_control_align = strcmp(cfg.analysis_mode, 'control_only') && strcmp(cfg.control_epoch_method, 'avg_task_lp');

if needs_task_data_for_processing || needs_task_data_for_control_align
    fprintf('Loading Task data from: %s (Required for analysis or control alignment)\n', cfg.task_data_file);
    if isfile(cfg.task_data_file)
        loaded_data = load(cfg.task_data_file, "preprocessed_data");
        task_data_raw = filterDataByArea(loaded_data.preprocessed_data, cfg.areas_to_include, cfg.area_field_map);
        fprintf('  Filtered Task data to %d areas: %s\n', numel(cfg.areas_to_include), strjoin(cfg.areas_to_include, ', '));

        % --- Process Task Data immediately IF needed for avg_lp ---
        if ~isempty(task_data_raw)
            % We need avg_lp if using 'avg_task_lp' method later for control OR if doing task analysis
            if strcmp(cfg.control_epoch_method, 'avg_task_lp') || needs_task_data_for_processing
                fprintf('Processing Task data to find learning points...\n');
                % Call processTaskData BUT only keep avg_learning_point if just needed for control alignment
                [task_data_processed_temp, learning_points_task_temp, avg_learning_point, aligned_lick_errors_temp] = processTaskData(task_data_raw, cfg);

                if isempty(avg_learning_point) && strcmp(cfg.control_epoch_method, 'avg_task_lp')
                    error('Control epoch alignment requires average task LP, but it could not be determined from the task data.');
                end

                % Assign processed task data ONLY if doing task analysis
                if needs_task_data_for_processing
                    task_data = task_data_processed_temp;
                    learning_points_task = learning_points_task_temp;
                    aligned_lick_errors = aligned_lick_errors_temp; % Keep aligned errors only if task analysis is run
                    fprintf('  Task data processed for inclusion in analysis.\n');
                else
                    fprintf('  Task data processed ONLY for control alignment reference point (avg_lp = %d).\n', avg_learning_point);
                    % Discard task_data_processed_temp etc. if in 'control_only' mode
                    task_data = [];
                    learning_points_task = [];
                    aligned_lick_errors = [];
                end
            end
        else
            % Task data file loaded but was empty after filtering
            if strcmp(cfg.control_epoch_method, 'avg_task_lp')
                error('Task data loaded but empty after filtering. Cannot determine avg_learning_point for control alignment.');
            end
        end
    else
        % Task data file not found
        if needs_task_data_for_control_align || needs_task_data_for_processing
            error('Required Task data file not found: %s', cfg.task_data_file);
        end
    end
else
    fprintf('Skipping Task data loading (not needed for mode "%s" with method "%s").\n', cfg.analysis_mode, cfg.control_epoch_method);
    task_data = []; % Ensure task_data is empty if not loaded/processed for analysis
    learning_points_task = [];
    aligned_lick_errors = [];
end

% --- Load Control Data if needed ---
if strcmp(cfg.analysis_mode, 'control_only') || strcmp(cfg.analysis_mode, 'task_and_control')
    fprintf('Loading Control data from: %s\n', cfg.control_data_file);
    if isfile(cfg.control_data_file)
        loaded_data = load(cfg.control_data_file, "preprocessed_data");
        control_data_raw = filterDataByArea(loaded_data.preprocessed_data, cfg.areas_to_include, cfg.area_field_map);
        fprintf('  Filtered Control data to %d areas: %s\n', numel(cfg.areas_to_include), strjoin(cfg.areas_to_include, ', '));
    else
        warning('Control data file not found: %s', cfg.control_data_file);
        control_data_raw = []; % Ensure it's empty if not found
    end
else
    control_data_raw = []; % Ensure it's empty if mode doesn't include control
end

% Check if any data relevant to the chosen mode was loaded
if (strcmp(cfg.analysis_mode,'task_only') || strcmp(cfg.analysis_mode,'task_and_control')) && isempty(task_data_raw)
    warning('Task data file was empty or not found, but analysis mode requires it.');
    % Allow continuing if control data might still be processed
end
if (strcmp(cfg.analysis_mode,'control_only') || strcmp(cfg.analysis_mode,'task_and_control')) && isempty(control_data_raw)
    warning('Control data file was empty or not found, but analysis mode requires it.');
    % Allow continuing if task data might still be processed
end
if isempty(task_data_raw) && isempty(control_data_raw)
    error('No data loaded for task or control. Check file paths.');
end


%% ================= Data Processing =====================================================
fprintf('--- Processing Data ---\n');
% Task data processing might have already happened above if needed for avg_lp

control_data = []; % Initialize processed control data

% --- Process Control Data ---
if ~isempty(control_data_raw) % Only proceed if raw control data exists
    control_ref_point = NaN;
    control_indices_info = []; % Initialize

    % Determine the reference point for control alignment
    if strcmp(cfg.control_epoch_method, 'avg_task_lp')
        if ~isempty(avg_learning_point)
            control_ref_point = avg_learning_point;
            fprintf('Using average task learning point (%d) for control alignment.\n', control_ref_point);
        else
            % This case should have been caught earlier by erroring if task data was needed but failed
            error('Internal Error: avg_learning_point is empty, but should have been calculated or errored earlier.');
        end
    elseif strcmp(cfg.control_epoch_method, 'fixed_trial')
        if isfield(cfg, 'control_fixed_ref_trial') && ~isempty(cfg.control_fixed_ref_trial) && isnumeric(cfg.control_fixed_ref_trial)
            control_ref_point = cfg.control_fixed_ref_trial;
            fprintf('Using fixed reference trial (%d) for control alignment.\n', control_ref_point);
        else
            error('Control epoch method set to "fixed_trial", but cfg.control_fixed_ref_trial is not set or invalid.');
        end
    else
        error('Unknown cfg.control_epoch_method: "%s"', cfg.control_epoch_method);
    end

    % Process control data using the determined reference point
    if ~isnan(control_ref_point)
        [control_data, control_indices_info] = processControlData(control_data_raw, control_ref_point, cfg);
        if isempty(control_data) && (strcmp(cfg.analysis_mode,'control_only') || strcmp(cfg.analysis_mode,'task_and_control'))
            warning('No control animals met the trial criteria for the chosen epochs.');
            % Don't error here, maybe user only wanted task analysis if mode='task_and_control'
        end
    else
        warning('Could not determine a valid control reference point. Skipping control data processing.');
        control_data = []; % Ensure control_data is empty
    end
else
    fprintf('No raw control data loaded or available. Skipping control processing.\n');
    control_data = []; % Ensure control_data is empty
    control_indices_info = [];
end

% --- Final Check for Data to Analyze ---
% Check if we have data corresponding to the analysis_mode requested
analysis_possible = false;
if (strcmp(cfg.analysis_mode,'task_only') || strcmp(cfg.analysis_mode,'task_and_control')) && ~isempty(task_data)
    analysis_possible = true;
end
if (strcmp(cfg.analysis_mode,'control_only') || strcmp(cfg.analysis_mode,'task_and_control')) && ~isempty(control_data)
    analysis_possible = true;
end

if ~analysis_possible
    error('No valid data remaining for the selected analysis mode (%s). Check processing steps and filtering.', cfg.analysis_mode);
end


%% ================= Build Combined Tensor ===============================================
fprintf('--- Building Combined Tensor ---\n');

% Call the REVISED buildCombinedTensor function with explicit arguments
% Pass empty arrays ([]) for data types not included in the current analysis mode
task_data_to_pass = [];
control_data_to_pass = [];
task_lps_to_pass = [];
control_indices_to_pass = [];

if strcmp(cfg.analysis_mode,'task_only') || strcmp(cfg.analysis_mode,'task_and_control')
    task_data_to_pass = task_data;
    task_lps_to_pass = learning_points_task;
end
if strcmp(cfg.analysis_mode,'control_only') || strcmp(cfg.analysis_mode,'task_and_control')
    control_data_to_pass = control_data;
    control_indices_to_pass = control_indices_info;
end

[supermouse_tensor_raw, combined_labels, tensor_info] = buildCombinedTensor(...
    task_data_to_pass, control_data_to_pass, task_lps_to_pass, control_indices_to_pass, cfg);

% --- NaN Handling ---
fprintf('Checking for NaN values in the combined tensor...\n');
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_neuron_indices = find(~neuron_has_nan);
invalid_neuron_indices = find(neuron_has_nan);

num_total_neurons = size(supermouse_tensor_raw, 1);
num_invalid_neurons = numel(invalid_neuron_indices);
num_valid_neurons = numel(valid_neuron_indices);

if num_valid_neurons == 0
    error('No valid (NaN-free) neurons found in the combined tensor. Cannot proceed.');
end

fprintf('Found %d neurons with NaN values (%.1f%%).\n', ...
    num_invalid_neurons, (num_invalid_neurons/num_total_neurons)*100);
fprintf('%d neurons are valid (NaN-free).\n', num_valid_neurons);

% --- Create Final Valid Tensor and Labels ---
supermouse_combined_valid = supermouse_tensor_raw(valid_neuron_indices, :, :);
labels_valid.mouse_labels = combined_labels.mouse_labels_all(valid_neuron_indices);
labels_valid.group_labels = combined_labels.group_labels_all(valid_neuron_indices); % 1=Task, 2=Control
labels_valid.area_labels = combined_labels.area_labels_all(valid_neuron_indices); % Cell array of area names
labels_valid.neurontype_labels = combined_labels.neurontype_labels_all(valid_neuron_indices);

fprintf('Combined tensor construction complete.\n');
fprintf('  Dimensions of valid tensor: [%d neurons x %d bins x %d trials]\n', size(supermouse_combined_valid, 1), size(supermouse_combined_valid, 2), size(supermouse_combined_valid, 3));
fprintf('  Total valid mice included: %d Task, %d Control\n', tensor_info.n_animals_task, tensor_info.n_animals_control); % Report actual included counts

%% ================= Subsample Units by Area =============================================
fprintf('--- Subsampling units ---\n');

% Use the area labels from the valid labels structure
area_labels_to_sample = labels_valid.area_labels;

% Count units per area
[unique_areas, ~, area_indices_in_unique_list] = unique(area_labels_to_sample);
unit_counts_per_area = accumarray(area_indices_in_unique_list, 1);

% --- Parameters for subsampling ---
min_unit_threshold = 10; 

% Identify areas to keep (those with enough units)
areas_to_keep_mask = unit_counts_per_area >= min_unit_threshold;
areas_to_keep = unique_areas(areas_to_keep_mask);
counts_of_areas_to_keep = unit_counts_per_area(areas_to_keep_mask);

% Identify and report skipped areas
areas_to_skip = unique_areas(~areas_to_keep_mask);
if ~isempty(areas_to_skip)
    fprintf('Skipping areas with fewer than %d units: %s\n', min_unit_threshold, strjoin(areas_to_skip, ', '));
end

% Perform subsampling only if there are areas that meet the criteria
if isempty(areas_to_keep)
    warning('No areas met the minimum unit threshold of %d. Skipping subsampling.', min_unit_threshold);
else
    % Determine the number of units to subsample to (minimum of the kept areas)
    target_n_units = min(counts_of_areas_to_keep);
    fprintf('Subsampling all kept areas to %d units.\n', target_n_units);
    fprintf('Kept areas: %s\n', strjoin(areas_to_keep, ', '));

    % Perform subsampling
    final_indices_to_keep = [];
    for i = 1:length(areas_to_keep)
        current_area = areas_to_keep{i};
        
        % Find indices of all units from the current area
        indices_for_this_area = find(strcmp(area_labels_to_sample, current_area));
        
        % Randomly select 'target_n_units' from them without replacement
        shuffled_indices = indices_for_this_area(randperm(length(indices_for_this_area)));
        selected_indices = sort(shuffled_indices(1:target_n_units));
        
        % Append to the final list of indices (ensure it's a column vector)
        final_indices_to_keep = [final_indices_to_keep; selected_indices(:)];
    end

    % Overwrite the data tensor and labels with the new balanced dataset
    % This ensures all downstream functions use the subsampled data
    original_unit_count = size(supermouse_combined_valid, 1);
    supermouse_combined_valid = supermouse_combined_valid(final_indices_to_keep, :, :);
    labels_valid.mouse_labels = labels_valid.mouse_labels(final_indices_to_keep);
    labels_valid.group_labels = labels_valid.group_labels(final_indices_to_keep);
    labels_valid.area_labels = labels_valid.area_labels(final_indices_to_keep);
    labels_valid.neurontype_labels = labels_valid.neurontype_labels(final_indices_to_keep);
    
    fprintf('Subsampling complete. Dataset reduced from %d to %d total units.\n\n', ...
            original_unit_count, size(supermouse_combined_valid, 1));
end

%% ================= Analyze and Visualize Response Range ================================
fprintf('--- Analyzing response ranges per area ---\n');

% Calculate the firing rate range for each neuron in the (now balanced) dataset
num_units = size(supermouse_combined_valid, 1);
response_ranges = zeros(num_units, 1);

for iUnit = 1:num_units
    % Extract all firing data for the current unit across all bins and trials
    unit_data = squeeze(supermouse_combined_valid(iUnit, :, :));
    
    % Calculate the range (max value - min value) and store it
    response_ranges(iUnit) = max(unit_data(:)) - min(unit_data(:));
end

% Visualize the distribution of ranges for each area using a box plot
figure('Name', 'Response Range by Area');

boxplot(response_ranges, labels_valid.area_labels, 'OutlierSize', 3,...
    'Colors', [cfg.plot.colors.acc;cfg.plot.colors.dls;cfg.plot.colors.dms], 'Widths', 0.75);

title('Distribution of Firing Rate Range by Brain Area');
ylabel('Response Range (Max Rate - Min Rate)');

fprintf('Displayed box plot of response ranges.\n\n');

%% ================= Visualize Spatiotemporal Activity by Area and Epoch =================
fprintf('--- Visualizing spatiotemporal activity by area and epoch from the final tensor ---\n');

% --- Analysis Configuration ---
% These variables should be in the workspace:
% supermouse_combined_valid: The final [neurons x bins x trials] data tensor.
% labels_valid: Struct with aligned labels, including .area_labels.
% cfg: The main configuration struct.

% Define epochs based on the 30-trial aligned structure
epoch_trials = {1:10, 11:20, 21:30};
epoch_names = cfg.plot.epoch_names; % {'Naive', 'Intermediate', 'Expert'}
epoch_colors = {cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

% Get unique areas present in the final dataset
areas_in_data = unique(labels_valid.area_labels);
num_areas = numel(areas_in_data);


% --- Figure 1: Spatial Tuning Across Epochs (Per Area) ---
figure('Name', 'Spatial Tuning by Area and Epoch', 'Position', [100, 100, 500 * num_areas, 450]);
t_spatial = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');

for i_area = 1:num_areas
    current_area = areas_in_data{i_area};
    nexttile;
    hold on;
    
    legend_handles_spatial = [];
    
    % Get the subset of the tensor for the current area
    idx_area = strcmp(labels_valid.area_labels, current_area);
    area_tensor = supermouse_combined_valid(idx_area, :, :); % [area_neurons x bins x trials]
    
    for i_epoch = 1:numel(epoch_trials)
        % Get the trials for the current epoch
        trials_for_epoch = epoch_trials{i_epoch};
        epoch_tensor = area_tensor(:, :, trials_for_epoch);
        
        % Average across trials to get [neurons x bins] for the epoch
        activity_per_neuron_epoch = squeeze(mean(epoch_tensor, 3, 'omitnan'));
        
        % Calculate mean and SEM across neurons
        mean_fr_space = mean(activity_per_neuron_epoch, 1, 'omitnan');
        sem_fr_space = std(activity_per_neuron_epoch, 0, 1, 'omitnan') / sqrt(sum(idx_area));
        
        % Plot using shadedErrorBar
        h = shadedErrorBar(1:size(mean_fr_space, 2), mean_fr_space, sem_fr_space, ...
            'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
        
        legend_handles_spatial(end+1) = h.mainLine;
    end
    
    % --- Add visual context with patches for zones ---
    yl = ylim;
    y_patch = [yl(1), yl(1), yl(2), yl(2)];
    
    bin_size = cfg.plot.zone_params.bin_size;
    patch([floor(cfg.plot.zone_params.visual_zones_au(1)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(2)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(2)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(1)/bin_size)], y_patch, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    patch([floor(cfg.plot.zone_params.reward_zone_au(1)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(2)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(2)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(1)/bin_size)], y_patch, cfg.plot.colors.dls, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    uistack(legend_handles_spatial, 'top');

    title(current_area);
    box on;
    xlim([0, size(supermouse_combined_valid, 2)]);
end

% Add shared figure elements for Figure 1
title(t_spatial, 'Evolution of Spatial Tuning Across Learning Epochs', 'FontSize', 16);
xlabel(t_spatial, 'Spatial Bin', 'FontSize', 12);
ylabel(t_spatial, 'Mean Firing Rate (a.u.)', 'FontSize', 12);
lg_spatial = legend(legend_handles_spatial, epoch_names, 'Location', 'best');


% --- Figure 2: Temporal Activity Evolution (Across Areas) ---
figure('Name', 'Temporal Evolution of Activity by Area', 'Position', [200, 200, 900, 500]);
hold on;
legend_handles_temporal = [];

for i_area = 1:num_areas
    current_area = areas_in_data{i_area};

    % Get the subset of the tensor for the current area
    idx_area = strcmp(labels_valid.area_labels, current_area);
    area_tensor = supermouse_combined_valid(idx_area, :, :); % [area_neurons x bins x trials]
    
    % Average across spatial bins to get [neurons x trials]
    activity_per_neuron_temporal = squeeze(mean(area_tensor, 2, 'omitnan'));

    % Calculate mean and SEM across neurons
    mean_fr_time = mean(activity_per_neuron_temporal, 1, 'omitnan');
    sem_fr_time = std(activity_per_neuron_temporal, 0, 1, 'omitnan') / sqrt(sum(idx_area));
    
    % Get the appropriate color from the config
    area_color = cfg.plot.colors.area_map(current_area);
    
    % Plot using shadedErrorBar
    h = shadedErrorBar(1:size(mean_fr_time, 2), mean_fr_time, sem_fr_time, ...
        'lineprops', {'-','Color', area_color, 'LineWidth', 2});
        
    legend_handles_temporal(end+1) = h.mainLine;
end

% --- Add vertical lines to demarcate epochs ---
xline(0, 'k--', 'Naive', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
xline(10, 'k--', 'Intermediate', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
xline(20, 'k--', 'Expert', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');

% --- Final plot formatting ---
title('Evolution of Mean Activity Across Aligned Trials');
xlabel('Aligned Trial Number');
ylabel('Mean Firing Rate (averaged over space)');
xlim([0, size(supermouse_combined_valid, 3)]);
legend(legend_handles_temporal, areas_in_data, 'Location', 'northwest');
box on;
hold off;

fprintf('Displayed spatiotemporal analysis plots.\n\n');

%% ================= Visualize Spatiotemporal Activity by Area and Epoch =================
fprintf('--- Visualizing spatiotemporal activity by area and epoch from the final tensor ---\n');
% --- Analysis Configuration ---
% These variables should be in the workspace:
% supermouse_combined_valid: The final [neurons x bins x trials] data tensor.
% labels_valid: Struct with aligned labels, including .area_labels.
% cfg: The main configuration struct.

% REVISED: Define epochs based on the 30-trial aligned structure, splitting the first 10
epoch_trials = {1:3, 4:10, 11:20, 21:30};

% REVISED: Update names locally to reflect the split
epoch_names = {'Trials 1-3', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};

% REVISED: Define a new color for the first 3 trials (lighter version of 'early' color)
color_trials_1_3 = min(cfg.plot.colors.epoch_early + 0.3, 1); % Lighten the original early color
epoch_colors = {color_trials_1_3, cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

% Get unique areas present in the final dataset
areas_in_data = unique(labels_valid.area_labels);
num_areas = numel(areas_in_data);

% --- Figure 1: Spatial Tuning Across Epochs (Per Area) ---
figure('Name', 'Spatial Tuning by Area and Epoch', 'Position', [100, 100, 500 * num_areas, 450]);
t_spatial = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
for i_area = 1:num_areas
    current_area = areas_in_data{i_area};
    nexttile;
    hold on;
    
    legend_handles_spatial = [];
    
    % Get the subset of the tensor for the current area
    idx_area = strcmp(labels_valid.area_labels, current_area);
    area_tensor = supermouse_combined_valid(idx_area, :, :); % [area_neurons x bins x trials]
    
    for i_epoch = 1:numel(epoch_trials)
        % Get the trials for the current epoch
        trials_for_epoch = epoch_trials{i_epoch};
        epoch_tensor = area_tensor(:, :, trials_for_epoch);
        
        % Average across trials to get [neurons x bins] for the epoch
        activity_per_neuron_epoch = squeeze(mean(epoch_tensor, 3, 'omitnan'));
        
        % Calculate mean and SEM across neurons
        mean_fr_space = mean(activity_per_neuron_epoch, 1, 'omitnan');
        sem_fr_space = std(activity_per_neuron_epoch, 0, 1, 'omitnan') / sqrt(sum(idx_area));
        
        % Plot using shadedErrorBar
        h = shadedErrorBar(1:size(mean_fr_space, 2), mean_fr_space, sem_fr_space, ...
            'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
        
        legend_handles_spatial(end+1) = h.mainLine;
    end
    
    % --- Add visual context with patches for zones ---
    yl = ylim;
    y_patch = [yl(1), yl(1), yl(2), yl(2)];
    
    bin_size = cfg.plot.zone_params.bin_size;
    patch([floor(cfg.plot.zone_params.visual_zones_au(1)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(2)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(2)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(1)/bin_size)], y_patch, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    patch([floor(cfg.plot.zone_params.reward_zone_au(1)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(2)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(2)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(1)/bin_size)], y_patch, cfg.plot.colors.dls, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    uistack(legend_handles_spatial, 'top');
    title(current_area);
    box on;
    xlim([0, size(supermouse_combined_valid, 2)]);
end
% Add shared figure elements for Figure 1
title(t_spatial, 'Evolution of Spatial Tuning Across Learning Epochs', 'FontSize', 16);
xlabel(t_spatial, 'Spatial Bin', 'FontSize', 12);
ylabel(t_spatial, 'Mean Firing Rate (a.u.)', 'FontSize', 12);
lg_spatial = legend(legend_handles_spatial, epoch_names, 'Location', 'best');

% --- Figure 2: Temporal Activity Evolution (Across Areas) ---
figure('Name', 'Temporal Evolution of Activity by Area', 'Position', [200, 200, 900, 500]);
hold on;
legend_handles_temporal = [];
for i_area = 1:num_areas
    current_area = areas_in_data{i_area};
    % Get the subset of the tensor for the current area
    idx_area = strcmp(labels_valid.area_labels, current_area);
    area_tensor = supermouse_combined_valid(idx_area, :, :); % [area_neurons x bins x trials]
    
    % Average across spatial bins to get [neurons x trials]
    activity_per_neuron_temporal = squeeze(mean(area_tensor, 2, 'omitnan'));
    % Calculate mean and SEM across neurons
    mean_fr_time = mean(activity_per_neuron_temporal, 1, 'omitnan');
    sem_fr_time = std(activity_per_neuron_temporal, 0, 1, 'omitnan') / sqrt(sum(idx_area));
    
    % Get the appropriate color from the config
    area_color = cfg.plot.colors.area_map(current_area);
    
    % Plot using shadedErrorBar
    h = shadedErrorBar(1:size(mean_fr_time, 2), mean_fr_time, sem_fr_time, ...
        'lineprops', {'-','Color', area_color, 'LineWidth', 2});
        
    legend_handles_temporal(end+1) = h.mainLine;
end

% --- REVISED: Add vertical lines to demarcate the newly split epochs ---
xline(0, 'k--', epoch_names{1}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
xline(3, 'k--', epoch_names{2}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
xline(10, 'k--', epoch_names{3}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
xline(20, 'k--', epoch_names{4}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');

% --- Final plot formatting ---
title('Evolution of Mean Activity Across Aligned Trials');
xlabel('Aligned Trial Number');
ylabel('Mean Firing Rate (averaged over space)');
xlim([0, size(supermouse_combined_valid, 3)]);
legend(legend_handles_temporal, areas_in_data, 'Location', 'northwest');
box on;
hold off;
fprintf('Displayed spatiotemporal analysis plots with split naive epoch.\n\n');
%% ================= Visualize Spatiotemporal Activity by Neuron Type ====================
fprintf('--- Visualizing spatiotemporal activity by NEURON TYPE ---\n');

% --- Ensure Epoch Definitions Exist ---
epoch_trials = {1:10, 11:20, 21:30};
epoch_names = cfg.plot.epoch_names; 
epoch_colors = {cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

% --- Define Target Neuron Types (MSN, FSN, TAN) ---
neurontype_map = containers.Map({1, 2, 3}, {'MSN', 'FSN', 'TAN'});

% Sort keys to ensure plots appear in logical order (1->2->3) rather than random hash order
target_keys = sort(cell2mat(neurontype_map.keys));
num_types = numel(target_keys);

% Define colors for these specific types
type_colors = lines(num_types); 

% --- Figure 3: Spatial Tuning Across Epochs (Per Neuron Type) ---
figure('Name', 'Spatial Tuning by Neuron Type and Epoch', 'Position', [100, 100, 500 * num_types, 450]);
t_spatial_type = tiledlayout(1, num_types, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:num_types
    current_key = target_keys(i);
    current_name = neurontype_map(current_key);
    
    nexttile;
    hold on;
    legend_handles_spatial_type = [];
    
    % Filter data for this specific numeric key (e.g., 1, 2, or 3)
    % We assume labels_valid.neurontype_labels contains numeric codes matching the keys
    idx_type = (labels_valid.neurontype_labels == current_key);
    
    % Ensure no NaNs slip through (comparison with number implies false, but good to be safe)
    idx_type(isnan(labels_valid.neurontype_labels)) = false;

    n_units_this_type = sum(idx_type);
    
    if n_units_this_type > 0
        type_tensor = supermouse_combined_valid(idx_type, :, :); % [type_neurons x bins x trials]
        
        for i_epoch = 1:numel(epoch_trials)
            trials_for_epoch = epoch_trials{i_epoch};
            epoch_tensor = type_tensor(:, :, trials_for_epoch);
            
            % Average across trials
            activity_per_neuron_epoch = squeeze(mean(epoch_tensor, 3, 'omitnan'));
            
            % Mean and SEM across neurons
            mean_fr_space = mean(activity_per_neuron_epoch, 1, 'omitnan');
            sem_fr_space = std(activity_per_neuron_epoch, 0, 1, 'omitnan') / sqrt(n_units_this_type);
            
            % Plot
            h = shadedErrorBar(1:size(mean_fr_space, 2), mean_fr_space, sem_fr_space, ...
                'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
            if ~isempty(h) && isvalid(h.mainLine)
                 legend_handles_spatial_type(end+1) = h.mainLine;
            end
        end
        
        % Context Patches (Visual/Reward)
        yl = ylim;
        y_patch = [yl(1), yl(1), yl(2), yl(2)];
        bin_size = cfg.plot.zone_params.bin_size;
        
        patch([floor(cfg.plot.zone_params.visual_zones_au(1)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(2)/bin_size), ...
               floor(cfg.plot.zone_params.visual_zones_au(2)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(1)/bin_size)], ...
               y_patch, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
           
        patch([floor(cfg.plot.zone_params.reward_zone_au(1)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(2)/bin_size), ...
               floor(cfg.plot.zone_params.reward_zone_au(2)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(1)/bin_size)], ...
               y_patch, cfg.plot.colors.dls, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
               
        if ~isempty(legend_handles_spatial_type)
            uistack(legend_handles_spatial_type, 'top');
        end
        title(sprintf('%s (n=%d)', current_name, n_units_this_type));
        box on;
        xlim([0, size(supermouse_combined_valid, 2)]);
    else
        title(sprintf('%s (n=0)', current_name));
        text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
    end
end

title(t_spatial_type, 'Spatial Tuning by Neuron Type (MSN, FSN, TAN)', 'FontSize', 16);
xlabel(t_spatial_type, 'Spatial Bin', 'FontSize', 12);
ylabel(t_spatial_type, 'Mean Firing Rate (a.u.)', 'FontSize', 12);
if ~isempty(legend_handles_spatial_type)
    lg_spatial_type = legend(legend_handles_spatial_type, epoch_names, 'Location', 'best');
end


% --- Figure 4: Temporal Activity Evolution (By Neuron Type) ---
figure('Name', 'Temporal Evolution by Neuron Type', 'Position', [250, 250, 900, 500]);
hold on;
legend_handles_temporal_type = [];
legend_names_temporal_type = {};

for i = 1:num_types
    current_key = target_keys(i);
    current_name = neurontype_map(current_key);
    
    idx_type = (labels_valid.neurontype_labels == current_key);
    idx_type(isnan(labels_valid.neurontype_labels)) = false;

    if sum(idx_type) > 0
        type_tensor = supermouse_combined_valid(idx_type, :, :); 
        
        % Average across spatial bins
        activity_per_neuron_temporal = squeeze(mean(type_tensor, 2, 'omitnan'));
        
        % Mean/SEM across neurons
        mean_fr_time = mean(activity_per_neuron_temporal, 1, 'omitnan');
        sem_fr_time = std(activity_per_neuron_temporal, 0, 1, 'omitnan') / sqrt(sum(idx_type));
        
        this_color = type_colors(i, :);
        
        h = shadedErrorBar(1:size(mean_fr_time, 2), mean_fr_time, sem_fr_time, ...
            'lineprops', {'-','Color', this_color, 'LineWidth', 2});
            
        if ~isempty(h) && isvalid(h.mainLine)
            legend_handles_temporal_type(end+1) = h.mainLine;
            legend_names_temporal_type{end+1} = current_name;
        end
    end
end

xline(0, 'k--', 'Naive', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
xline(10, 'k--', 'Intermediate', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
xline(20, 'k--', 'Expert', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');

title('Evolution of Mean Activity by Neuron Type');
xlabel('Aligned Trial Number');
ylabel('Mean Firing Rate (averaged over space)');
xlim([0, size(supermouse_combined_valid, 3)]);

if ~isempty(legend_handles_temporal_type)
    legend(legend_handles_temporal_type, legend_names_temporal_type, 'Location', 'northwest');
end
box on;
hold off;

fprintf('Displayed neuron type analysis plots (MSN, FSN, TAN).\n\n');

%% ================= Visualize Spatiotemporal Activity by Neuron Type ====================
fprintf('--- Visualizing spatiotemporal activity by NEURON TYPE ---\n');

% --- Ensure Epoch Definitions Exist ---
epoch_trials = {1:10, 11:20, 21:30};
epoch_names = cfg.plot.epoch_names; 
epoch_colors = {cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

% --- Define Target Neuron Types (MSN, FSN, TAN) ---
neurontype_map = containers.Map({1, 2, 3}, {'MSN', 'FSN', 'TAN'});

% Sort keys to ensure plots appear in logical order (1->2->3)
target_keys = sort(cell2mat(neurontype_map.keys));
num_types = numel(target_keys);

% Define colors for these specific types
type_colors = lines(num_types); 

% --- Figure 3: Spatial Tuning Across Epochs (Per Neuron Type) ---
figure('Name', 'Spatial Tuning by Neuron Type and Epoch', 'Position', [100, 100, 500 * num_types, 450]);
t_spatial_type = tiledlayout(1, num_types, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:num_types
    current_key = target_keys(i);
    current_name = neurontype_map(current_key);
    
    nexttile;
    hold on;
    legend_handles_spatial_type = [];
    
    % Filter data for this specific numeric key
    idx_type = (labels_valid.neurontype_labels == current_key);
    idx_type(isnan(labels_valid.neurontype_labels)) = false;

    n_units_this_type = sum(idx_type);
    
    if n_units_this_type > 0
        type_tensor = supermouse_combined_valid(idx_type, :, :); % [type_neurons x bins x trials]
        
        for i_epoch = 1:numel(epoch_trials)
            trials_for_epoch = epoch_trials{i_epoch};
            epoch_tensor = type_tensor(:, :, trials_for_epoch);
            
            % Average across trials
            activity_per_neuron_epoch = squeeze(mean(epoch_tensor, 3, 'omitnan'));
            
            % Mean and SEM across neurons
            mean_fr_space = mean(activity_per_neuron_epoch, 1, 'omitnan');
            sem_fr_space = std(activity_per_neuron_epoch, 0, 1, 'omitnan') / sqrt(n_units_this_type);
            
            % Plot
            h = shadedErrorBar(1:size(mean_fr_space, 2), mean_fr_space, sem_fr_space, ...
                'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
            if ~isempty(h) && isvalid(h.mainLine)
                 legend_handles_spatial_type(end+1) = h.mainLine;
            end
        end
        
        % Context Patches (Visual/Reward)
        yl = ylim;
        y_patch = [yl(1), yl(1), yl(2), yl(2)];
        bin_size = cfg.plot.zone_params.bin_size;
        
        patch([floor(cfg.plot.zone_params.visual_zones_au(1)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(2)/bin_size), ...
               floor(cfg.plot.zone_params.visual_zones_au(2)/bin_size), floor(cfg.plot.zone_params.visual_zones_au(1)/bin_size)], ...
               y_patch, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
           
        patch([floor(cfg.plot.zone_params.reward_zone_au(1)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(2)/bin_size), ...
               floor(cfg.plot.zone_params.reward_zone_au(2)/bin_size), floor(cfg.plot.zone_params.reward_zone_au(1)/bin_size)], ...
               y_patch, cfg.plot.colors.dls, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
               
        if ~isempty(legend_handles_spatial_type)
            uistack(legend_handles_spatial_type, 'top');
        end
        title(sprintf('%s (n=%d)', current_name, n_units_this_type));
        box on;
        xlim([0, size(supermouse_combined_valid, 2)]);
    else
        title(sprintf('%s (n=0)', current_name));
        text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
    end
end

title(t_spatial_type, 'Spatial Tuning by Neuron Type (MSN, FSN, TAN)', 'FontSize', 16);
xlabel(t_spatial_type, 'Spatial Bin', 'FontSize', 12);
ylabel(t_spatial_type, 'Mean Firing Rate (a.u.)', 'FontSize', 12);
if ~isempty(legend_handles_spatial_type)
    lg_spatial_type = legend(legend_handles_spatial_type, epoch_names, 'Location', 'best');
end


% --- Figure 4: Temporal Activity Evolution (By Neuron Type) ---
% REVISED: Only using spatial bins 15-25
bins_to_average = 15:25; 

figure('Name', 'Temporal Evolution by Neuron Type', 'Position', [250, 250, 900, 500]);
hold on;
legend_handles_temporal_type = [];
legend_names_temporal_type = {};

for i = 1:num_types
    current_key = target_keys(i);
    current_name = neurontype_map(current_key);
    
    idx_type = (labels_valid.neurontype_labels == current_key);
    idx_type(isnan(labels_valid.neurontype_labels)) = false;

    if sum(idx_type) > 0
        type_tensor = supermouse_combined_valid(idx_type, :, :); 
        
        % Ensure bin selection is valid for data dimensions
        valid_bins = bins_to_average(bins_to_average <= size(type_tensor, 2));
        
        % Average across SPECIFIC spatial bins (15-25)
        if ~isempty(valid_bins)
            % Select only the specific bins before averaging
            % [neurons x selected_bins x trials] -> [neurons x trials]
            activity_per_neuron_temporal = squeeze(mean(type_tensor(:, valid_bins, :), 2, 'omitnan'));
            
            % Mean/SEM across neurons
            mean_fr_time = mean(activity_per_neuron_temporal, 1, 'omitnan');
            sem_fr_time = std(activity_per_neuron_temporal, 0, 1, 'omitnan') / sqrt(sum(idx_type));
            
            this_color = type_colors(i, :);
            
            h = shadedErrorBar(1:size(mean_fr_time, 2), mean_fr_time, sem_fr_time, ...
                'lineprops', {'-','Color', this_color, 'LineWidth', 2});
                
            if ~isempty(h) && isvalid(h.mainLine)
                legend_handles_temporal_type(end+1) = h.mainLine;
                legend_names_temporal_type{end+1} = current_name;
            end
        else
            warning('Bins 15-25 are out of range for the data size.');
        end
    end
end

xline(0, 'k--', 'Naive', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
xline(10, 'k--', 'Intermediate', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
xline(20, 'k--', 'Expert', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');

title('Evolution of Mean Activity by Neuron Type (Spatial Bins 15-25)');
xlabel('Aligned Trial Number');
ylabel(sprintf('Mean Firing Rate (Bins %d-%d)', min(bins_to_average), max(bins_to_average)));
xlim([0, size(supermouse_combined_valid, 3)]);

if ~isempty(legend_handles_temporal_type)
    legend(legend_handles_temporal_type, legend_names_temporal_type, 'Location', 'northwest');
end
box on;
hold off;

fprintf('Displayed neuron type analysis plots (MSN, FSN, TAN).\n\n');

%% ================= Run TCA Analysis ====================================================
% best_mdl = [];
% best_n_factors = [];
tca_results = []; % To store detailed TCA outputs

if cfg.run_tca
    fprintf('--- Running TCA Analysis ---\n');
    % This now calls the wrapper which uses tca_with_bic_extended internally
    [best_mdl, best_n_factors, tca_results] = runTCAAnalysis(supermouse_combined_valid, cfg.tca);

    % Check if the analysis was successful and returned a model
    if isempty(best_mdl) || isempty(best_n_factors)
        warning('TCA analysis (via tca_with_bic_extended) did not yield a valid model/factor count. Skipping subsequent analyses dependent on TCA factors.');
        cfg.run_tca = false; % Turn off flag to prevent errors in plotting sections
    else
        % Report the selection made by the underlying function
        fprintf('==> tca_with_bic_extended selected %d factors based on its internal criteria (BIC + tolerance).\n', best_n_factors);

        % --- Plot Diagnostics (BIC, Recon Error, Similarity) ---
        plot_diagnostics = false; % Flag to check if there's anything to plot
        if isfield(tca_results, 'bic_values') && ~isempty(tca_results.bic_values) && ~all(isnan(tca_results.bic_values))
            plot_diagnostics = true; % BIC values exist
        end
        if isfield(tca_results, 'recon_errors') && ~isempty(tca_results.recon_errors) && ~all(isnan(tca_results.recon_errors))
            plot_diagnostics = true; % Recon errors exist
        end
        if isfield(tca_results, 'init_similarity_scores') && ~isempty(tca_results.init_similarity_scores) && ~all(isnan(tca_results.init_similarity_scores))
            plot_diagnostics = true; % Similarity scores exist
        end

        if plot_diagnostics
            figure;
            legend_items = {}; % Initialize cell array for legend entries
            x_axis_values = 1:cfg.tca.max_factors;

            % Plot BIC on the left axis
            if isfield(tca_results, 'bic_values') && ~isempty(tca_results.bic_values) && ~all(isnan(tca_results.bic_values))
                plot(x_axis_values, tca_results.bic_values, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
                legend_items{end+1} = 'BIC';
                hold on; % Hold on for other plots
            end

            right_axis_plotted = false; % Keep track if right axis is used

            % Plot Normalized Reconstruction Error on right axis
            if isfield(tca_results,'recon_errors') && ~isempty(tca_results.recon_errors) && ~all(isnan(tca_results.recon_errors))
                yyaxis right;
                valid_recon = tca_results.recon_errors(~isnan(tca_results.recon_errors));
                if ~isempty(valid_recon)
                    norm_recon_err = tca_results.recon_errors ./ max(valid_recon);
                    plot(x_axis_values, norm_recon_err, 'r-*', 'LineWidth', 1);
                    ylabel('Normalized Recon Error');
                    ax = gca; ax.YAxis(2).Color = 'r'; % Color right axis red
                    legend_items{end+1} = 'Norm Recon Error';
                    right_axis_plotted = true;
                else
                    yyaxis left; % Switch back if no valid recon error plotted
                end
            end

            % Plot Initialization Similarity Score (also on right axis if used)
            if isfield(tca_results,'init_similarity_scores') && ~isempty(tca_results.init_similarity_scores) && ~all(isnan(tca_results.init_similarity_scores))
                yyaxis right; % Ensure we are on the right axis
                plot(x_axis_values, tca_results.init_similarity_scores, 'g-s', 'LineWidth', 1, 'MarkerFaceColor', 'g');
                legend_items{end+1} = 'Avg Similarity';
                if right_axis_plotted % Already plotted recon error?
                    ylabel('Norm Recon Error / Avg Similarity'); % Combined label
                else % Only similarity is on right axis
                    ylabel('Avg Factor Similarity');
                    ax = gca; ax.YAxis(2).Color = 'g'; % Color right axis green
                end
                right_axis_plotted = true; % Mark right axis as used
            end

            % Switch back to left axis for final annotations
            yyaxis left;
            xlabel('Number of Factors');
            ylabel('BIC'); % Primary Y-label remains BIC
            title('TCA Factor Selection Diagnostics');



            if ~isempty(legend_items)
                legend(legend_items, 'Location', 'best');
            end

            % Add vertical line for selected factors
            if ~isempty(best_n_factors)
                xline(best_n_factors, 'k--', sprintf('Selected = %d', best_n_factors), 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
            end

            box off;
            grid on;
            xlim([1, cfg.tca.max_factors + 0.5]); % Adjust xlim slightly
            hold off;
        else
            fprintf('  No valid diagnostic data (BIC, Recon Error, Similarity) found in tca_results. Skipping diagnostic plot.\n');
        end
    end
else
    fprintf('--- Skipping TCA Analysis (cfg.run_tca is false) ---\n');
end

%% ================= Override selection if needed ========================================
best_n_factors = 5;
best_mdl = tca_results.all_best_models{best_n_factors};


figure;
legend_items = {}; % Initialize cell array for legend entries
x_axis_values = 1:cfg.tca.max_factors;

% Plot BIC on the left axis
if isfield(tca_results, 'bic_values') && ~isempty(tca_results.bic_values) && ~all(isnan(tca_results.bic_values))
    plot(x_axis_values, tca_results.bic_values, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
    legend_items{end+1} = 'BIC';
    hold on; % Hold on for other plots
end

right_axis_plotted = false; % Keep track if right axis is used

% Plot Normalized Reconstruction Error on right axis
if isfield(tca_results,'recon_errors') && ~isempty(tca_results.recon_errors) && ~all(isnan(tca_results.recon_errors))
    yyaxis right;
    valid_recon = tca_results.recon_errors(~isnan(tca_results.recon_errors));
    if ~isempty(valid_recon)
        norm_recon_err = tca_results.recon_errors ./ max(valid_recon);
        plot(x_axis_values, norm_recon_err, 'r-*', 'LineWidth', 1);
        ylabel('Normalized Recon Error');
        ax = gca; ax.YAxis(2).Color = 'r'; % Color right axis red
        legend_items{end+1} = 'Norm Recon Error';
        right_axis_plotted = true;
    else
        yyaxis left; % Switch back if no valid recon error plotted
    end
end

% Plot Initialization Similarity Score (also on right axis if used)
if isfield(tca_results,'init_similarity_scores') && ~isempty(tca_results.init_similarity_scores) && ~all(isnan(tca_results.init_similarity_scores))
    yyaxis right; % Ensure we are on the right axis
    plot(x_axis_values, tca_results.init_similarity_scores, 'g-s', 'LineWidth', 1, 'MarkerFaceColor', 'g');
    legend_items{end+1} = 'Avg Similarity';
    if right_axis_plotted % Already plotted recon error?
        ylabel('Norm Recon Error / Avg Similarity'); % Combined label
    else % Only similarity is on right axis
        ylabel('Avg Factor Similarity');
        ax = gca; ax.YAxis(2).Color = 'g'; % Color right axis green
    end
    right_axis_plotted = true; % Mark right axis as used
end

% Switch back to left axis for final annotations
yyaxis left;
xlabel('Number of Factors');
ylabel('BIC'); % Primary Y-label remains BIC
title('TCA Factor Selection Diagnostics');



if ~isempty(legend_items)
    legend(legend_items, 'Location', 'best');
end

% Add vertical line for selected factors
if ~isempty(best_n_factors)
    xline(best_n_factors, 'k--', sprintf('Selected = %d', best_n_factors), 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal');
end

box off;
grid on;
xlim([1, cfg.tca.max_factors + 0.5]); % Adjust xlim slightly
hold off;

%% ================= Visualization and Further Analysis ==================================
fprintf('--- Generating Plots and Further Analyses ---\n');

% --- Plot Aligned Behavior (Task Only) ---
if cfg.plot.aligned_behavior && ~isempty(aligned_lick_errors)
    plotAlignedBehavior(aligned_lick_errors, cfg);
elseif cfg.plot.aligned_behavior
    fprintf('Skipping aligned behavior plot: No valid task data/lick errors available.\n');
end

% --- Plots Requiring TCA Results ---
if cfg.run_tca && ~isempty(best_mdl)
    nFactors = size(best_mdl.U{1}, 2); % Use actual factors from model

    % --- Spatial Factors ---
    if cfg.plot.spatial_factors
        plotSpatialFactors(best_mdl, cfg.plot.zone_params, nFactors);
    end

    % --- Neuron Factors by Mouse ---
    if cfg.plot.neuron_factors_by_mouse
        plotNeuronFactors(best_mdl, labels_valid.mouse_labels, 'mouse', nFactors, tensor_info.n_animals_total);
    end

    % --- Neuron Factors by Area (Raw & Downsampled) ---
    if cfg.plot.neuron_factors_by_area || cfg.plot.neuron_factors_by_area_downsampled
        plotNeuronFactorsByArea(best_mdl, labels_valid.area_labels, cfg, nFactors, false);
    end

    % --- Trial Factors ---
    if cfg.plot.trial_factors
        plotTrialFactors(best_mdl, cfg, nFactors);
    end

    % --- Trial Factors by Epoch ---
    if cfg.plot.trial_factors_by_epoch
        plotTrialFactorsByEpoch(best_mdl, cfg, nFactors);
    end

    % % --- Trial Factors vs Aligned Behavior ---
    % if cfg.plot.trial_factors_vs_behavior && ~isempty(aligned_lick_errors)
    %     plotTrialFactorVsBehaviorCorr(best_mdl, aligned_lick_errors, nFactors);
    % elseif cfg.plot.trial_factors_vs_behavior
    %     fprintf('Skipping trial factor vs behavior correlation plot: Aligned lick errors not available.\n');
    % end

    % --- Trial Factors vs Within-Epoch Behavior ---
    if cfg.plot.trial_factors_vs_epoch_behavior && ~isempty(aligned_lick_errors)
        plotTrialFactorVsEpochBehaviorCorr(best_mdl, aligned_lick_errors, cfg, nFactors);
    elseif cfg.plot.trial_factors_vs_epoch_behavior
        fprintf('Skipping trial factor vs within-epoch behavior correlation plot: Aligned lick errors not available.\n');
    end

    % % --- Factor Contribution Heatmaps ---
    if cfg.plot.factor_heatmaps_by_mouse_area
        plotFactorHeatmapsByMouseArea(best_mdl, labels_valid.mouse_labels, labels_valid.area_labels, nFactors, tensor_info.n_animals_total);
    end

    % --- t-SNE Embedding ---
    if cfg.plot.tsne_embedding
        plotTSNEembedding(best_mdl, labels_valid, cfg);
    end

    % --- Trial Factor Scatter 3D ---
    if cfg.plot.trial_factor_scatter3d && nFactors >= 3
        plotTrialFactorScatter3D(best_mdl, cfg, min(nFactors, 3)); % Plot first 3 factors
    elseif cfg.plot.trial_factor_scatter3d
        fprintf('Skipping trial factor 3D scatter: Fewer than 3 factors found.\n');
    end

else
    fprintf('Skipping plots dependent on TCA results.\n');
end

fprintf('--- Analysis Pipeline Finished ---\n');

