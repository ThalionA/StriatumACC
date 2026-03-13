%% ================= Comprehensive Spatiotemporal Visualization (Pre-Subsampling) =================
fprintf('--- Generating Comprehensive Spatiotemporal Plots (Pooled vs Hierarchical) ---\n');

% --- 1. Prepare Full Clean Data (Pre-subsampling) ---
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_idx = ~neuron_has_nan;

tensor_full_raw = supermouse_tensor_raw(valid_idx, :, :);
lbls_full.mouse  = combined_labels.mouse_labels_all(valid_idx);
lbls_full.group  = combined_labels.group_labels_all(valid_idx); % 1=Task, 2=Control
lbls_full.area   = combined_labels.area_labels_all(valid_idx);

% --- 2. Calculate Z-Scored Tensor ---
fprintf('Calculating Z-scored firing rates for all units...\n');
tensor_full_z = nan(size(tensor_full_raw));
for i = 1:size(tensor_full_raw, 1)
    unit_data = squeeze(tensor_full_raw(i, :, :));
    mu = mean(unit_data(:), 'omitnan');
    sig = std(unit_data(:), 'omitnan');
    if sig > 0
        tensor_full_z(i, :, :) = (tensor_full_raw(i, :, :) - mu) / sig;
    else
        tensor_full_z(i, :, :) = 0; % Handle silent neurons safely
    end
end

% --- 3. Plotting Configuration ---
epoch_trials = {1:3, 4:10, 11:20, 21:30};
epoch_names  = {'Trials 1-3', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
color_t1_3   = min(cfg.plot.colors.epoch_early + 0.3, 1);
epoch_colors = {color_t1_3, cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

areas_in_data = unique(lbls_full.area);
num_areas = numel(areas_in_data);
bin_size = cfg.plot.zone_params.bin_size;
v_zone = cfg.plot.zone_params.visual_zones_au / bin_size;
r_zone = cfg.plot.zone_params.reward_zone_au / bin_size;

% Loop structures to iterate through all requested combinations
datasets = {1, 'Task'; 2, 'Control'};
metrics  = {tensor_full_raw, 'Raw FR'; tensor_full_z, 'Z-Scored'};
avg_methods = {'Pooled', 'Hierarchical'};

for ds_idx = 1:size(datasets, 1)
    group_id = datasets{ds_idx, 1};
    ds_name  = datasets{ds_idx, 2};
    
    % Check if this dataset has any data
    group_mask = (lbls_full.group == group_id);
    if sum(group_mask) == 0
        fprintf('Skipping %s dataset (no data found).\n', ds_name);
        continue;
    end
    
    for met_idx = 1:size(metrics, 1)
        current_tensor = metrics{met_idx, 1};
        met_name       = metrics{met_idx, 2};
        
        for avg_idx = 1:length(avg_methods)
            avg_mode = avg_methods{avg_idx};
            
            fig_prefix = sprintf('[%s] %s - %s', ds_name, met_name, avg_mode);
            fprintf('Generating plots for: %s\n', fig_prefix);
            
            % -----------------------------------------------------------------
            % FIGURE A: SPATIAL TUNING BY AREA AND EPOCH
            % -----------------------------------------------------------------
            fig_spatial = figure('Name', sprintf('%s: Spatial', fig_prefix), 'Position', [100, 100, 500 * num_areas, 450]);
            t_spatial = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
            
            for i_area = 1:num_areas
                current_area = areas_in_data{i_area};
                nexttile; hold on;
                legend_handles_spatial = [];
                
                idx_target = group_mask & strcmp(lbls_full.area, current_area);
                target_tensor = current_tensor(idx_target, :, :);
                target_mice = lbls_full.mouse(idx_target);
                unique_target_mice = unique(target_mice);
                
                if sum(idx_target) == 0; title(sprintf('%s (N=0)', current_area)); continue; end
                
                for i_epoch = 1:numel(epoch_trials)
                    trs = epoch_trials{i_epoch};
                    epoch_data = target_tensor(:, :, trs);
                    
                    if strcmp(avg_mode, 'Pooled')
                        activity_per_neuron = squeeze(mean(epoch_data, 3, 'omitnan'));
                        mean_space = mean(activity_per_neuron, 1, 'omitnan');
                        sem_space  = std(activity_per_neuron, 0, 1, 'omitnan') / sqrt(size(activity_per_neuron, 1));
                    else
                        n_m = length(unique_target_mice);
                        mouse_means = nan(n_m, size(epoch_data, 2));
                        for m = 1:n_m
                            m_idx = (target_mice == unique_target_mice(m));
                            m_data = epoch_data(m_idx, :, :);
                            mouse_means(m, :) = mean(mean(m_data, 3, 'omitnan'), 1, 'omitnan');
                        end
                        mean_space = mean(mouse_means, 1, 'omitnan');
                        sem_space  = std(mouse_means, 0, 1, 'omitnan') / sqrt(n_m);
                    end
                    
                    h = shadedErrorBar(1:size(mean_space, 2), mean_space, sem_space, ...
                        'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
                    legend_handles_spatial(end+1) = h.mainLine;
                end
                
                yl = ylim; y_p = [yl(1), yl(1), yl(2), yl(2)];
                patch([v_zone(1), v_zone(2), v_zone(2), v_zone(1)], y_p, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                patch([r_zone(1), r_zone(2), r_zone(2), r_zone(1)], y_p, cfg.plot.colors.dls, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                uistack(legend_handles_spatial, 'top');
                
                if strcmp(avg_mode, 'Pooled')
                    title(sprintf('%s (n=%d units)', current_area, sum(idx_target)));
                else
                    title(sprintf('%s (N=%d mice)', current_area, length(unique_target_mice)));
                end
                box on; xlim([0, size(current_tensor, 2)]);
            end
            title(t_spatial, sprintf('%s: Spatial Tuning Evolution', fig_prefix), 'FontSize', 16);
            xlabel(t_spatial, 'Spatial Bin', 'FontSize', 12);
            ylabel(t_spatial, met_name, 'FontSize', 12);
            lg_spatial = legend(legend_handles_spatial, epoch_names, 'Location', 'best');
            
            % Export Figure A
            % Standardise formatting for filesystem logic (removing spaces, brackets, etc)
            clean_name_spatial = regexprep(sprintf('%s_Spatial', fig_prefix), '[\[\]\s:]', '_'); 
            save_to_svg(clean_name_spatial);

            % -----------------------------------------------------------------
            % FIGURE B: TEMPORAL EVOLUTION BY AREA
            % -----------------------------------------------------------------
            fig_temporal = figure('Name', sprintf('%s: Temporal', fig_prefix), 'Position', [200, 200, 900, 500]);
            hold on;
            legend_handles_temporal = [];
            
            for i_area = 1:num_areas
                current_area = areas_in_data{i_area};
                idx_target = group_mask & strcmp(lbls_full.area, current_area);
                target_tensor = current_tensor(idx_target, :, :);
                target_mice = lbls_full.mouse(idx_target);
                unique_target_mice = unique(target_mice);
                
                if sum(idx_target) == 0; continue; end
                
                if strcmp(avg_mode, 'Pooled')
                    activity_per_neuron_temp = squeeze(mean(target_tensor, 2, 'omitnan'));
                    mean_time = mean(activity_per_neuron_temp, 1, 'omitnan');
                    sem_time  = std(activity_per_neuron_temp, 0, 1, 'omitnan') / sqrt(size(activity_per_neuron_temp, 1));
                else
                    n_m = length(unique_target_mice);
                    mouse_means_temp = nan(n_m, size(target_tensor, 3));
                    for m = 1:n_m
                        m_idx = (target_mice == unique_target_mice(m));
                        m_data = target_tensor(m_idx, :, :);
                        mouse_means_temp(m, :) = mean(mean(m_data, 2, 'omitnan'), 1, 'omitnan');
                    end
                    mean_time = mean(mouse_means_temp, 1, 'omitnan');
                    sem_time  = std(mouse_means_temp, 0, 1, 'omitnan') / sqrt(n_m);
                end
                
                area_color = cfg.plot.colors.area_map(current_area);
                h = shadedErrorBar(1:size(mean_time, 2), mean_time, sem_time, ...
                    'lineprops', {'-','Color', area_color, 'LineWidth', 2});
                legend_handles_temporal(end+1) = h.mainLine;
            end
            
            xline(0, 'k--', epoch_names{1}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
            xline(3, 'k--', epoch_names{2}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
            xline(10, 'k--', epoch_names{3}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
            xline(20, 'k--', epoch_names{4}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
            
            title(sprintf('%s: Temporal Evolution', fig_prefix));
            xlabel('Aligned Trial Number');
            ylabel(sprintf('Mean %s (Averaged over Space)', met_name));
            xlim([0, size(current_tensor, 3)]);
            legend(legend_handles_temporal, areas_in_data, 'Location', 'northwest');
            box on; hold off;

            % Export Figure B
            clean_name_temporal = regexprep(sprintf('%s_Temporal', fig_prefix), '[\[\]\s:]', '_'); 
            save_to_svg(clean_name_temporal);
            
        end % End loop averaging methods
    end % End loop metrics
end % End loop datasets
fprintf('--- Comprehensive Spatiotemporal Plots Complete ---\n\n');

%% ================= Comprehensive Spatiotemporal Activity by Neuron Type =================
fprintf('--- Generating Comprehensive Plots by NEURON TYPE (Task & Control) ---\n');

% --- 1. Prepare Data Labels ---
valid_group = combined_labels.group_labels_all(valid_idx);
valid_mouse = combined_labels.mouse_labels_all(valid_idx);
valid_ntype = combined_labels.neurontype_labels_all(valid_idx);

% Define groups to loop over
group_ids = [1, 2];
group_names = {'Task', 'Control'};

% --- 2. Plotting Configuration ---
epoch_trials = {1:3, 4:10, 11:20, 21:30};
epoch_names  = {'Trials 1-3', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
color_t1_3   = min(cfg.plot.colors.epoch_early + 0.3, 1);
epoch_colors = {color_t1_3, cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

neurontype_map = containers.Map({1, 2, 3}, {'MSN', 'FSN', 'TAN'});
target_keys = sort(cell2mat(neurontype_map.keys));
num_types = numel(target_keys);
type_colors = lines(num_types); 

bin_size = cfg.plot.zone_params.bin_size;
v_zone = cfg.plot.zone_params.visual_zones_au / bin_size;
r_zone = cfg.plot.zone_params.reward_zone_au / bin_size;

avg_methods = {'Pooled', 'Hierarchical'};

% --- 3. Loop over Groups (Task and Control) ---
for g_idx = 1:length(group_ids)
    current_group_id = group_ids(g_idx);
    current_group_name = group_names{g_idx};
    
    fprintf('\n>>> Processing Group: %s <<<\n', current_group_name);
    
    idx_group = (valid_group == current_group_id);
    if sum(idx_group) == 0
        fprintf('No data found for %s group. Skipping...\n', current_group_name);
        continue;
    end
    
    group_tensor_raw = tensor_full_raw(idx_group, :, :);
    group_tensor_z   = tensor_full_z(idx_group, :, :);
    group_mouse      = valid_mouse(idx_group);
    group_ntype      = valid_ntype(idx_group);
    
    metrics  = {group_tensor_raw, 'Raw FR'; group_tensor_z, 'Z-Scored'};
    
    for met_idx = 1:size(metrics, 1)
        current_tensor = metrics{met_idx, 1};
        met_name       = metrics{met_idx, 2};
        
        for avg_idx = 1:length(avg_methods)
            avg_mode = avg_methods{avg_idx};
            
            fig_prefix = sprintf('[%s] Neuron Types: %s - %s', current_group_name, met_name, avg_mode);
            fprintf('Generating plots for: %s\n', fig_prefix);
            
            % -----------------------------------------------------------------
            % FIGURE A: SPATIAL TUNING BY NEURON TYPE AND EPOCH
            % -----------------------------------------------------------------
            fig_spatial_type = figure('Name', sprintf('%s: Spatial', fig_prefix), 'Position', [100, 100, 500 * num_types, 450]);
            t_spatial_type = tiledlayout(1, num_types, 'TileSpacing', 'compact', 'Padding', 'compact');
            
            for i = 1:num_types
                current_key = target_keys(i);
                current_name = neurontype_map(current_key);
                
                nexttile; hold on;
                legend_handles_spatial_type = [];
                
                idx_type = (group_ntype == current_key);
                idx_type(isnan(idx_type)) = false;
                
                n_units_this_type = sum(idx_type);
                
                if n_units_this_type > 0
                    target_tensor = current_tensor(idx_type, :, :);
                    target_mice = group_mouse(idx_type);
                    unique_target_mice = unique(target_mice);
                    
                    for i_epoch = 1:numel(epoch_trials)
                        trs = epoch_trials{i_epoch};
                        epoch_data = target_tensor(:, :, trs);
                        
                        if strcmp(avg_mode, 'Pooled')
                            activity_per_neuron = squeeze(mean(epoch_data, 3, 'omitnan'));
                            mean_space = mean(activity_per_neuron, 1, 'omitnan');
                            sem_space  = std(activity_per_neuron, 0, 1, 'omitnan') / sqrt(n_units_this_type);
                        else
                            n_m = length(unique_target_mice);
                            mouse_means = nan(n_m, size(epoch_data, 2));
                            for m = 1:n_m
                                m_idx = (target_mice == unique_target_mice(m));
                                m_data = epoch_data(m_idx, :, :);
                                mouse_means(m, :) = mean(mean(m_data, 3, 'omitnan'), 1, 'omitnan');
                            end
                            mean_space = mean(mouse_means, 1, 'omitnan');
                            sem_space  = std(mouse_means, 0, 1, 'omitnan') / sqrt(n_m);
                        end
                        
                        h = shadedErrorBar(1:size(mean_space, 2), mean_space, sem_space, ...
                            'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
                        if ~isempty(h) && isvalid(h.mainLine)
                             legend_handles_spatial_type(end+1) = h.mainLine;
                        end
                    end
                    
                    yl = ylim; y_p = [yl(1), yl(1), yl(2), yl(2)];
                    patch([v_zone(1), v_zone(2), v_zone(2), v_zone(1)], y_p, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                    patch([r_zone(1), r_zone(2), r_zone(2), r_zone(1)], y_p, cfg.plot.colors.dls, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                    if ~isempty(legend_handles_spatial_type); uistack(legend_handles_spatial_type, 'top'); end
                    
                    if strcmp(avg_mode, 'Pooled')
                        title(sprintf('%s (n=%d units)', current_name, n_units_this_type));
                    else
                        title(sprintf('%s (N=%d mice)', current_name, length(unique_target_mice)));
                    end
                    box on; xlim([0, size(current_tensor, 2)]);
                else
                    title(sprintf('%s (n=0)', current_name));
                    text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
                end
            end
            title(t_spatial_type, sprintf('%s: Spatial Tuning Evolution', fig_prefix), 'FontSize', 16);
            xlabel(t_spatial_type, 'Spatial Bin', 'FontSize', 12);
            ylabel(t_spatial_type, met_name, 'FontSize', 12);
            if ~isempty(legend_handles_spatial_type)
                lg_spatial_type = legend(legend_handles_spatial_type, epoch_names, 'Location', 'best');
            end

            % Export Figure A
            clean_name_spatial_type = regexprep(sprintf('%s_Spatial_Type', fig_prefix), '[\[\]\s:]', '_'); 
            save_to_svg(clean_name_spatial_type);
            
            % -----------------------------------------------------------------
            % FIGURE B: TEMPORAL EVOLUTION BY NEURON TYPE
            % -----------------------------------------------------------------
            fig_temporal_type = figure('Name', sprintf('%s: Temporal', fig_prefix), 'Position', [250, 250, 900, 500]);
            hold on;
            legend_handles_temporal_type = [];
            legend_names_temporal_type = {};
            
            for i = 1:num_types
                current_key = target_keys(i);
                current_name = neurontype_map(current_key);
                
                idx_type = (group_ntype == current_key);
                idx_type(isnan(idx_type)) = false;
                
                if sum(idx_type) > 0
                    target_tensor = current_tensor(idx_type, :, :);
                    target_mice = group_mouse(idx_type);
                    unique_target_mice = unique(target_mice);
                    
                    if strcmp(avg_mode, 'Pooled')
                        activity_per_neuron_temp = squeeze(mean(target_tensor, 2, 'omitnan'));
                        mean_time = mean(activity_per_neuron_temp, 1, 'omitnan');
                        sem_time  = std(activity_per_neuron_temp, 0, 1, 'omitnan') / sqrt(sum(idx_type));
                    else
                        n_m = length(unique_target_mice);
                        mouse_means_temp = nan(n_m, size(target_tensor, 3));
                        for m = 1:n_m
                            m_idx = (target_mice == unique_target_mice(m));
                            m_data = target_tensor(m_idx, :, :);
                            mouse_means_temp(m, :) = mean(mean(m_data, 2, 'omitnan'), 1, 'omitnan');
                        end
                        mean_time = mean(mouse_means_temp, 1, 'omitnan');
                        sem_time  = std(mouse_means_temp, 0, 1, 'omitnan') / sqrt(n_m);
                    end
                    
                    this_color = type_colors(i, :);
                    h = shadedErrorBar(1:size(mean_time, 2), mean_time, sem_time, ...
                        'lineprops', {'-','Color', this_color, 'LineWidth', 2});
                    if ~isempty(h) && isvalid(h.mainLine)
                        legend_handles_temporal_type(end+1) = h.mainLine;
                        legend_names_temporal_type{end+1} = current_name;
                    end
                end
            end
            
            xline(0, 'k--', epoch_names{1}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
            xline(3, 'k--', epoch_names{2}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
            xline(10, 'k--', epoch_names{3}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
            xline(20, 'k--', epoch_names{4}, 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
            
            title(sprintf('%s: Temporal Evolution', fig_prefix));
            xlabel('Aligned Trial Number');
            ylabel(sprintf('Mean %s (Averaged over Space)', met_name));
            xlim([0, size(current_tensor, 3)]);
            if ~isempty(legend_handles_temporal_type)
                legend(legend_handles_temporal_type, legend_names_temporal_type, 'Location', 'northwest');
            end
            box on; hold off;

            % Export Figure B
            clean_name_temporal_type = regexprep(sprintf('%s_Temporal_Type', fig_prefix), '[\[\]\s:]', '_'); 
            save_to_svg(clean_name_temporal_type);
            
        end % End loop averaging methods
    end % End loop metrics
end % End loop groups (Task/Control)
fprintf('--- Comprehensive Neuron Type Plots Complete ---\n\n');

%% ================= Comprehensive Spatiotemporal Visualization (Increase/Decrease/Maintain) =================
fprintf('--- Generating Comprehensive Spatiotemporal Plots (Pooled vs Hierarchical) ---\n');

% --- 1. Prepare Full Clean Data (Pre-subsampling) ---
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_idx = ~neuron_has_nan;

tensor_full_raw = supermouse_tensor_raw(valid_idx, :, :);
lbls_full.mouse  = combined_labels.mouse_labels_all(valid_idx);
lbls_full.group  = combined_labels.group_labels_all(valid_idx); % 1=Task, 2=Control
lbls_full.area   = combined_labels.area_labels_all(valid_idx);

% --- 2. Calculate Z-Scored Tensor ---
fprintf('Calculating Z-scored firing rates for all units...\n');
tensor_full_z = nan(size(tensor_full_raw));
for i = 1:size(tensor_full_raw, 1)
    unit_data = squeeze(tensor_full_raw(i, :, :));
    mu = mean(unit_data(:), 'omitnan');
    sig = std(unit_data(:), 'omitnan');
    if sig > 0
        tensor_full_z(i, :, :) = (tensor_full_raw(i, :, :) - mu) / sig;
    else
        tensor_full_z(i, :, :) = 0; % Handle silent neurons safely
    end
end

% --- 2.5 Classify Neurons by Learning Modulation (Naive vs Expert) ---
fprintf('Classifying neurons into Increasers, Decreasers, and Maintainers...\n');
epoch_trials = {1:3, 4:10, 11:20, 21:30}; % Ensure these match your actual trial indices

act_naive  = mean(tensor_full_z(:, :, epoch_trials{1}), [2, 3], 'omitnan');
act_expert = mean(tensor_full_z(:, :, epoch_trials{4}), [2, 3], 'omitnan');
delta_act  = act_expert - act_naive;

modulation_class = zeros(size(delta_act)); % 1=Inc, 2=Dec, 3=Main
z_thresh = 0.25; % Threshold in Z-score units to define a "meaningful" change

modulation_class(delta_act > z_thresh) = 1;
modulation_class(delta_act < -z_thresh) = 2;
modulation_class(abs(delta_act) <= z_thresh) = 3;

mod_labels = {'Increasers', 'Decreasers', 'Maintainers'};
num_mods = length(mod_labels);

% --- 3. Plotting Configuration ---
epoch_names  = {'Trials 1-3', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
color_t1_3   = min(cfg.plot.colors.epoch_early + 0.3, 1);
epoch_colors = {color_t1_3, cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

areas_in_data = unique(lbls_full.area);
num_areas = numel(areas_in_data);
bin_size = cfg.plot.zone_params.bin_size;
v_zone = cfg.plot.zone_params.visual_zones_au / bin_size;
r_zone = cfg.plot.zone_params.reward_zone_au / bin_size;

% Loop structures
datasets = {1, 'Task'; 2, 'Control'};
metrics  = {tensor_full_raw, 'Raw FR'; tensor_full_z, 'Z-Scored'};
avg_methods = {'Pooled', 'Hierarchical'};

for ds_idx = 1:size(datasets, 1)
    group_id = datasets{ds_idx, 1};
    ds_name  = datasets{ds_idx, 2};
    
    group_mask = (lbls_full.group == group_id);
    if sum(group_mask) == 0
        fprintf('Skipping %s dataset (no data found).\n', ds_name);
        continue;
    end
    
    for met_idx = 1:size(metrics, 1)
        current_tensor = metrics{met_idx, 1};
        met_name       = metrics{met_idx, 2};
        
        for avg_idx = 1:length(avg_methods)
            avg_mode = avg_methods{avg_idx};
            
            fig_prefix = sprintf('[%s] %s - %s', ds_name, met_name, avg_mode);
            fprintf('Generating plots for: %s\n', fig_prefix);
            
            % -----------------------------------------------------------------
            % FIGURE A: SPATIAL TUNING BY AREA AND EPOCH (Split by Modulation)
            % -----------------------------------------------------------------
            fig_spatial = figure('Name', sprintf('%s: Spatial', fig_prefix), 'Position', [100, 100, 500 * num_areas, 300 * num_mods]);
            t_spatial = tiledlayout(num_mods, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
            
            for i_mod = 1:num_mods
                for i_area = 1:num_areas
                    current_area = areas_in_data{i_area};
                    nexttile; hold on;
                    legend_handles_spatial = [];
                    
                    % Isolate Area AND Modulation Class
                    idx_target = group_mask & strcmp(lbls_full.area, current_area) & (modulation_class == i_mod);
                    target_tensor = current_tensor(idx_target, :, :);
                    target_mice = lbls_full.mouse(idx_target);
                    unique_target_mice = unique(target_mice);
                    
                    if sum(idx_target) == 0
                        title(sprintf('%s - %s (n=0)', current_area, mod_labels{i_mod})); 
                        axis off;
                        continue; 
                    end
                    
                    for i_epoch = 1:numel(epoch_trials)
                        trs = epoch_trials{i_epoch};
                        epoch_data = target_tensor(:, :, trs);
                        
                        if strcmp(avg_mode, 'Pooled')
                            activity_per_neuron = squeeze(mean(epoch_data, 3, 'omitnan'));
                            mean_space = mean(activity_per_neuron, 1, 'omitnan');
                            sem_space  = std(activity_per_neuron, 0, 1, 'omitnan') / sqrt(size(activity_per_neuron, 1));
                        else
                            n_m = length(unique_target_mice);
                            mouse_means = nan(n_m, size(epoch_data, 2));
                            for m = 1:n_m
                                m_idx = (target_mice == unique_target_mice(m));
                                m_data = epoch_data(m_idx, :, :);
                                mouse_means(m, :) = mean(mean(m_data, 3, 'omitnan'), 1, 'omitnan');
                            end
                            mean_space = mean(mouse_means, 1, 'omitnan');
                            sem_space  = std(mouse_means, 0, 1, 'omitnan') / sqrt(n_m);
                        end
                        
                        h = shadedErrorBar(1:size(mean_space, 2), mean_space, sem_space, ...
                            'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
                        legend_handles_spatial(end+1) = h.mainLine;
                    end
                    
                    yl = ylim; y_p = [yl(1), yl(1), yl(2), yl(2)];
                    patch([v_zone(1), v_zone(2), v_zone(2), v_zone(1)], y_p, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                    patch([r_zone(1), r_zone(2), r_zone(2), r_zone(1)], y_p, cfg.plot.colors.dls, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                    uistack(legend_handles_spatial, 'top');
                    
                    if strcmp(avg_mode, 'Pooled')
                        title(sprintf('%s - %s (n=%d)', current_area, mod_labels{i_mod}, sum(idx_target)));
                    else
                        title(sprintf('%s - %s (N=%d mice)', current_area, mod_labels{i_mod}, length(unique_target_mice)));
                    end
                    box on; xlim([0, size(current_tensor, 2)]);
                    
                    if i_mod == num_mods && i_area == num_areas
                        legend(legend_handles_spatial, epoch_names, 'Location', 'best');
                    end
                end
            end
            title(t_spatial, sprintf('%s: Spatial Tuning Evolution (Split by Modulation)', fig_prefix), 'FontSize', 16);
            xlabel(t_spatial, 'Spatial Bin', 'FontSize', 12);
            ylabel(t_spatial, met_name, 'FontSize', 12);
            
            clean_name_spatial = regexprep(sprintf('%s_Spatial_Subpops', fig_prefix), '[\[\]\s:]', '_'); 
            save_to_svg(clean_name_spatial);
            
            % -----------------------------------------------------------------
            % FIGURE B: TEMPORAL EVOLUTION BY AREA (Split by Modulation)
            % -----------------------------------------------------------------
            fig_temporal = figure('Name', sprintf('%s: Temporal', fig_prefix), 'Position', [200, 200, 1400, 450]);
            t_temporal = tiledlayout(1, num_mods, 'TileSpacing', 'compact', 'Padding', 'compact');
            
            for i_mod = 1:num_mods
                nexttile; hold on;
                legend_handles_temporal = [];
                valid_areas = {};
                
                for i_area = 1:num_areas
                    current_area = areas_in_data{i_area};
                    idx_target = group_mask & strcmp(lbls_full.area, current_area) & (modulation_class == i_mod);
                    target_tensor = current_tensor(idx_target, :, :);
                    target_mice = lbls_full.mouse(idx_target);
                    unique_target_mice = unique(target_mice);
                    
                    if sum(idx_target) == 0; continue; end
                    valid_areas{end+1} = current_area;
                    
                    if strcmp(avg_mode, 'Pooled')
                        activity_per_neuron_temp = squeeze(mean(target_tensor, 2, 'omitnan'));
                        mean_time = mean(activity_per_neuron_temp, 1, 'omitnan');
                        sem_time  = std(activity_per_neuron_temp, 0, 1, 'omitnan') / sqrt(size(activity_per_neuron_temp, 1));
                    else
                        n_m = length(unique_target_mice);
                        mouse_means_temp = nan(n_m, size(target_tensor, 3));
                        for m = 1:n_m
                            m_idx = (target_mice == unique_target_mice(m));
                            m_data = target_tensor(m_idx, :, :);
                            mouse_means_temp(m, :) = mean(mean(m_data, 2, 'omitnan'), 1, 'omitnan');
                        end
                        mean_time = mean(mouse_means_temp, 1, 'omitnan');
                        sem_time  = std(mouse_means_temp, 0, 1, 'omitnan') / sqrt(n_m);
                    end
                    
                    area_color = cfg.plot.colors.area_map(current_area);
                    h = shadedErrorBar(1:size(mean_time, 2), mean_time, sem_time, ...
                        'lineprops', {'-','Color', area_color, 'LineWidth', 2});
                    legend_handles_temporal(end+1) = h.mainLine;
                end
                
                xline(0, 'k--', epoch_names{1}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(3, 'k--', epoch_names{2}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(10, 'k--', epoch_names{3}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(20, 'k--', epoch_names{4}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                
                title(sprintf('%s', mod_labels{i_mod}));
                xlabel('Aligned Trial Number');
                if i_mod == 1; ylabel(sprintf('Mean %s', met_name)); end
                xlim([0, size(current_tensor, 3)]);
                if ~isempty(legend_handles_temporal)
                    legend(legend_handles_temporal, valid_areas, 'Location', 'best');
                end
                box on; hold off;
            end
            title(t_temporal, sprintf('%s: Temporal Evolution by Subpopulation', fig_prefix), 'FontSize', 16);
            
            clean_name_temporal = regexprep(sprintf('%s_Temporal_Subpops', fig_prefix), '[\[\]\s:]', '_'); 
            save_to_svg(clean_name_temporal);
            
        end % End loop averaging methods
    end % End loop metrics
end % End loop datasets
fprintf('--- Comprehensive Spatiotemporal Plots Complete ---\n\n');

%% ================= Comprehensive Spatiotemporal Activity by Neuron Type =================
fprintf('--- Generating Comprehensive Plots by NEURON TYPE (Task & Control) ---\n');

% --- 1. Prepare Data Labels ---
valid_group = combined_labels.group_labels_all(valid_idx);
valid_mouse = combined_labels.mouse_labels_all(valid_idx);
valid_ntype = combined_labels.neurontype_labels_all(valid_idx);

group_ids = [1, 2];
group_names = {'Task', 'Control'};

% --- 2. Plotting Configuration ---
neurontype_map = containers.Map({1, 2, 3}, {'MSN', 'FSN', 'TAN'});
target_keys = sort(cell2mat(neurontype_map.keys));
num_types = numel(target_keys);
type_colors = lines(num_types); 

% --- 3. Loop over Groups (Task and Control) ---
for g_idx = 1:length(group_ids)
    current_group_id = group_ids(g_idx);
    current_group_name = group_names{g_idx};
    
    fprintf('\n>>> Processing Group: %s <<<\n', current_group_name);
    
    idx_group = (valid_group == current_group_id);
    if sum(idx_group) == 0; continue; end
    
    group_tensor_raw = tensor_full_raw(idx_group, :, :);
    group_tensor_z   = tensor_full_z(idx_group, :, :);
    group_mouse      = valid_mouse(idx_group);
    group_ntype      = valid_ntype(idx_group);
    group_mod_class  = modulation_class(idx_group); % Subpopulation alignment
    
    metrics  = {group_tensor_raw, 'Raw FR'; group_tensor_z, 'Z-Scored'};
    
    for met_idx = 1:size(metrics, 1)
        current_tensor = metrics{met_idx, 1};
        met_name       = metrics{met_idx, 2};
        
        for avg_idx = 1:length(avg_methods)
            avg_mode = avg_methods{avg_idx};
            fig_prefix = sprintf('[%s] Neuron Types: %s - %s', current_group_name, met_name, avg_mode);
            
            % -----------------------------------------------------------------
            % FIGURE A: SPATIAL TUNING BY NEURON TYPE (Split by Modulation)
            % -----------------------------------------------------------------
            fig_spatial_type = figure('Name', sprintf('%s: Spatial', fig_prefix), 'Position', [100, 100, 500 * num_types, 300 * num_mods]);
            t_spatial_type = tiledlayout(num_mods, num_types, 'TileSpacing', 'compact', 'Padding', 'compact');
            
            for i_mod = 1:num_mods
                for i = 1:num_types
                    current_key = target_keys(i);
                    current_name = neurontype_map(current_key);
                    
                    nexttile; hold on;
                    legend_handles_spatial_type = [];
                    
                    idx_type = (group_ntype == current_key) & (group_mod_class == i_mod);
                    idx_type(isnan(idx_type)) = false;
                    n_units_this_type = sum(idx_type);
                    
                    if n_units_this_type > 0
                        target_tensor = current_tensor(idx_type, :, :);
                        target_mice = group_mouse(idx_type);
                        unique_target_mice = unique(target_mice);
                        
                        for i_epoch = 1:numel(epoch_trials)
                            trs = epoch_trials{i_epoch};
                            epoch_data = target_tensor(:, :, trs);
                            
                            if strcmp(avg_mode, 'Pooled')
                                activity_per_neuron = squeeze(mean(epoch_data, 3, 'omitnan'));
                                mean_space = mean(activity_per_neuron, 1, 'omitnan');
                                sem_space  = std(activity_per_neuron, 0, 1, 'omitnan') / sqrt(n_units_this_type);
                            else
                                n_m = length(unique_target_mice);
                                mouse_means = nan(n_m, size(epoch_data, 2));
                                for m = 1:n_m
                                    m_idx = (target_mice == unique_target_mice(m));
                                    m_data = epoch_data(m_idx, :, :);
                                    mouse_means(m, :) = mean(mean(m_data, 3, 'omitnan'), 1, 'omitnan');
                                end
                                mean_space = mean(mouse_means, 1, 'omitnan');
                                sem_space  = std(mouse_means, 0, 1, 'omitnan') / sqrt(n_m);
                            end
                            
                            h = shadedErrorBar(1:size(mean_space, 2), mean_space, sem_space, ...
                                'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
                            if ~isempty(h) && isvalid(h.mainLine); legend_handles_spatial_type(end+1) = h.mainLine; end
                        end
                        
                        yl = ylim; y_p = [yl(1), yl(1), yl(2), yl(2)];
                        patch([v_zone(1), v_zone(2), v_zone(2), v_zone(1)], y_p, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                        patch([r_zone(1), r_zone(2), r_zone(2), r_zone(1)], y_p, cfg.plot.colors.dls, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                        if ~isempty(legend_handles_spatial_type); uistack(legend_handles_spatial_type, 'top'); end
                        
                        if strcmp(avg_mode, 'Pooled')
                            title(sprintf('%s - %s (n=%d)', current_name, mod_labels{i_mod}, n_units_this_type));
                        else
                            title(sprintf('%s - %s (N=%d mice)', current_name, mod_labels{i_mod}, length(unique_target_mice)));
                        end
                        box on; xlim([0, size(current_tensor, 2)]);
                        
                        if i_mod == num_mods && i == num_types
                             legend(legend_handles_spatial_type, epoch_names, 'Location', 'best');
                        end
                    else
                        title(sprintf('%s - %s (n=0)', current_name, mod_labels{i_mod}));
                        axis off;
                    end
                end
            end
            title(t_spatial_type, sprintf('%s: Spatial Tuning Evolution (Split by Modulation)', fig_prefix), 'FontSize', 16);
            xlabel(t_spatial_type, 'Spatial Bin', 'FontSize', 12);
            ylabel(t_spatial_type, met_name, 'FontSize', 12);
            
            clean_name_spatial_type = regexprep(sprintf('%s_Spatial_Type_Subpops', fig_prefix), '[\[\]\s:]', '_'); 
            save_to_svg(clean_name_spatial_type);
            
            % -----------------------------------------------------------------
            % FIGURE B: TEMPORAL EVOLUTION BY NEURON TYPE (Split by Modulation)
            % -----------------------------------------------------------------
            fig_temporal_type = figure('Name', sprintf('%s: Temporal', fig_prefix), 'Position', [250, 250, 1400, 450]);
            t_temporal_type = tiledlayout(1, num_mods, 'TileSpacing', 'compact', 'Padding', 'compact');
            
            for i_mod = 1:num_mods
                nexttile; hold on;
                legend_handles_temporal_type = [];
                valid_type_names = {};
                
                for i = 1:num_types
                    current_key = target_keys(i);
                    current_name = neurontype_map(current_key);
                    
                    idx_type = (group_ntype == current_key) & (group_mod_class == i_mod);
                    idx_type(isnan(idx_type)) = false;
                    
                    if sum(idx_type) > 0
                        valid_type_names{end+1} = current_name;
                        target_tensor = current_tensor(idx_type, :, :);
                        target_mice = group_mouse(idx_type);
                        unique_target_mice = unique(target_mice);
                        
                        if strcmp(avg_mode, 'Pooled')
                            activity_per_neuron_temp = squeeze(mean(target_tensor, 2, 'omitnan'));
                            mean_time = mean(activity_per_neuron_temp, 1, 'omitnan');
                            sem_time  = std(activity_per_neuron_temp, 0, 1, 'omitnan') / sqrt(sum(idx_type));
                        else
                            n_m = length(unique_target_mice);
                            mouse_means_temp = nan(n_m, size(target_tensor, 3));
                            for m = 1:n_m
                                m_idx = (target_mice == unique_target_mice(m));
                                m_data = target_tensor(m_idx, :, :);
                                mouse_means_temp(m, :) = mean(mean(m_data, 2, 'omitnan'), 1, 'omitnan');
                            end
                            mean_time = mean(mouse_means_temp, 1, 'omitnan');
                            sem_time  = std(mouse_means_temp, 0, 1, 'omitnan') / sqrt(n_m);
                        end
                        
                        this_color = type_colors(i, :);
                        h = shadedErrorBar(1:size(mean_time, 2), mean_time, sem_time, ...
                            'lineprops', {'-','Color', this_color, 'LineWidth', 2});
                        if ~isempty(h) && isvalid(h.mainLine); legend_handles_temporal_type(end+1) = h.mainLine; end
                    end
                end
                
                xline(0, 'k--', epoch_names{1}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(3, 'k--', epoch_names{2}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(10, 'k--', epoch_names{3}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(20, 'k--', epoch_names{4}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                
                title(sprintf('%s', mod_labels{i_mod}));
                xlabel('Aligned Trial Number');
                if i_mod == 1; ylabel(sprintf('Mean %s', met_name)); end
                xlim([0, size(current_tensor, 3)]);
                if ~isempty(legend_handles_temporal_type)
                    legend(legend_handles_temporal_type, valid_type_names, 'Location', 'best');
                end
                box on; hold off;
            end
            title(t_temporal_type, sprintf('%s: Temporal Evolution by Neuron Type', fig_prefix), 'FontSize', 16);
            
            clean_name_temporal_type = regexprep(sprintf('%s_Temporal_Type_Subpops', fig_prefix), '[\[\]\s:]', '_'); 
            save_to_svg(clean_name_temporal_type);
            
        end % End loop averaging methods
    end % End loop metrics
end % End loop groups (Task/Control)
fprintf('--- Comprehensive Neuron Type Plots Complete ---\n\n');

%% ================= Comprehensive Spatiotemporal Activity by Area AND Neuron Type =================
fprintf('--- Generating 3x4 Spatiotemporal Grids (Area x Cell Type) [Z-scored, Pooled] ---\n');

% --- 1. Prepare Full Clean Data (Pre-subsampling) ---
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_idx = ~neuron_has_nan;

tensor_full_raw = supermouse_tensor_raw(valid_idx, :, :);

% Extract labels
lbls_full.mouse = combined_labels.mouse_labels_all(valid_idx);
lbls_full.group = combined_labels.group_labels_all(valid_idx); % 1=Task, 2=Control
lbls_full.area  = combined_labels.area_labels_all(valid_idx);

% Handle Neuron Types (1=MSN, 2=FSN, 3=TAN, 4=Unclassified)
raw_ntypes = combined_labels.neurontype_labels_all(valid_idx);
processed_ntypes = raw_ntypes;
processed_ntypes(isnan(raw_ntypes) | (raw_ntypes < 1) | (raw_ntypes > 3)) = 4;
lbls_full.ntype = processed_ntypes;

% --- 2. Calculate Z-Scored Tensor ---
fprintf('Calculating Z-scored firing rates for all units...\n');
tensor_full_z = nan(size(tensor_full_raw));
for i = 1:size(tensor_full_raw, 1)
    unit_data = squeeze(tensor_full_raw(i, :, :));
    mu = mean(unit_data(:), 'omitnan');
    sig = std(unit_data(:), 'omitnan');
    if sig > 0
        tensor_full_z(i, :, :) = (tensor_full_raw(i, :, :) - mu) / sig;
    else
        tensor_full_z(i, :, :) = 0; % Handle silent neurons safely
    end
end

% --- 3. Plotting Configuration ---
epoch_trials = {1:3, 4:10, 11:20, 21:30};
epoch_names  = {'Trials 1-3', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
color_t1_3   = min(cfg.plot.colors.epoch_early + 0.3, 1);
epoch_colors = {color_t1_3, cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

bin_size = cfg.plot.zone_params.bin_size;
v_zone = cfg.plot.zone_params.visual_zones_au / bin_size;
r_zone = cfg.plot.zone_params.reward_zone_au / bin_size;

% Grid Definitions
target_areas = {'DMS', 'DLS', 'ACC'};
target_types = [1, 2, 3, 4];
type_names   = {'MSN', 'FSN', 'TAN', 'Unclassified'};

num_areas = length(target_areas);
num_types = length(target_types);
min_units = 5; % Threshold for plotting

datasets = {1, 'Task'; 2, 'Control'};

% --- 4. Generate Grids for Task and Control ---
for ds_idx = 1:size(datasets, 1)
    group_id = datasets{ds_idx, 1};
    ds_name  = datasets{ds_idx, 2};
    
    group_mask = (lbls_full.group == group_id);
    if sum(group_mask) == 0
        fprintf('Skipping %s dataset (no data found).\n', ds_name);
        continue;
    end
    
    fprintf('Generating 3x4 grids for: %s\n', ds_name);
    
    % =========================================================================
    % FIGURE A: SPATIAL TUNING (3x4 Grid)
    % Rows: Area | Cols: Neuron Type
    % =========================================================================
    fig_spatial = figure('Name', sprintf('[%s] Spatial Tuning (Area x Type)', ds_name), ...
                         'Position', [50, 50, 400 * num_types, 300 * num_areas], 'Color', 'w');
    t_spatial = tiledlayout(num_areas, num_types, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_spatial = gobjects(num_areas * num_types, 1);
    tile_counter = 1;
    
    for i_area = 1:num_areas
        current_area = target_areas{i_area};
        
        for i_type = 1:num_types
            current_type_idx = target_types(i_type);
            current_type_name = type_names{i_type};
            
            ax_spatial(tile_counter) = nexttile; hold on;
            legend_handles_spatial = [];
            
            % Intersect Group, Area, and Neuron Type
            idx_target = group_mask & strcmp(lbls_full.area, current_area) & (lbls_full.ntype == current_type_idx);
            n_units = sum(idx_target);
            
            if n_units >= min_units
                target_tensor = tensor_full_z(idx_target, :, :);
                
                for i_epoch = 1:numel(epoch_trials)
                    trs = epoch_trials{i_epoch};
                    epoch_data = target_tensor(:, :, trs);
                    
                    % Pooled Averaging (safe since n_units >= 5)
                    activity_per_neuron = squeeze(mean(epoch_data, 3, 'omitnan'));
                    mean_space = mean(activity_per_neuron, 1, 'omitnan');
                    sem_space  = std(activity_per_neuron, 0, 1, 'omitnan') / sqrt(n_units);
                    
                    h = shadedErrorBar(1:size(mean_space, 2), mean_space, sem_space, ...
                        'lineprops', {'-','Color', epoch_colors{i_epoch}, 'LineWidth', 2});
                    legend_handles_spatial(end+1) = h.mainLine;
                end
                
                % Plot Zones
                yl = ylim; y_p = [yl(1), yl(1), yl(2), yl(2)];
                patch([v_zone(1), v_zone(2), v_zone(2), v_zone(1)], y_p, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                patch([r_zone(1), r_zone(2), r_zone(2), r_zone(1)], y_p, cfg.plot.colors.dls, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                uistack(legend_handles_spatial, 'top');
                
                xlim([0, size(tensor_full_z, 2)]);
            else
                axis off;
                if n_units > 0
                    text(0.5, 0.5, sprintf('Insufficient Units\\n(n=%d)', n_units), ...
                        'HorizontalAlignment', 'center', 'Units', 'normalized', 'Color', [0.5 0.5 0.5]);
                end
            end
            
            title(sprintf('%s - %s (n=%d)', current_area, current_type_name, n_units));
            box on;
            
            % Add legend only to the top-right valid tile
            if i_area == 1 && i_type == num_types && n_units >= min_units
                legend(legend_handles_spatial, epoch_names, 'Location', 'northeast');
            end
            
            tile_counter = tile_counter + 1;
        end
    end
    % linkaxes(ax_spatial(isgraphics(ax_spatial)), 'y'); % Link all Y-axes for scale comparison
    title(t_spatial, sprintf('%s: Spatial Tuning by Area and Cell Type (Z-Scored, Pooled)', ds_name), 'FontSize', 16);
    xlabel(t_spatial, 'Spatial Bin', 'FontSize', 14);
    ylabel(t_spatial, 'Z-Scored FR', 'FontSize', 14);
    
    clean_name_spatial = regexprep(sprintf('[%s]_Spatial_Area_x_Type', ds_name), '[\[\]\s:]', '_'); 
    save_to_svg(clean_name_spatial);
    
    % =========================================================================
    % FIGURE B: TEMPORAL EVOLUTION (3x4 Grid)
    % Rows: Area | Cols: Neuron Type
    % =========================================================================
    fig_temporal = figure('Name', sprintf('[%s] Temporal Evolution (Area x Type)', ds_name), ...
                          'Position', [100, 100, 400 * num_types, 300 * num_areas], 'Color', 'w');
    t_temporal = tiledlayout(num_areas, num_types, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_temp = gobjects(num_areas * num_types, 1);
    tile_counter = 1;
    
    for i_area = 1:num_areas
        current_area = target_areas{i_area};
        area_color = cfg.plot.colors.area_map(current_area); % Assuming area_map exists in your cfg
        
        for i_type = 1:num_types
            current_type_idx = target_types(i_type);
            current_type_name = type_names{i_type};
            
            ax_temp(tile_counter) = nexttile; hold on;
            
            idx_target = group_mask & strcmp(lbls_full.area, current_area) & (lbls_full.ntype == current_type_idx);
            n_units = sum(idx_target);
            
            if n_units >= min_units
                target_tensor = tensor_full_z(idx_target, :, :);
                
                % Average across space first, then plot across trials
                activity_per_neuron_temp = squeeze(mean(target_tensor, 2, 'omitnan'));
                mean_time = mean(activity_per_neuron_temp, 1, 'omitnan');
                sem_time  = std(activity_per_neuron_temp, 0, 1, 'omitnan') / sqrt(n_units);
                
                shadedErrorBar(1:size(mean_time, 2), mean_time, sem_time, ...
                    'lineprops', {'-','Color', area_color, 'LineWidth', 2});
                
                % Plot Trial Markers
                xline(0, 'k--', epoch_names{1}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(3, 'k--', epoch_names{2}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(10, 'k--', epoch_names{3}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                xline(20, 'k--', epoch_names{4}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
                
                xlim([0, size(tensor_full_z, 3)]);
            else
                axis off;
                if n_units > 0
                    text(0.5, 0.5, sprintf('Insufficient Units\\n(n=%d)', n_units), ...
                        'HorizontalAlignment', 'center', 'Units', 'normalized', 'Color', [0.5 0.5 0.5]);
                end
            end
            
            title(sprintf('%s - %s (n=%d)', current_area, current_type_name, n_units));
            box on;
            tile_counter = tile_counter + 1;
        end
    end
    % linkaxes(ax_temp(isgraphics(ax_temp)), 'y'); 
    title(t_temporal, sprintf('%s: Temporal Evolution by Area and Cell Type (Z-Scored, Pooled)', ds_name), 'FontSize', 16);
    xlabel(t_temporal, 'Aligned Trial Number', 'FontSize', 14);
    ylabel(t_temporal, 'Mean Z-Scored FR', 'FontSize', 14);
    
    clean_name_temporal = regexprep(sprintf('[%s]_Temporal_Area_x_Type', ds_name), '[\[\]\s:]', '_'); 
    save_to_svg(clean_name_temporal);
    
end
fprintf('--- 3x4 Spatiotemporal Grids Complete ---\n\n');


%% 14. Spatial Profile of Population Skewness (Hierarchical, Z-Scored)
fprintf('--- Generating Hierarchical Spatial Profiles of Population Skewness ---\n');

% --- 1. Prepare Full Clean Data ---
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_idx = ~neuron_has_nan;

tensor_full_raw = supermouse_tensor_raw(valid_idx, :, :);
lbls_full.mouse = combined_labels.mouse_labels_all(valid_idx);
lbls_full.group = combined_labels.group_labels_all(valid_idx); % 1=Task, 2=Control
lbls_full.area  = combined_labels.area_labels_all(valid_idx);

% --- 2. Calculate Z-Scored Tensor ---
fprintf('Calculating Z-scored firing rates...\n');
tensor_full_z = nan(size(tensor_full_raw));
for i = 1:size(tensor_full_raw, 1)
    unit_data = squeeze(tensor_full_raw(i, :, :));
    mu = mean(unit_data(:), 'omitnan');
    sig = std(unit_data(:), 'omitnan');
    if sig > 0
        tensor_full_z(i, :, :) = (tensor_full_raw(i, :, :) - mu) / sig;
    else
        tensor_full_z(i, :, :) = 0; 
    end
end

% --- 3. Plotting Configuration ---
epoch_trials = {1:3, 4:10, 11:20, 21:30};
epoch_names  = {'Trials 1-3', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
n_epochs = length(epoch_trials);

color_t1_3   = min(cfg.plot.colors.epoch_early + 0.3, 1);
epoch_colors = {color_t1_3, cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

bin_size = cfg.plot.zone_params.bin_size;
v_zone = cfg.plot.zone_params.visual_zones_au / bin_size;
r_zone = cfg.plot.zone_params.reward_zone_au / bin_size;
n_bins = size(tensor_full_z, 2);

target_areas = {'DMS', 'DLS', 'ACC'};
num_areas = length(target_areas);
datasets = {1, 'Task'; 2, 'Control'};

min_units = 5; % Require at least 5 neurons per area PER MOUSE

% --- 4. Generate Spatial Plots (Hierarchical) ---
for ds_idx = 1:size(datasets, 1)
    group_id = datasets{ds_idx, 1};
    ds_name  = datasets{ds_idx, 2};
    
    group_mask = (lbls_full.group == group_id);
    if sum(group_mask) == 0, continue; end
    
    unique_mice = unique(lbls_full.mouse(group_mask));
    n_mice_total = length(unique_mice);
    
    fig_name = sprintf('[%s] Spatial_Skewness_Profile_Hierarchical', ds_name);
    figure('Name', fig_name, 'Position', [100, 100, 400 * num_areas, 400], 'Color', 'w');
    t_spatial = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_spatial = gobjects(1, num_areas);
    
    for i_area = 1:num_areas
        current_area = target_areas{i_area};
        ax_spatial(i_area) = nexttile; hold on;
        legend_handles_spatial = [];
        
        % Initialize storage: [Mice x Epochs x Bins]
        mouse_skewness = nan(n_mice_total, n_epochs, n_bins);
        valid_mice_count = 0;
        
        % Calculate Skewness PER MOUSE
        for m = 1:n_mice_total
            curr_mouse = unique_mice(m);
            idx_target = group_mask & (lbls_full.mouse == curr_mouse) & strcmp(lbls_full.area, current_area);
            n_units_mouse = sum(idx_target);
            
            if n_units_mouse >= min_units
                valid_mice_count = valid_mice_count + 1;
                mouse_tensor = tensor_full_z(idx_target, :, :); % [Neurons x Bins x Trials]
                
                for e = 1:n_epochs
                    trs = epoch_trials{e};
                    % Get stable spatial tuning curves for this epoch
                    tuning_curves = mean(mouse_tensor(:, :, trs), 3, 'omitnan'); % [Neurons x Bins]
                    
                    for b = 1:n_bins
                        r = tuning_curves(:, b); % True population vector for this mouse
                        if var(r) > 1e-6 
                            mouse_skewness(m, e, b) = skewness(r, 0); 
                        end
                    end
                end
            end
        end
        
        % Plot Average Across Mice
        if valid_mice_count > 0
            for e = 1:n_epochs
                % Extract all mice for this epoch: [Mice x Bins]
                data_e = squeeze(mouse_skewness(:, e, :)); 
                
                % Hierarchical mean and SEM (N = valid mice)
                mu_space = mean(data_e, 1, 'omitnan');
                sem_space = std(data_e, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data_e), 1));
                
                h = shadedErrorBar(1:n_bins, mu_space, sem_space, ...
                    'lineprops', {'-','Color', epoch_colors{e}, 'LineWidth', 2});
                
                if ~isempty(h) && isvalid(h.mainLine)
                    legend_handles_spatial(end+1) = h.mainLine;
                end
            end
            
            % Plot Landmarks
            yl = ylim; y_p = [yl(1), yl(1), yl(2), yl(2)];
            patch([v_zone(1), v_zone(2), v_zone(2), v_zone(1)], y_p, [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
            patch([r_zone(1), r_zone(2), r_zone(2), r_zone(1)], y_p, cfg.plot.colors.dls, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            uistack(legend_handles_spatial, 'top');
            
            xlim([0, n_bins]);
        else
            axis off;
            text(0.5, 0.5, 'Insufficient Mice', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'Color', [0.5 0.5 0.5]);
        end
        
        title(sprintf('%s (N=%d mice)', current_area, valid_mice_count));
        box on;
        
        if i_area == num_areas && valid_mice_count > 0
            legend(legend_handles_spatial, epoch_names, 'Location', 'northeast');
        end
    end
    
    linkaxes(ax_spatial(isgraphics(ax_spatial)), 'y');
    title(t_spatial, sprintf('%s: Spatial Profile of Population Skewness (Hierarchical)', ds_name), 'FontSize', 16);
    xlabel(t_spatial, 'Spatial Bin', 'FontSize', 14);
    ylabel(t_spatial, 'Population Skewness', 'FontSize', 14);
    
    clean_name = regexprep(sprintf('[%s]_Spatial_Skewness_Profile_Hierarchical', ds_name), '[\[\]\s:]', '_'); 
    save_to_svg(clean_name);
end
fprintf('--- Hierarchical Spatial Skewness Profiles Complete ---\n\n');


%% 15. Temporal Evolution of Population Skewness (Hierarchical, Z-Scored)
fprintf('--- Generating Hierarchical Temporal Profiles of Population Skewness ---\n');

% --- 1. Prepare Full Clean Data ---
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_idx = ~neuron_has_nan;

tensor_full_raw = supermouse_tensor_raw(valid_idx, :, :);
lbls_full.mouse = combined_labels.mouse_labels_all(valid_idx);
lbls_full.group = combined_labels.group_labels_all(valid_idx); % 1=Task, 2=Control
lbls_full.area  = combined_labels.area_labels_all(valid_idx);

% --- 2. Calculate Z-Scored Tensor ---
fprintf('Calculating Z-scored firing rates...\n');
tensor_full_z = nan(size(tensor_full_raw));
for i = 1:size(tensor_full_raw, 1)
    unit_data = squeeze(tensor_full_raw(i, :, :));
    mu = mean(unit_data(:), 'omitnan');
    sig = std(unit_data(:), 'omitnan');
    if sig > 0
        tensor_full_z(i, :, :) = (tensor_full_raw(i, :, :) - mu) / sig;
    else
        tensor_full_z(i, :, :) = 0; 
    end
end

% --- 3. Plotting Configuration ---
epoch_names  = {'Trials 1-3', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
n_trials_total = size(tensor_full_z, 3);

target_areas = {'DMS', 'DLS', 'ACC'};
num_areas = length(target_areas);
datasets = {1, 'Task'; 2, 'Control'};

min_units = 5; % Require at least 5 neurons per area PER MOUSE
mov_avg_win = 5; % Smoothing window for trial-by-trial temporal plotting

% --- 4. Generate Temporal Plots (Hierarchical) ---
for ds_idx = 1:size(datasets, 1)
    group_id = datasets{ds_idx, 1};
    ds_name  = datasets{ds_idx, 2};
    
    group_mask = (lbls_full.group == group_id);
    if sum(group_mask) == 0, continue; end
    
    unique_mice = unique(lbls_full.mouse(group_mask));
    n_mice_total = length(unique_mice);
    
    fig_name = sprintf('[%s] Temporal_Skewness_Profile_Hierarchical', ds_name);
    figure('Name', fig_name, 'Position', [150, 150, 400 * num_areas, 400], 'Color', 'w');
    t_temporal = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_temp = gobjects(1, num_areas);
    
    for i_area = 1:num_areas
        current_area = target_areas{i_area};
        area_color = cfg.plot.colors.area_map(current_area);
        ax_temp(i_area) = nexttile; hold on;
        
        % Initialize storage: [Mice x Trials]
        mouse_skewness_time = nan(n_mice_total, n_trials_total);
        valid_mice_count = 0;
        
        % Calculate Skewness PER MOUSE across time
        for m = 1:n_mice_total
            curr_mouse = unique_mice(m);
            idx_target = group_mask & (lbls_full.mouse == curr_mouse) & strcmp(lbls_full.area, current_area);
            n_units_mouse = sum(idx_target);
            
            if n_units_mouse >= min_units
                valid_mice_count = valid_mice_count + 1;
                
                % [Neurons x Bins x Trials]
                mouse_tensor = tensor_full_z(idx_target, :, :); 
                
                % Average across spatial bins to get mean trial rate: [Neurons x Trials]
                mouse_trial_rates = squeeze(mean(mouse_tensor, 2, 'omitnan'));
                
                for tr = 1:n_trials_total
                    r = mouse_trial_rates(:, tr); % Population vector for this trial
                    if var(r) > 1e-6 
                        mouse_skewness_time(m, tr) = skewness(r, 0); 
                    end
                end
            end
        end
        
        % Plot Average Across Mice
        if valid_mice_count > 0
            % Extract valid mice
            valid_rows = ~all(isnan(mouse_skewness_time), 2);
            data_valid = mouse_skewness_time(valid_rows, :);
            
            % Smooth trial-by-trial data for each mouse to reduce high-frequency noise
            data_smoothed = smoothdata(data_valid, 2, 'movmean', mov_avg_win, 'omitnan');
            
            % Hierarchical mean and SEM (N = valid mice)
            mu_time = mean(data_smoothed, 1, 'omitnan');
            sem_time = std(data_smoothed, 0, 1, 'omitnan') ./ sqrt(valid_mice_count);
            
            shadedErrorBar(1:n_trials_total, mu_time, sem_time, ...
                'lineprops', {'-','Color', area_color, 'LineWidth', 2});
            
            % Plot Epoch Markers (assuming 0 is trial 1, 3 is start of epoch 2, etc. based on prior code)
            xline(0, 'k--', epoch_names{1}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            xline(3, 'k--', epoch_names{2}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            xline(10, 'k--', epoch_names{3}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            xline(20, 'k--', epoch_names{4}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            
            xlim([0, n_trials_total]);
        else
            axis off;
            text(0.5, 0.5, 'Insufficient Mice', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'Color', [0.5 0.5 0.5]);
        end
        
        title(sprintf('%s (N=%d mice)', current_area, valid_mice_count));
        box on;
    end
    
    linkaxes(ax_temp(isgraphics(ax_temp)), 'y');
    title(t_temporal, sprintf('%s: Temporal Evolution of Population Skewness (Hierarchical)', ds_name), 'FontSize', 16);
    xlabel(t_temporal, 'Aligned Trial Number', 'FontSize', 14);
    ylabel(t_temporal, 'Population Skewness (Smoothed)', 'FontSize', 14);
    
    clean_name = regexprep(sprintf('[%s]_Temporal_Skewness_Profile_Hierarchical', ds_name), '[\[\]\s:]', '_'); 
    save_to_svg(clean_name);
end
fprintf('--- Hierarchical Temporal Skewness Profiles Complete ---\n\n');

%% 15. Temporal Evolution of Population Skewness (Hierarchical, Averaged across Space)
fprintf('--- Generating Hierarchical Temporal Profiles of Population Skewness ---\n');

% --- 1. Prepare Full Clean Data ---
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_idx = ~neuron_has_nan;

tensor_full_raw = supermouse_tensor_raw(valid_idx, :, :);
lbls_full.mouse = combined_labels.mouse_labels_all(valid_idx);
lbls_full.group = combined_labels.group_labels_all(valid_idx); % 1=Task, 2=Control
lbls_full.area  = combined_labels.area_labels_all(valid_idx);

% --- 2. Calculate Z-Scored Tensor ---
fprintf('Calculating Z-scored firing rates...\n');
tensor_full_z = nan(size(tensor_full_raw));
for i = 1:size(tensor_full_raw, 1)
    unit_data = squeeze(tensor_full_raw(i, :, :));
    mu = mean(unit_data(:), 'omitnan');
    sig = std(unit_data(:), 'omitnan');
    if sig > 0
        tensor_full_z(i, :, :) = (tensor_full_raw(i, :, :) - mu) / sig;
    else
        tensor_full_z(i, :, :) = 0; 
    end
end

% --- 3. Plotting Configuration ---
epoch_names  = {'Trials 1-3', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
n_bins         = size(tensor_full_z, 2);
n_trials_total = size(tensor_full_z, 3);

target_areas = {'DMS', 'DLS', 'ACC'};
num_areas = length(target_areas);
datasets = {1, 'Task'; 2, 'Control'};

min_units = 5; % Require at least 5 neurons per area PER MOUSE
mov_avg_win = 5; % Smoothing window for trial-by-trial temporal plotting

% --- 4. Generate Temporal Plots (Hierarchical) ---
for ds_idx = 1:size(datasets, 1)
    group_id = datasets{ds_idx, 1};
    ds_name  = datasets{ds_idx, 2};
    
    group_mask = (lbls_full.group == group_id);
    if sum(group_mask) == 0, continue; end
    
    unique_mice = unique(lbls_full.mouse(group_mask));
    n_mice_total = length(unique_mice);
    
    fig_name = sprintf('[%s] Temporal_Skewness_Profile_Hierarchical', ds_name);
    figure('Name', fig_name, 'Position', [150, 150, 400 * num_areas, 400], 'Color', 'w');
    t_temporal = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_temp = gobjects(1, num_areas);
    
    for i_area = 1:num_areas
        current_area = target_areas{i_area};
        area_color = cfg.plot.colors.area_map(current_area);
        ax_temp(i_area) = nexttile; hold on;
        
        % Initialize storage: [Mice x Trials]
        mouse_skewness_time = nan(n_mice_total, n_trials_total);
        valid_mice_count = 0;
        
        % Calculate Skewness PER MOUSE across time
        for m = 1:n_mice_total
            curr_mouse = unique_mice(m);
            idx_target = group_mask & (lbls_full.mouse == curr_mouse) & strcmp(lbls_full.area, current_area);
            n_units_mouse = sum(idx_target);
            
            if n_units_mouse >= min_units
                valid_mice_count = valid_mice_count + 1;
                
                % [Neurons x Bins x Trials]
                mouse_tensor = tensor_full_z(idx_target, :, :); 
                
                for tr = 1:n_trials_total
                    % Calculate skewness for EACH BIN in this trial
                    bin_skewness = nan(1, n_bins);
                    for b = 1:n_bins
                        r = mouse_tensor(:, b, tr); % Population vector for this bin
                        if var(r) > 1e-6 
                            bin_skewness(b) = skewness(r, 0); 
                        end
                    end
                    
                    % Average the bin-wise skewness to get a single trial metric
                    mouse_skewness_time(m, tr) = mean(bin_skewness, 'omitnan');
                end
            end
        end
        
        % Plot Average Across Mice
        if valid_mice_count > 0
            % Extract valid mice
            valid_rows = ~all(isnan(mouse_skewness_time), 2);
            data_valid = mouse_skewness_time(valid_rows, :);
            
            % Smooth trial-by-trial data for each mouse to reduce high-frequency noise
            data_smoothed = smoothdata(data_valid, 2, 'movmean', mov_avg_win, 'omitnan');
            
            % Hierarchical mean and SEM (N = valid mice)
            mu_time = mean(data_smoothed, 1, 'omitnan');
            sem_time = std(data_smoothed, 0, 1, 'omitnan') ./ sqrt(valid_mice_count);
            
            shadedErrorBar(1:n_trials_total, mu_time, sem_time, ...
                'lineprops', {'-','Color', area_color, 'LineWidth', 2});
            
            % Plot Epoch Markers 
            xline(0, 'k--', epoch_names{1}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            xline(3, 'k--', epoch_names{2}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            xline(10, 'k--', epoch_names{3}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            xline(20, 'k--', epoch_names{4}, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
            
            xlim([0, n_trials_total]);
        else
            axis off;
            text(0.5, 0.5, 'Insufficient Mice', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'Color', [0.5 0.5 0.5]);
        end
        
        title(sprintf('%s (N=%d mice)', current_area, valid_mice_count));
        box on;
    end
    
    linkaxes(ax_temp(isgraphics(ax_temp)), 'y');
    title(t_temporal, sprintf('%s: Temporal Evolution of Population Skewness (Hierarchical)', ds_name), 'FontSize', 16);
    xlabel(t_temporal, 'Aligned Trial Number', 'FontSize', 14);
    ylabel(t_temporal, 'Mean Spatial Skewness (Smoothed)', 'FontSize', 14);
    
    clean_name = regexprep(sprintf('[%s]_Temporal_Skewness_Profile_Hierarchical', ds_name), '[\[\]\s:]', '_'); 
    save_to_svg(clean_name);
end
fprintf('--- Hierarchical Temporal Skewness Profiles Complete ---\n\n');

%% 16. Population Distributions of Mean Unit Activity (Hierarchical & KS Test)
fprintf('--- Generating Hierarchical Mean Activity Distributions by Area and Cell Type ---\n');

% --- 1. Prepare Full Clean Data ---
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_idx = ~neuron_has_nan;

tensor_full_raw = supermouse_tensor_raw(valid_idx, :, :);
lbls_full.mouse = combined_labels.mouse_labels_all(valid_idx);
lbls_full.group = combined_labels.group_labels_all(valid_idx); 
lbls_full.area  = combined_labels.area_labels_all(valid_idx);

% Handle Neuron Types
raw_ntypes = combined_labels.neurontype_labels_all(valid_idx);
processed_ntypes = raw_ntypes;
processed_ntypes(isnan(raw_ntypes) | (raw_ntypes < 1) | (raw_ntypes > 3)) = 4;
lbls_full.ntype = processed_ntypes;

% --- 2. Calculate Z-Scored Tensor ---
fprintf('Calculating Z-scored firing rates...\n');
tensor_full_z = nan(size(tensor_full_raw));
for i = 1:size(tensor_full_raw, 1)
    unit_data = squeeze(tensor_full_raw(i, :, :));
    mu = mean(unit_data(:), 'omitnan');
    sig = std(unit_data(:), 'omitnan');
    if sig > 0
        tensor_full_z(i, :, :) = (tensor_full_raw(i, :, :) - mu) / sig;
    else
        tensor_full_z(i, :, :) = 0; 
    end
end

% --- 3. Plotting Configuration ---
epoch_trials = {1:3, 4:10, 11:20, 21:30};
epoch_names  = {'Trials 1-3 (Naive)', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
color_t1_3   = min(cfg.plot.colors.epoch_early + 0.3, 1);
epoch_colors = {color_t1_3, cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

target_trials = [1, 4, 21];
trial_names   = {'Trial 1 (Naive)', 'Trial 4', 'Trial 1 of Expert (Tr 21)'};
trial_colors  = {[0 0.4470 0.7410], [0.4940 0.1840 0.5560], [0.8500 0.3250 0.0980]};

target_areas = {'DMS', 'DLS', 'ACC'};
target_types = [1, 2, 3];
type_names   = {'MSN', 'FSN', 'TAN'};

num_areas = length(target_areas);
num_types = length(target_types);
datasets = {1, 'Task'; 2, 'Control'};

min_units_per_mouse = 3; % Need a few neurons per mouse to form a coherent PDF

% Define histogram edges and compute bin centers for plotting lines
hist_edges = linspace(-2, 2, 25); 
bin_centers = hist_edges(1:end-1) + diff(hist_edges)/2;

% --- 4. Generate Distribution Plots ---
for ds_idx = 1:size(datasets, 1)
    group_id = datasets{ds_idx, 1};
    ds_name  = datasets{ds_idx, 2};
    
    group_mask = (lbls_full.group == group_id);
    if sum(group_mask) == 0, continue; end
    
    for i_type = 1:num_types
        current_type_idx = target_types(i_type);
        current_type_name = type_names{i_type};
        
        % =====================================================================
        % FIGURE A: Distributions by Epoch
        % =====================================================================
        fig_epochs = figure('Name', sprintf('[%s] %s Mean Activity - Epochs', ds_name, current_type_name), ...
                            'Position', [100, 100, 400 * num_areas, 400], 'Color', 'w');
        t_epochs = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
        ax_epochs = gobjects(1, num_areas);
        
        for i_area = 1:num_areas
            current_area = target_areas{i_area};
            ax_epochs(i_area) = nexttile; hold on;
            
            idx_target = group_mask & strcmp(lbls_full.area, current_area) & (lbls_full.ntype == current_type_idx);
            
            % Find valid mice (>= min_units_per_mouse)
            target_mice = unique(lbls_full.mouse(idx_target));
            valid_mice = [];
            for m = 1:length(target_mice)
                if sum(idx_target & (lbls_full.mouse == target_mice(m))) >= min_units_per_mouse
                    valid_mice(end+1) = target_mice(m);
                end
            end
            
            n_mice = length(valid_mice);
            
            if n_mice >= 2 % Need at least 2 mice to compute SEM
                epoch_raw_pooled = cell(1, length(epoch_trials));
                legend_handles = [];
                
                for e = 1:length(epoch_trials)
                    trs = epoch_trials{e};
                    mouse_pdfs = nan(n_mice, length(bin_centers));
                    raw_e = [];
                    
                    for m = 1:n_mice
                        idx_mouse = idx_target & (lbls_full.mouse == valid_mice(m));
                        target_tensor = tensor_full_z(idx_mouse, :, :); % [Neurons x Bins x Trials]
                        
                        % Mean activity per trial: [Neurons x Trials]
                        unit_trial_rates = squeeze(mean(target_tensor, 2, 'omitnan')); 
                        
                        % Mean activity within epoch: [Neurons x 1]
                        epoch_rates = mean(unit_trial_rates(:, trs), 2, 'omitnan');
                        
                        % Compute PDF for this mouse
                        mouse_pdfs(m, :) = histcounts(epoch_rates, hist_edges, 'Normalization', 'pdf');
                        raw_e = [raw_e; epoch_rates]; % Accumulate raw rates for KS test
                    end
                    
                    epoch_raw_pooled{e} = raw_e(~isnan(raw_e));
                    
                    % Average PDFs across mice
                    mu_pdf = mean(mouse_pdfs, 1, 'omitnan');
                    se_pdf = std(mouse_pdfs, 0, 1, 'omitnan') / sqrt(n_mice);
                    
                    h = shadedErrorBar(bin_centers, mu_pdf, se_pdf, ...
                        'lineprops', {'-', 'Color', epoch_colors{e}, 'LineWidth', 2}, 'patchSaturation', 0.15);
                    legend_handles(end+1) = h.mainLine;
                end
                
                % KS Test: Naive (Epoch 1) vs Expert (Epoch 4)
                [~, p_ks, ks_stat] = kstest2(epoch_raw_pooled{1}, epoch_raw_pooled{4});
                
                % Significance formatting
                sig_star = ''; 
                if p_ks < 0.05, sig_star = '*'; end
                if p_ks < 0.01, sig_star = '**'; end
                if p_ks < 0.001, sig_star = '***'; end
                
                ks_txt = sprintf('KS(Naive, Exp):\nD=%.2f, p=%.1e %s', ks_stat, p_ks, sig_star);
                text(0.05, 0.95, ks_txt, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                    'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7]);
                
                grid on;
            else
                axis off;
                text(0.5, 0.5, 'Insufficient Mice', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
            end
            
            title(sprintf('%s (N=%d mice)', current_area, n_mice));
            if i_area == 1; ylabel('Probability Density'); end
            xlabel('Mean Z-Scored Activity');
            if i_area == num_areas && n_mice >= 2; legend(legend_handles, epoch_names, 'Location', 'northeast'); end
        end
        linkaxes(ax_epochs(isgraphics(ax_epochs)), 'y');
        xlim(ax_epochs(isgraphics(ax_epochs)), [hist_edges(1), hist_edges(end)]);
        title(t_epochs, sprintf('%s: %s Mean Activity Distributions by Epoch (Hierarchical)', ds_name, current_type_name), 'FontSize', 16);
        
        save_to_svg(regexprep(sprintf('[%s]_MeanDist_Epochs_Hierarchical_%s', ds_name, current_type_name), '[\[\]\s:]', '_'));
        
        % =====================================================================
        % FIGURE B: Distributions by Specific Trial
        % =====================================================================
        fig_trials = figure('Name', sprintf('[%s] %s Mean Activity - Specific Trials', ds_name, current_type_name), ...
                            'Position', [150, 150, 400 * num_areas, 400], 'Color', 'w');
        t_trials = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
        ax_trials = gobjects(1, num_areas);
        
        for i_area = 1:num_areas
            current_area = target_areas{i_area};
            ax_trials(i_area) = nexttile; hold on;
            
            idx_target = group_mask & strcmp(lbls_full.area, current_area) & (lbls_full.ntype == current_type_idx);
            
            % Find valid mice (>= min_units_per_mouse)
            target_mice = unique(lbls_full.mouse(idx_target));
            valid_mice = [];
            for m = 1:length(target_mice)
                if sum(idx_target & (lbls_full.mouse == target_mice(m))) >= min_units_per_mouse
                    valid_mice(end+1) = target_mice(m);
                end
            end
            
            n_mice = length(valid_mice);
            
            if n_mice >= 2
                trial_raw_pooled = cell(1, length(target_trials));
                legend_handles = [];
                
                for t = 1:length(target_trials)
                    tr_idx = target_trials(t);
                    mouse_pdfs = nan(n_mice, length(bin_centers));
                    raw_t = [];
                    
                    for m = 1:n_mice
                        idx_mouse = idx_target & (lbls_full.mouse == valid_mice(m));
                        target_tensor = tensor_full_z(idx_mouse, :, :); 
                        
                        % If trial doesn't exist for this tensor, skip
                        if tr_idx > size(target_tensor, 3)
                            continue;
                        end
                        
                        % Mean activity per trial: [Neurons x Trials]
                        unit_trial_rates = squeeze(mean(target_tensor, 2, 'omitnan')); 
                        
                        % Extract specific trial: [Neurons x 1]
                        trial_rates = unit_trial_rates(:, tr_idx);
                        
                        mouse_pdfs(m, :) = histcounts(trial_rates, hist_edges, 'Normalization', 'pdf');
                        raw_t = [raw_t; trial_rates];
                    end
                    
                    trial_raw_pooled{t} = raw_t(~isnan(raw_t));
                    
                    % Average PDFs across mice
                    mu_pdf = mean(mouse_pdfs, 1, 'omitnan');
                    se_pdf = std(mouse_pdfs, 0, 1, 'omitnan') / sqrt(n_mice);
                    
                    h = shadedErrorBar(bin_centers, mu_pdf, se_pdf, ...
                        'lineprops', {'-', 'Color', trial_colors{t}, 'LineWidth', 2}, 'patchSaturation', 0.15);
                    legend_handles(end+1) = h.mainLine;
                end
                
                % KS Test: Trial 1 vs Trial 21
                if ~isempty(trial_raw_pooled{1}) && ~isempty(trial_raw_pooled{3})
                    [~, p_ks, ks_stat] = kstest2(trial_raw_pooled{1}, trial_raw_pooled{3});
                    
                    sig_star = ''; 
                    if p_ks < 0.05, sig_star = '*'; end
                    if p_ks < 0.01, sig_star = '**'; end
                    if p_ks < 0.001, sig_star = '***'; end
                    
                    ks_txt = sprintf('KS(Tr1, Tr21):\nD=%.2f, p=%.1e %s', ks_stat, p_ks, sig_star);
                    text(0.05, 0.95, ks_txt, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                        'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7]);
                end
                
                grid on;
            else
                axis off;
                text(0.5, 0.5, 'Insufficient Mice', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
            end
            
            title(sprintf('%s (N=%d mice)', current_area, n_mice));
            if i_area == 1; ylabel('Probability Density'); end
            xlabel('Mean Z-Scored Activity');
            if i_area == num_areas && n_mice >= 2; legend(legend_handles, trial_names, 'Location', 'northeast'); end
        end
        linkaxes(ax_trials(isgraphics(ax_trials)), 'y');
        xlim(ax_trials(isgraphics(ax_trials)), [hist_edges(1), hist_edges(end)]);
        title(t_trials, sprintf('%s: %s Mean Activity Distributions by Specific Trial (Hierarchical)', ds_name, current_type_name), 'FontSize', 16);
        
        save_to_svg(regexprep(sprintf('[%s]_MeanDist_Trials_Hierarchical_%s', ds_name, current_type_name), '[\[\]\s:]', '_'));
    end
end
fprintf('--- Hierarchical Mean Activity Distribution Plots Complete ---\n\n');

%% 16. Population Distributions of Mean Unit Activity (Hierarchical KDE & KS Test)
fprintf('--- Generating Hierarchical Mean Activity Distributions (KDE) by Area and Cell Type ---\n');

% --- 1. Prepare Full Clean Data ---
neuron_has_nan = squeeze(any(isnan(supermouse_tensor_raw), [2, 3]));
valid_idx = ~neuron_has_nan;

tensor_full_raw = supermouse_tensor_raw(valid_idx, :, :);
lbls_full.mouse = combined_labels.mouse_labels_all(valid_idx);
lbls_full.group = combined_labels.group_labels_all(valid_idx); 
lbls_full.area  = combined_labels.area_labels_all(valid_idx);

% Handle Neuron Types
raw_ntypes = combined_labels.neurontype_labels_all(valid_idx);
processed_ntypes = raw_ntypes;
processed_ntypes(isnan(raw_ntypes) | (raw_ntypes < 1) | (raw_ntypes > 3)) = 4;
lbls_full.ntype = processed_ntypes;

% --- 2. Calculate Z-Scored Tensor ---
fprintf('Calculating Z-scored firing rates...\n');
tensor_full_z = nan(size(tensor_full_raw));
for i = 1:size(tensor_full_raw, 1)
    unit_data = squeeze(tensor_full_raw(i, :, :));
    mu = mean(unit_data(:), 'omitnan');
    sig = std(unit_data(:), 'omitnan');
    if sig > 0
        tensor_full_z(i, :, :) = (tensor_full_raw(i, :, :) - mu) / sig;
    else
        tensor_full_z(i, :, :) = 0; 
    end
end

% --- 3. Plotting Configuration ---
epoch_trials = {1:3, 4:10, 11:20, 21:30};
epoch_names  = {'Trials 1-3 (Naive)', 'Trials 4-10', cfg.plot.epoch_names{2}, cfg.plot.epoch_names{3}};
color_t1_3   = min(cfg.plot.colors.epoch_early + 0.3, 1);
epoch_colors = {color_t1_3, cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};

target_trials = [1, 4, 21];
trial_names   = {'Trial 1 (Naive)', 'Trial 4', 'Trial 1 of Expert (Tr 21)'};
trial_colors  = {[0 0.4470 0.7410], [0.4940 0.1840 0.5560], [0.8500 0.3250 0.0980]};

target_areas = {'DMS', 'DLS', 'ACC'};
target_types = [1, 2, 3];
type_names   = {'MSN', 'FSN', 'TAN'};

num_areas = length(target_areas);
num_types = length(target_types);
datasets = {1, 'Task'; 2, 'Control'};

min_units_per_mouse = 3; 

% KDE Evaluation Points (Smoother and continuous compared to histogram edges)
x_eval = linspace(-1.5, 1.5, 100); 

% --- 4. Generate Distribution Plots ---
for ds_idx = 1:size(datasets, 1)
    group_id = datasets{ds_idx, 1};
    ds_name  = datasets{ds_idx, 2};
    
    group_mask = (lbls_full.group == group_id);
    if sum(group_mask) == 0, continue; end
    
    for i_type = 1:num_types
        current_type_idx = target_types(i_type);
        current_type_name = type_names{i_type};
        
        % =====================================================================
        % FIGURE A: Distributions by Epoch
        % =====================================================================
        fig_epochs = figure('Name', sprintf('[%s] %s Mean Activity - Epochs', ds_name, current_type_name), ...
                            'Position', [100, 100, 400 * num_areas, 400], 'Color', 'w');
        t_epochs = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
        ax_epochs = gobjects(1, num_areas);
        
        for i_area = 1:num_areas
            current_area = target_areas{i_area};
            ax_epochs(i_area) = nexttile; hold on;
            
            idx_target = group_mask & strcmp(lbls_full.area, current_area) & (lbls_full.ntype == current_type_idx);
            
            % Find valid mice
            target_mice = unique(lbls_full.mouse(idx_target));
            valid_mice = [];
            for m = 1:length(target_mice)
                if sum(idx_target & (lbls_full.mouse == target_mice(m))) >= min_units_per_mouse
                    valid_mice(end+1) = target_mice(m);
                end
            end
            
            n_mice = length(valid_mice);
            
            if n_mice >= 2 
                epoch_raw_pooled = cell(1, length(epoch_trials));
                legend_handles = [];
                
                for e = 1:length(epoch_trials)
                    trs = epoch_trials{e};
                    mouse_pdfs = nan(n_mice, length(x_eval));
                    raw_e = [];
                    
                    for m = 1:n_mice
                        idx_mouse = idx_target & (lbls_full.mouse == valid_mice(m));
                        target_tensor = tensor_full_z(idx_mouse, :, :); 
                        
                        % Mean activity per trial: [Neurons x Trials]
                        unit_trial_rates = squeeze(mean(target_tensor, 2, 'omitnan')); 
                        
                        % Mean activity within epoch: [Neurons x 1]
                        epoch_rates = mean(unit_trial_rates(:, trs), 2, 'omitnan');
                        epoch_rates = epoch_rates(~isnan(epoch_rates));
                        
                        % Compute continuous KDE for this mouse
                        if length(epoch_rates) >= 2
                            [f_mouse, ~] = ksdensity(epoch_rates, x_eval);
                            mouse_pdfs(m, :) = f_mouse;
                        end
                        raw_e = [raw_e; epoch_rates]; 
                    end
                    
                    epoch_raw_pooled{e} = raw_e(~isnan(raw_e));
                    
                    % Average KDE PDFs across mice
                    mu_pdf = mean(mouse_pdfs, 1, 'omitnan');
                    se_pdf = std(mouse_pdfs, 0, 1, 'omitnan') / sqrt(n_mice);
                    
                    h = shadedErrorBar(x_eval, mu_pdf, se_pdf, ...
                        'lineprops', {'-', 'Color', epoch_colors{e}, 'LineWidth', 2}, 'patchSaturation', 0.15);
                    legend_handles(end+1) = h.mainLine;
                end
                
                % KS Test: Naive (Epoch 1) vs Expert (Epoch 4)
                if ~isempty(epoch_raw_pooled{1}) && ~isempty(epoch_raw_pooled{4})
                    [~, p_ks, ks_stat] = kstest2(epoch_raw_pooled{1}, epoch_raw_pooled{4});
                    
                    sig_star = ''; 
                    if p_ks < 0.05, sig_star = '*'; end
                    if p_ks < 0.01, sig_star = '**'; end
                    if p_ks < 0.001, sig_star = '***'; end
                    
                    ks_txt = sprintf('KS(Naive, Exp):\nD=%.2f, p=%.1e %s', ks_stat, p_ks, sig_star);
                    text(0.05, 0.95, ks_txt, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                        'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7]);
                end
                grid on;
            else
                axis off;
                text(0.5, 0.5, 'Insufficient Mice', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
            end
            
            title(sprintf('%s (N=%d mice)', current_area, n_mice));
            if i_area == 1; ylabel('Probability Density'); end
            xlabel('Mean Z-Scored Activity');
            if i_area == num_areas && n_mice >= 2; legend(legend_handles, epoch_names, 'Location', 'northeast'); end
        end
        linkaxes(ax_epochs(isgraphics(ax_epochs)), 'y');
        xlim(ax_epochs(isgraphics(ax_epochs)), [x_eval(1), x_eval(end)]);
        title(t_epochs, sprintf('%s: %s Mean Activity Distributions by Epoch (KDE)', ds_name, current_type_name), 'FontSize', 16);
        
        save_to_svg(regexprep(sprintf('[%s]_MeanDist_Epochs_Hierarchical_%s', ds_name, current_type_name), '[\[\]\s:]', '_'));
        
        % =====================================================================
        % FIGURE B: Distributions by Specific Trial
        % =====================================================================
        fig_trials = figure('Name', sprintf('[%s] %s Mean Activity - Specific Trials', ds_name, current_type_name), ...
                            'Position', [150, 150, 400 * num_areas, 400], 'Color', 'w');
        t_trials = tiledlayout(1, num_areas, 'TileSpacing', 'compact', 'Padding', 'compact');
        ax_trials = gobjects(1, num_areas);
        
        for i_area = 1:num_areas
            current_area = target_areas{i_area};
            ax_trials(i_area) = nexttile; hold on;
            
            idx_target = group_mask & strcmp(lbls_full.area, current_area) & (lbls_full.ntype == current_type_idx);
            
            % Find valid mice
            target_mice = unique(lbls_full.mouse(idx_target));
            valid_mice = [];
            for m = 1:length(target_mice)
                if sum(idx_target & (lbls_full.mouse == target_mice(m))) >= min_units_per_mouse
                    valid_mice(end+1) = target_mice(m);
                end
            end
            
            n_mice = length(valid_mice);
            
            if n_mice >= 2
                trial_raw_pooled = cell(1, length(target_trials));
                legend_handles = [];
                
                for t = 1:length(target_trials)
                    tr_idx = target_trials(t);
                    mouse_pdfs = nan(n_mice, length(x_eval));
                    raw_t = [];
                    
                    for m = 1:n_mice
                        idx_mouse = idx_target & (lbls_full.mouse == valid_mice(m));
                        target_tensor = tensor_full_z(idx_mouse, :, :); 
                        
                        if tr_idx > size(target_tensor, 3)
                            continue;
                        end
                        
                        % Mean activity per trial: [Neurons x Trials]
                        unit_trial_rates = squeeze(mean(target_tensor, 2, 'omitnan')); 
                        
                        % Extract specific trial
                        trial_rates = unit_trial_rates(:, tr_idx);
                        trial_rates = trial_rates(~isnan(trial_rates));
                        
                        % Compute continuous KDE for this mouse
                        if length(trial_rates) >= 2
                            [f_mouse, ~] = ksdensity(trial_rates, x_eval);
                            mouse_pdfs(m, :) = f_mouse;
                        end
                        raw_t = [raw_t; trial_rates];
                    end
                    
                    trial_raw_pooled{t} = raw_t(~isnan(raw_t));
                    
                    % Average KDE PDFs across mice
                    mu_pdf = mean(mouse_pdfs, 1, 'omitnan');
                    se_pdf = std(mouse_pdfs, 0, 1, 'omitnan') / sqrt(n_mice);
                    
                    h = shadedErrorBar(x_eval, mu_pdf, se_pdf, ...
                        'lineprops', {'-', 'Color', trial_colors{t}, 'LineWidth', 2}, 'patchSaturation', 0.15);
                    legend_handles(end+1) = h.mainLine;
                end
                
                % KS Test: Trial 1 vs Trial 21
                if ~isempty(trial_raw_pooled{1}) && ~isempty(trial_raw_pooled{3})
                    [~, p_ks, ks_stat] = kstest2(trial_raw_pooled{1}, trial_raw_pooled{3});
                    
                    sig_star = ''; 
                    if p_ks < 0.05, sig_star = '*'; end
                    if p_ks < 0.01, sig_star = '**'; end
                    if p_ks < 0.001, sig_star = '***'; end
                    
                    ks_txt = sprintf('KS(Tr1, Tr21):\nD=%.2f, p=%.1e %s', ks_stat, p_ks, sig_star);
                    text(0.05, 0.95, ks_txt, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                        'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7]);
                end
                grid on;
            else
                axis off;
                text(0.5, 0.5, 'Insufficient Mice', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
            end
            
            title(sprintf('%s (N=%d mice)', current_area, n_mice));
            if i_area == 1; ylabel('Probability Density'); end
            xlabel('Mean Z-Scored Activity');
            if i_area == num_areas && n_mice >= 2; legend(legend_handles, trial_names, 'Location', 'northeast'); end
        end
        linkaxes(ax_trials(isgraphics(ax_trials)), 'y');
        xlim(ax_trials(isgraphics(ax_trials)), [x_eval(1), x_eval(end)]);
        title(t_trials, sprintf('%s: %s Mean Activity Distributions by Specific Trial (KDE)', ds_name, current_type_name), 'FontSize', 16);
        
        save_to_svg(regexprep(sprintf('[%s]_MeanDist_Trials_Hierarchical_%s', ds_name, current_type_name), '[\[\]\s:]', '_'));
    end
end
fprintf('--- Hierarchical KDE Mean Activity Distribution Plots Complete ---\n\n');