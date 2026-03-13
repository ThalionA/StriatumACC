function [supermouse_tensor_raw, combined_labels, tensor_info] = buildCombinedTensor(task_data, control_data, task_lps, control_indices_info, cfg)
% Builds the combined 'supermouse' tensor from processed task and/or control data.
% Accepts explicit task_data and control_data structs.
% Robust to missing 'final_neurontypes' fields, explicitly retaining them for task data.
    fprintf('--- Building Combined Tensor ---\n');
    
    % --- Initialization ---
    supermouse_tensor_raw = [];
    combined_labels = struct('mouse_labels_all', [], 'group_labels_all', [], 'area_labels_all', {{}}, 'neurontype_labels_all', []);
    tensor_info = struct('n_animals_task', 0, 'n_animals_control', 0, 'n_animals_total', 0, ...
                         'bins', 0, 'trials_aligned', 0, 'mouse_units_starts', [], 'mouse_units_ends', []);
    
    % Determine which datasets are present
    include_task = ~isempty(task_data);
    include_control = ~isempty(control_data);
    task_group_id = 1;     % Fixed ID for Task
    control_group_id = 2;  % Fixed ID for Control
    
    % --- Pre-Concatenation Field Filtering ---
    task_data_filtered = []; 
    control_data_filtered = [];
    
    if include_task && include_control
        fprintf('  Filtering task and control structs to common fields before concatenation...\n');
        task_fields = fieldnames(task_data);
        control_fields = fieldnames(control_data);
        common_fields = intersect(task_fields, control_fields);
        
        % --- RESCUE 'final_neurontypes' ---
        % If task has it but control doesn't, pad control with empty so we don't lose the field
        if isfield(task_data, 'final_neurontypes') && ~isfield(control_data, 'final_neurontypes')
            for c = 1:length(control_data)
                control_data(c).final_neurontypes = [];
            end
            common_fields{end+1} = 'final_neurontypes';
            control_fields{end+1} = 'final_neurontypes';
            fprintf('    Rescuing "final_neurontypes" from task data (padding control with empty).\n');
        elseif ~isfield(task_data, 'final_neurontypes') && isfield(control_data, 'final_neurontypes')
            for t = 1:length(task_data)
                task_data(t).final_neurontypes = [];
            end
            common_fields{end+1} = 'final_neurontypes';
            task_fields{end+1} = 'final_neurontypes';
            fprintf('    Rescuing "final_neurontypes" from control data (padding task with empty).\n');
        end
        % ----------------------------------
        
        % Check if filtering is actually needed
        if length(common_fields) == length(task_fields) && length(common_fields) == length(control_fields)
             fprintf('    Task and Control structs already have the same fields. No filtering needed.\n');
             task_data_filtered = task_data;
             control_data_filtered = control_data;
        else
            task_fields_to_remove = setdiff(task_fields, common_fields);
            control_fields_to_remove = setdiff(control_fields, common_fields);
            
            fprintf('    Identified %d common fields.\n', length(common_fields));
            
            task_data_filtered = rmfield(task_data, task_fields_to_remove);
            control_data_filtered = rmfield(control_data, control_fields_to_remove);
            fprintf('    Field filtering complete.\n');
            
            % Verification
            required_fields = {'spatial_binned_fr_all'}; 
            if isfield(cfg, 'area_field_map') && ~isempty(cfg.area_field_map)
                required_fields = [required_fields, values(cfg.area_field_map)];
            end
            missing_essential = setdiff(required_fields, common_fields);
            if ~isempty(missing_essential)
                 error('buildCombinedTensor: Essential field(s) "%s" are not present in both task and control data after filtering.', strjoin(missing_essential, ', '));
            end
        end
    elseif include_task 
        fprintf('  Only task data provided. No field filtering needed.\n');
        task_data_filtered = task_data;
        control_data_filtered = [];    
    elseif include_control 
        fprintf('  Only control data provided. No field filtering needed.\n');
        control_data_filtered = control_data; 
        task_data_filtered = [];          
    else
         warning('buildCombinedTensor: No task or control animals provided.');
         return; 
    end
    
    % --- Consolidate Data ---
    XY_all = {};             
    data_struct_all = [];    
    group_ids_all = [];      
    local_indices = [];      
    
    tensor_info.n_animals_task = 0; 
    tensor_info.n_animals_control = 0;
    
    if include_task && ~isempty(task_data_filtered)
        tensor_info.n_animals_task = length(task_data_filtered);
        fprintf('  Including %d task animals (post-filtering).\n', tensor_info.n_animals_task);
        if isfield(task_data_filtered, 'spatial_binned_fr_all')
            XY_all = [XY_all, {task_data_filtered(:).spatial_binned_fr_all}];
        else
             error('buildCombinedTensor: Field "spatial_binned_fr_all" missing from filtered task data.');
        end
        data_struct_all = [data_struct_all; task_data_filtered(:)]; 
        group_ids_all = [group_ids_all; repmat(task_group_id, tensor_info.n_animals_task, 1)];
        local_indices = [local_indices; (1:tensor_info.n_animals_task)']; 
    end
    
    if include_control && ~isempty(control_data_filtered) 
        tensor_info.n_animals_control = length(control_data_filtered);
         fprintf('  Including %d control animals (post-filtering).\n', tensor_info.n_animals_control);
        if isfield(control_data_filtered, 'spatial_binned_fr_all')
           XY_all = [XY_all, {control_data_filtered(:).spatial_binned_fr_all}];
        else
            error('buildCombinedTensor: Field "spatial_binned_fr_all" missing from filtered control data.');
        end
        data_struct_all = [data_struct_all; control_data_filtered(:)]; 
        group_ids_all = [group_ids_all; repmat(control_group_id, tensor_info.n_animals_control, 1)];
         local_indices = [local_indices; (1:tensor_info.n_animals_control)']; 
    end
    
    tensor_info.n_animals_total = tensor_info.n_animals_task + tensor_info.n_animals_control;
    if tensor_info.n_animals_total == 0
        warning('buildCombinedTensor: No animals remain after filtering.');
        return;
    end
    
    % --- Calculate Tensor Dimensions ---
    try
        if isempty(XY_all)
             warning('buildCombinedTensor: No neural data available.');
             return; 
        end
        neuron_counts_per_animal = cellfun(@(xy) size(xy, 1), XY_all);
        all_units_total = sum(neuron_counts_per_animal);
        
        if all_units_total == 0
             warning('buildCombinedTensor: No neurons found in the provided data.');
             return;
        end
        
        mouse_units_cumsum_total = cumsum(neuron_counts_per_animal);
        tensor_info.mouse_units_starts = [1, mouse_units_cumsum_total(1:end-1) + 1]; 
        tensor_info.mouse_units_ends = mouse_units_cumsum_total;
        tensor_info.bins = size(XY_all{1}, 2); 
    catch ME_dim
        error('buildCombinedTensor: Error calculating tensor dimensions: %s', ME_dim.message);
    end
    
    % Determine aligned trials
     try
        n_trials_aligned = numel(cfg.control_epoch_windows{1}) + ...
                           (cfg.control_epoch_windows{2}(2) - cfg.control_epoch_windows{2}(1) + 1) + ... 
                           (cfg.control_epoch_windows{3}(2) - cfg.control_epoch_windows{3}(1) + 1);    
     catch ME_trials
         error('buildCombinedTensor: Error calculating n_trials_aligned. Check cfg.control_epoch_windows format. Error: %s', ME_trials.message);
     end
    tensor_info.trials_aligned = n_trials_aligned;
    
    % --- Initialize the Combined Tensor ---
    supermouse_tensor_raw = nan(all_units_total, tensor_info.bins, tensor_info.trials_aligned);
    fprintf('  Initializing raw tensor: [%d neurons x %d bins x %d aligned trials]\n', ...
        all_units_total, tensor_info.bins, tensor_info.trials_aligned);
    
    % --- Populate Tensor and Create Labels ---
    mouse_labels_all = zeros(all_units_total, 1);
    group_labels_all_out = zeros(all_units_total, 1); 
    area_labels_all = cell(all_units_total, 1);
    neurontype_labels_all = nan(all_units_total, 1); % Initialize with NaN
    
    % Get area fields
    processed_areas = {};
    area_fields = {};
    if isfield(cfg, 'areas_to_include') && isfield(cfg, 'area_field_map')
        processed_areas = cfg.areas_to_include;
        try
            area_fields = values(cfg.area_field_map, processed_areas); 
        catch
             area_fields = {}; 
        end
    end
    
    for i_global_animal = 1:tensor_info.n_animals_total 
         current_data_struct = data_struct_all(i_global_animal);
         current_neural_data = XY_all{i_global_animal}; 
         current_group_id = group_ids_all(i_global_animal);
         current_local_idx = local_indices(i_global_animal); 
         
         start_idx = tensor_info.mouse_units_starts(i_global_animal);
         end_idx = tensor_info.mouse_units_ends(i_global_animal);
         n_units_mouse = end_idx - start_idx + 1;
         
         if n_units_mouse == 0; continue; end 
         
         % 1. Populate Neural Data (Tensor)
         if current_group_id == task_group_id
             % Task Logic
             fprintf('    Populating Task animal %d (Global %d)...\n', current_local_idx, i_global_animal);
             if current_local_idx > length(task_lps)
                  continue; 
             end
             task_lp = task_lps{current_local_idx};
             task_total_trials = size(current_neural_data, 3);
             
             idx_early = cfg.control_epoch_windows{1};
             idx_pre   = task_lp + (cfg.control_epoch_windows{2}(1) : cfg.control_epoch_windows{2}(2));
             idx_post  = task_lp + (cfg.control_epoch_windows{3}(1) : cfg.control_epoch_windows{3}(2));
             
             indices_valid = all(idx_early >= 1) && all(idx_early <= task_total_trials) && ...
                             all(idx_pre >= 1)   && all(idx_pre <= task_total_trials) && ...
                             all(idx_post >= 1)  && all(idx_post <= task_total_trials);
             
             if indices_valid
                 try
                     supermouse_tensor_raw(start_idx:end_idx, :, 1:10)  = current_neural_data(:, :, idx_early);
                     supermouse_tensor_raw(start_idx:end_idx, :, 11:20) = current_neural_data(:, :, idx_pre);
                     supermouse_tensor_raw(start_idx:end_idx, :, 21:30) = current_neural_data(:, :, idx_post);
                 catch ME_populate_task
                      fprintf('      Warning: Error populating tensor for Task mouse %d: %s. Leaving NaNs.\n', current_local_idx, ME_populate_task.message);
                      supermouse_tensor_raw(start_idx:end_idx, :, :) = nan;
                 end
             else
                 fprintf('      Warning: Task mouse %d lacks trials around LP (%d). Leaving NaNs.\n', current_local_idx, task_lp);
                 supermouse_tensor_raw(start_idx:end_idx, :, :) = nan;
             end
             
         elseif current_group_id == control_group_id
             % Control Logic
              fprintf('    Populating Control animal %d (Global %d)...\n', current_local_idx, i_global_animal);
              try
                 indices_ctrl_valid = all(control_indices_info.early >= 1) && all(control_indices_info.early <= size(current_neural_data, 3)) && ...
                                      all(control_indices_info.pre_ref >= 1) && all(control_indices_info.pre_ref <= size(current_neural_data, 3)) && ...
                                      all(control_indices_info.post_ref >= 1) && all(control_indices_info.post_ref <= size(current_neural_data, 3));
                 if indices_ctrl_valid
                     supermouse_tensor_raw(start_idx:end_idx, :, 1:10)  = current_neural_data(:, :, control_indices_info.early);
                     supermouse_tensor_raw(start_idx:end_idx, :, 11:20) = current_neural_data(:, :, control_indices_info.pre_ref);
                     supermouse_tensor_raw(start_idx:end_idx, :, 21:30) = current_neural_data(:, :, control_indices_info.post_ref);
                 else
                      fprintf('      Warning: Control mouse %d lacks trials for predefined indices. Leaving NaNs.\n', current_local_idx);
                      supermouse_tensor_raw(start_idx:end_idx, :, :) = nan; 
                 end
              catch
                   supermouse_tensor_raw(start_idx:end_idx, :, :) = nan; 
              end
         end
         
        % 2. Assign Labels (Common to Task & Control)
        mouse_labels_all(start_idx:end_idx) = i_global_animal; 
        group_labels_all_out(start_idx:end_idx) = current_group_id; 
        
        % A. Area Labels
         if isempty(area_fields) 
              area_labels_all(start_idx:end_idx) = {'Unknown'};
         else
             for n_idx = 1:n_units_mouse
                 neuron_global_idx = start_idx + n_idx - 1;
                 assigned_area = false;
                 for area_k = 1:length(processed_areas)
                     current_area_field = area_fields{area_k};
                     if isfield(current_data_struct, current_area_field) && ...
                        n_idx <= length(current_data_struct.(current_area_field)) && ...
                        current_data_struct.(current_area_field)(n_idx)
                             area_labels_all{neuron_global_idx} = processed_areas{area_k};
                             assigned_area = true;
                             break; 
                     end
                 end 
                 if ~assigned_area
                     area_labels_all{neuron_global_idx} = 'Unknown'; 
                 end
             end 
         end 
         
         % B. Neuron Type Labels (Robust Check)
         if isfield(current_data_struct, 'final_neurontypes') && ~isempty(current_data_struct.final_neurontypes)
             types_data = current_data_struct.final_neurontypes;
             [rows, cols] = size(types_data);
             
             % Ensure dimension matching to avoid crashes
             if rows == n_units_mouse
                 if cols >= 5
                     neurontype_labels_all(start_idx:end_idx) = types_data(:, 5);
                 elseif cols == 1
                     neurontype_labels_all(start_idx:end_idx) = types_data(:, 1);
                 else
                     fprintf('      Warning: "final_neurontypes" has unexpected columns (%d) for animal %d. Leaving NaN.\n', cols, i_global_animal);
                 end
             else
                 fprintf('      Warning: Mismatch in neuron type rows (%d) vs units (%d) for animal %d. Leaving NaN.\n', rows, n_units_mouse, i_global_animal);
             end
         else
             % Field missing or empty: implicitly leaves as NaN (from initialization)
         end
         
    end % End loop through global animals
    
    % --- Assign labels to output structure ---
    combined_labels.mouse_labels_all = mouse_labels_all;
    combined_labels.group_labels_all = group_labels_all_out; 
    combined_labels.area_labels_all = area_labels_all;
    combined_labels.neurontype_labels_all = neurontype_labels_all;
    
    fprintf('--- Combined Tensor Build Complete ---\n');
end