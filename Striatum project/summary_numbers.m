num_units = cellfun(@(x) size(x, 2), {task_data_raw(:).is_dms});

mean_n_units = mean(num_units);
sem_n_units = sem(num_units);

n_total_units = sum(num_units);

%% 

num_units_dms = cellfun(@(x) sum(x), {task_data_raw(:).is_dms});
mean_n_units_dms = mean(num_units_dms);
sem_n_units_dms = sem(num_units_dms);
n_total_units_dms = sum(num_units_dms);


num_units_dls = cellfun(@(x) sum(x), {task_data_raw(:).is_dls});
mean_n_units_dls = mean(num_units_dls);
sem_n_units_dls = sem(num_units_dls);
n_total_units_dls = sum(num_units_dls);

num_units_acc = cellfun(@(x) sum(x), {task_data_raw(:).is_acc});
mean_n_units_acc = mean(num_units_acc);
sem_n_units_acc = sem(num_units_acc);
n_total_units_acc = sum(num_units_acc);

% ================= Extract Summary Metrics (task_data_raw) =================
if ~isempty(task_data_raw)
    n_animals_raw = numel(task_data_raw);
    
    % Initialize storage arrays
    total_units   = zeros(n_animals_raw, 1);
    units_dms     = zeros(n_animals_raw, 1);
    units_dls     = zeros(n_animals_raw, 1);
    units_acc     = zeros(n_animals_raw, 1);
    
    units_msn     = zeros(n_animals_raw, 1);
    units_fsn     = zeros(n_animals_raw, 1);
    units_tan     = zeros(n_animals_raw, 1);
    units_unclass = zeros(n_animals_raw, 1);
    
    for i = 1:n_animals_raw
        % Area counts
        is_dms = task_data_raw(i).is_dms;
        is_dls = task_data_raw(i).is_dls;
        is_acc = task_data_raw(i).is_acc;
        
        total_units(i) = length(is_dms); % Assumes logical arrays are same length
        units_dms(i)   = sum(is_dms);
        units_dls(i)   = sum(is_dls);
        units_acc(i)   = sum(is_acc);
        
        % Neuron type counts (1=MSN, 2=FSN, 3=TAN)
        if isfield(task_data_raw(i), 'final_neurontypes') && ~isempty(task_data_raw(i).final_neurontypes)
            ntypes_raw = task_data_raw(i).final_neurontypes;
            [~, cols] = size(ntypes_raw);
            
            % Extract the correct column based on the same logic as buildCombinedTensor
            if cols >= 5
                ntypes = ntypes_raw(:, 5);
            elseif cols == 1
                ntypes = ntypes_raw(:, 1);
            else
                ntypes = nan(size(ntypes_raw, 1), 1); % Unclassified
            end
            
            units_msn(i) = sum(ntypes == 1);
            units_fsn(i) = sum(ntypes == 2);
            units_tan(i) = sum(ntypes == 3);
            units_unclass(i) = sum(isnan(ntypes) | (ntypes ~= 1 & ntypes ~= 2 & ntypes ~= 3));
        else
            % If field is missing entirely, all units for this mouse are unclassified
            units_unclass(i) = total_units(i);
        end
    end
    
    % Anonymous function for quick string formatting (Mean +/- SEM)
    fmt_stats = @(x) sprintf('%7.1f +/- %5.1f', mean(x), std(x)/sqrt(n_animals_raw));
    
    fprintf('\n================ DATA SUMMARY (Raw, Pre-subsampling) ================\n');
    fprintf('Total Animals: %d\n\n', n_animals_raw);
    
    fprintf('--- Units per Animal (Mean +/- SEM) [Total across all animals] ---\n');
    fprintf('Total Units : %s  [Sum: %d]\n', fmt_stats(total_units), sum(total_units));
    fprintf('DMS Units   : %s  [Sum: %d]\n', fmt_stats(units_dms), sum(units_dms));
    fprintf('DLS Units   : %s  [Sum: %d]\n', fmt_stats(units_dls), sum(units_dls));
    fprintf('ACC Units   : %s  [Sum: %d]\n', fmt_stats(units_acc), sum(units_acc));
    
    fprintf('\n--- Neuron Types per Animal (Mean +/- SEM) [Total] ---\n');
    fprintf('MSN (Type 1): %s  [Sum: %d]\n', fmt_stats(units_msn), sum(units_msn));
    fprintf('FSN (Type 2): %s  [Sum: %d]\n', fmt_stats(units_fsn), sum(units_fsn));
    fprintf('TAN (Type 3): %s  [Sum: %d]\n', fmt_stats(units_tan), sum(units_tan));
    fprintf('Unclassified: %s  [Sum: %d]\n', fmt_stats(units_unclass), sum(units_unclass));
    fprintf('=====================================================================\n\n');
end

%% ================= Extended Summary Metrics =================
if ~isempty(task_data_raw)
    % Initialize arrays for extended metrics
    total_trials = zeros(n_animals_raw, 1);
    
    fr_msn = []; fr_fsn = []; fr_tan = [];
    fr_dms = []; fr_dls = []; fr_acc = [];
    
    for i = 1:n_animals_raw
        % 1. Trial counts (assuming spatial_binned_data.licks is [bins x trials])
        if isfield(task_data_raw(i), 'spatial_binned_data') && isfield(task_data_raw(i).spatial_binned_data, 'licks')
            total_trials(i) = size(task_data_raw(i).spatial_binned_data.licks, 1);
        end
        
        % 2. Firing rates setup
        if isfield(task_data_raw(i), 'spatial_binned_fr_all')
            % Average FR across bins and trials for each neuron
            % spatial_binned_fr_all is [neurons x bins x trials]
            mean_fr_per_neuron = mean(task_data_raw(i).spatial_binned_fr_all, [2, 3], 'omitnan');
            
            % Area Firing Rates
            if isfield(task_data_raw(i), 'is_dms'), fr_dms = [fr_dms; mean_fr_per_neuron(task_data_raw(i).is_dms)]; end
            if isfield(task_data_raw(i), 'is_dls'), fr_dls = [fr_dls; mean_fr_per_neuron(task_data_raw(i).is_dls)]; end
            if isfield(task_data_raw(i), 'is_acc'), fr_acc = [fr_acc; mean_fr_per_neuron(task_data_raw(i).is_acc)]; end
            
            % Cell Type Firing Rates
            if isfield(task_data_raw(i), 'final_neurontypes')
                ntypes_raw = task_data_raw(i).final_neurontypes;
                [~, cols] = size(ntypes_raw);
                if cols >= 5
                    ntypes = ntypes_raw(:, 5);
                elseif cols == 1
                    ntypes = ntypes_raw(:, 1);
                else
                    ntypes = nan(size(ntypes_raw, 1), 1);
                end
                
                fr_msn = [fr_msn; mean_fr_per_neuron(ntypes == 1)];
                fr_fsn = [fr_fsn; mean_fr_per_neuron(ntypes == 2)];
                fr_tan = [fr_tan; mean_fr_per_neuron(ntypes == 3)];
            end
        end
    end
    
    fprintf('\n================ BEHAVIOR & FIRING RATE SUMMARY ================\n');
    fprintf('Total Trials per Session : %7.1f +/- %5.1f\n\n', mean(total_trials), std(total_trials)/sqrt(n_animals_raw));
    
    fprintf('--- Global Firing Rates by Area (Hz, Mean +/- SEM across neurons) ---\n');
    fprintf('DMS Units : %6.2f +/- %5.2f\n', mean(fr_dms), std(fr_dms)/sqrt(max(1,length(fr_dms))));
    fprintf('DLS Units : %6.2f +/- %5.2f\n', mean(fr_dls), std(fr_dls)/sqrt(max(1,length(fr_dls))));
    fprintf('ACC Units : %6.2f +/- %5.2f\n\n', mean(fr_acc), std(fr_acc)/sqrt(max(1,length(fr_acc))));
    
    fprintf('--- Global Firing Rates by Putative Cell Type (Hz, Mean +/- SEM) ---\n');
    fprintf('MSN (Type 1): %6.2f +/- %5.2f\n', mean(fr_msn), std(fr_msn)/sqrt(max(1,length(fr_msn))));
    fprintf('FSN (Type 2): %6.2f +/- %5.2f\n', mean(fr_fsn), std(fr_fsn)/sqrt(max(1,length(fr_fsn))));
    fprintf('TAN (Type 3): %6.2f +/- %5.2f\n', mean(fr_tan), std(fr_tan)/sqrt(max(1,length(fr_tan))));
    fprintf('=====================================================================\n\n');
end

%% ================= Extract Summary Metrics (control_data_raw) =================

% 1. Check if control data is already loaded; if not, attempt to load it
if ~exist('control_data_raw', 'var') || isempty(control_data_raw)
    if exist('cfg', 'var') && isfield(cfg, 'control_data_file') && isfile(cfg.control_data_file)
        fprintf('\nLoading raw control data from %s for summary metrics...\n', cfg.control_data_file);
        ctrl_load = load(cfg.control_data_file, 'preprocessed_data');
        if isfield(ctrl_load, 'preprocessed_data')
            control_data_raw = ctrl_load.preprocessed_data;
        else
            warning('Variable "preprocessed_data" not found in the control data file.');
            control_data_raw = [];
        end
    else
        fprintf('\nControl data not found in workspace, and cfg.control_data_file is missing or invalid. Skipping control summary.\n');
        control_data_raw = [];
    end
end

% 2. Extract and print metrics if data is available
if ~isempty(control_data_raw)
    n_animals_ctrl = numel(control_data_raw);
    
    % Initialize storage arrays for unit counts
    ctrl_total_units   = zeros(n_animals_ctrl, 1);
    ctrl_units_dms     = zeros(n_animals_ctrl, 1);
    ctrl_units_dls     = zeros(n_animals_ctrl, 1);
    ctrl_units_acc     = zeros(n_animals_ctrl, 1);
    
    ctrl_units_msn     = zeros(n_animals_ctrl, 1);
    ctrl_units_fsn     = zeros(n_animals_ctrl, 1);
    ctrl_units_tan     = zeros(n_animals_ctrl, 1);
    ctrl_units_unclass = zeros(n_animals_ctrl, 1);
    
    % Initialize arrays for behavior and firing rates
    ctrl_total_trials = zeros(n_animals_ctrl, 1);
    ctrl_fr_msn = []; ctrl_fr_fsn = []; ctrl_fr_tan = [];
    ctrl_fr_dms = []; ctrl_fr_dls = []; ctrl_fr_acc = [];
    
    for i = 1:n_animals_ctrl
        % --- Area counts ---
        is_dms = control_data_raw(i).is_dms;
        is_dls = control_data_raw(i).is_dls;
        is_acc = control_data_raw(i).is_acc;
        
        ctrl_total_units(i) = length(is_dms); 
        ctrl_units_dms(i)   = sum(is_dms);
        ctrl_units_dls(i)   = sum(is_dls);
        ctrl_units_acc(i)   = sum(is_acc);
        
        % --- Neuron type counts (Robust extraction) ---
        if isfield(control_data_raw(i), 'final_neurontypes') && ~isempty(control_data_raw(i).final_neurontypes)
            ntypes_raw = control_data_raw(i).final_neurontypes;
            [~, cols] = size(ntypes_raw);
            
            if cols >= 5
                ntypes = ntypes_raw(:, 5);
            elseif cols == 1
                ntypes = ntypes_raw(:, 1);
            else
                ntypes = nan(size(ntypes_raw, 1), 1);
            end
            
            ctrl_units_msn(i) = sum(ntypes == 1);
            ctrl_units_fsn(i) = sum(ntypes == 2);
            ctrl_units_tan(i) = sum(ntypes == 3);
            ctrl_units_unclass(i) = sum(isnan(ntypes) | (ntypes ~= 1 & ntypes ~= 2 & ntypes ~= 3));
        else
            ctrl_units_unclass(i) = ctrl_total_units(i);
            ntypes = nan(ctrl_total_units(i), 1); % Dummy array for the FR calculation below
        end
        
        % --- Trial counts & Firing Rates ---
        if isfield(control_data_raw(i), 'spatial_binned_fr_all')
            % Use the 3rd dimension of the tensor for trial counts
            ctrl_total_trials(i) = size(control_data_raw(i).spatial_binned_fr_all, 3);
            
            % Average FR across bins and trials for each neuron
            mean_fr_per_neuron = mean(control_data_raw(i).spatial_binned_fr_all, [2, 3], 'omitnan');
            
            % Area Firing Rates
            if isfield(control_data_raw(i), 'is_dms'), ctrl_fr_dms = [ctrl_fr_dms; mean_fr_per_neuron(control_data_raw(i).is_dms)]; end
            if isfield(control_data_raw(i), 'is_dls'), ctrl_fr_dls = [ctrl_fr_dls; mean_fr_per_neuron(control_data_raw(i).is_dls)]; end
            if isfield(control_data_raw(i), 'is_acc'), ctrl_fr_acc = [ctrl_fr_acc; mean_fr_per_neuron(control_data_raw(i).is_acc)]; end
            
            % Cell Type Firing Rates
            ctrl_fr_msn = [ctrl_fr_msn; mean_fr_per_neuron(ntypes == 1)];
            ctrl_fr_fsn = [ctrl_fr_fsn; mean_fr_per_neuron(ntypes == 2)];
            ctrl_fr_tan = [ctrl_fr_tan; mean_fr_per_neuron(ntypes == 3)];
        end
    end
    
    % Anonymous function for quick string formatting (Mean +/- SEM)
    fmt_stats = @(x) sprintf('%7.1f +/- %5.1f', mean(x), std(x)/sqrt(n_animals_ctrl));
    fmt_fr    = @(x) sprintf('%6.2f +/- %5.2f', mean(x), std(x)/sqrt(max(1,length(x))));
    
    fprintf('\n================ CONTROL DATA SUMMARY (Raw, Pre-subsampling) ================\n');
    fprintf('Total Control Animals: %d\n\n', n_animals_ctrl);
    
    fprintf('--- Units per Animal (Mean +/- SEM) [Total across all animals] ---\n');
    fprintf('Total Units : %s  [Sum: %d]\n', fmt_stats(ctrl_total_units), sum(ctrl_total_units));
    fprintf('DMS Units   : %s  [Sum: %d]\n', fmt_stats(ctrl_units_dms), sum(ctrl_units_dms));
    fprintf('DLS Units   : %s  [Sum: %d]\n', fmt_stats(ctrl_units_dls), sum(ctrl_units_dls));
    fprintf('ACC Units   : %s  [Sum: %d]\n', fmt_stats(ctrl_units_acc), sum(ctrl_units_acc));
    
    fprintf('\n--- Neuron Types per Animal (Mean +/- SEM) [Total] ---\n');
    fprintf('MSN (Type 1): %s  [Sum: %d]\n', fmt_stats(ctrl_units_msn), sum(ctrl_units_msn));
    fprintf('FSN (Type 2): %s  [Sum: %d]\n', fmt_stats(ctrl_units_fsn), sum(ctrl_units_fsn));
    fprintf('TAN (Type 3): %s  [Sum: %d]\n', fmt_stats(ctrl_units_tan), sum(ctrl_units_tan));
    fprintf('Unclassified: %s  [Sum: %d]\n', fmt_stats(ctrl_units_unclass), sum(ctrl_units_unclass));
    
    fprintf('\n--- Behavior & Global Firing Rates ---\n');
    fprintf('Total Trials per Session : %s\n\n', fmt_stats(ctrl_total_trials));
    
    fprintf('--- Global Firing Rates by Area (Hz, Mean +/- SEM across neurons) ---\n');
    fprintf('DMS Units : %s\n', fmt_fr(ctrl_fr_dms));
    fprintf('DLS Units : %s\n', fmt_fr(ctrl_fr_dls));
    fprintf('ACC Units : %s\n\n', fmt_fr(ctrl_fr_acc));
    
    fprintf('--- Global Firing Rates by Putative Cell Type (Hz, Mean +/- SEM) ---\n');
    fprintf('MSN (Type 1): %s\n', fmt_fr(ctrl_fr_msn));
    fprintf('FSN (Type 2): %s\n', fmt_fr(ctrl_fr_fsn));
    fprintf('TAN (Type 3): %s\n', fmt_fr(ctrl_fr_tan));
    fprintf('=============================================================================\n\n');
end

%% ================= Extract Summary Metrics (control2_data_raw) =================

% 1. Check if control2 data is already loaded; if not, attempt to load it
if ~exist('control2_data_raw', 'var') || isempty(control2_data_raw)
    if exist('cfg', 'var') && isfield(cfg, 'control2_data_file') && isfile(cfg.control2_data_file)
        fprintf('\nLoading raw control2 data from %s for summary metrics...\n', cfg.control2_data_file);
        ctrl2_load = load(cfg.control2_data_file, 'preprocessed_data');
        if isfield(ctrl2_load, 'preprocessed_data')
            control2_data_raw = ctrl2_load.preprocessed_data;
        else
            warning('Variable "preprocessed_data" not found in the control2 data file.');
            control2_data_raw = [];
        end
    else
        fprintf('\nControl2 data not found in workspace, and cfg.control2_data_file is missing or invalid. Skipping control2 summary.\n');
        control2_data_raw = [];
    end
end

% 2. Extract and print metrics if data is available
if ~isempty(control2_data_raw)
    n_animals_ctrl2 = numel(control2_data_raw);
    
    % Initialize storage arrays for unit counts
    ctrl2_total_units   = zeros(n_animals_ctrl2, 1);
    ctrl2_units_dms     = zeros(n_animals_ctrl2, 1);
    ctrl2_units_dls     = zeros(n_animals_ctrl2, 1);
    ctrl2_units_acc     = zeros(n_animals_ctrl2, 1);
    
    ctrl2_units_msn     = zeros(n_animals_ctrl2, 1);
    ctrl2_units_fsn     = zeros(n_animals_ctrl2, 1);
    ctrl2_units_tan     = zeros(n_animals_ctrl2, 1);
    ctrl2_units_unclass = zeros(n_animals_ctrl2, 1);
    
    % Initialize arrays for behavior and firing rates
    ctrl2_total_trials = zeros(n_animals_ctrl2, 1);
    ctrl2_fr_msn = []; ctrl2_fr_fsn = []; ctrl2_fr_tan = [];
    ctrl2_fr_dms = []; ctrl2_fr_dls = []; ctrl2_fr_acc = [];
    
    for i = 1:n_animals_ctrl2
        % --- Area counts ---
        is_dms = control2_data_raw(i).is_dms;
        is_dls = control2_data_raw(i).is_dls;
        is_acc = control2_data_raw(i).is_acc;
        
        ctrl2_total_units(i) = length(is_dms); 
        ctrl2_units_dms(i)   = sum(is_dms);
        ctrl2_units_dls(i)   = sum(is_dls);
        ctrl2_units_acc(i)   = sum(is_acc);
        
        % --- Neuron type counts (Robust extraction) ---
        if isfield(control2_data_raw(i), 'final_neurontypes') && ~isempty(control2_data_raw(i).final_neurontypes)
            ntypes_raw = control2_data_raw(i).final_neurontypes;
            [~, cols] = size(ntypes_raw);
            
            if cols >= 5
                ntypes = ntypes_raw(:, 5);
            elseif cols == 1
                ntypes = ntypes_raw(:, 1);
            else
                ntypes = nan(size(ntypes_raw, 1), 1);
            end
            
            ctrl2_units_msn(i) = sum(ntypes == 1);
            ctrl2_units_fsn(i) = sum(ntypes == 2);
            ctrl2_units_tan(i) = sum(ntypes == 3);
            ctrl2_units_unclass(i) = sum(isnan(ntypes) | (ntypes ~= 1 & ntypes ~= 2 & ntypes ~= 3));
        else
            ctrl2_units_unclass(i) = ctrl2_total_units(i);
            ntypes = nan(ctrl2_total_units(i), 1); % Dummy array for the FR calculation below
        end
        
        % --- Trial counts & Firing Rates ---
        if isfield(control2_data_raw(i), 'spatial_binned_fr_all')
            % Use the 3rd dimension of the tensor for trial counts
            ctrl2_total_trials(i) = size(control2_data_raw(i).spatial_binned_fr_all, 3);
            
            % Average FR across bins and trials for each neuron
            mean_fr_per_neuron = mean(control2_data_raw(i).spatial_binned_fr_all, [2, 3], 'omitnan');
            
            % Area Firing Rates
            if isfield(control2_data_raw(i), 'is_dms'), ctrl2_fr_dms = [ctrl2_fr_dms; mean_fr_per_neuron(control2_data_raw(i).is_dms)]; end
            if isfield(control2_data_raw(i), 'is_dls'), ctrl2_fr_dls = [ctrl2_fr_dls; mean_fr_per_neuron(control2_data_raw(i).is_dls)]; end
            if isfield(control2_data_raw(i), 'is_acc'), ctrl2_fr_acc = [ctrl2_fr_acc; mean_fr_per_neuron(control2_data_raw(i).is_acc)]; end
            
            % Cell Type Firing Rates
            ctrl2_fr_msn = [ctrl2_fr_msn; mean_fr_per_neuron(ntypes == 1)];
            ctrl2_fr_fsn = [ctrl2_fr_fsn; mean_fr_per_neuron(ntypes == 2)];
            ctrl2_fr_tan = [ctrl2_fr_tan; mean_fr_per_neuron(ntypes == 3)];
        end
    end
    
    % Anonymous function for quick string formatting (Mean +/- SEM)
    fmt_stats = @(x) sprintf('%7.1f +/- %5.1f', mean(x), std(x)/sqrt(n_animals_ctrl2));
    fmt_fr    = @(x) sprintf('%6.2f +/- %5.2f', mean(x), std(x)/sqrt(max(1,length(x))));
    
    fprintf('\n================ CONTROL2 DATA SUMMARY (Raw, Pre-subsampling) ================\n');
    fprintf('Total Control2 Animals: %d\n\n', n_animals_ctrl2);
    
    fprintf('--- Units per Animal (Mean +/- SEM) [Total across all animals] ---\n');
    fprintf('Total Units : %s  [Sum: %d]\n', fmt_stats(ctrl2_total_units), sum(ctrl2_total_units));
    fprintf('DMS Units   : %s  [Sum: %d]\n', fmt_stats(ctrl2_units_dms), sum(ctrl2_units_dms));
    fprintf('DLS Units   : %s  [Sum: %d]\n', fmt_stats(ctrl2_units_dls), sum(ctrl2_units_dls));
    fprintf('ACC Units   : %s  [Sum: %d]\n', fmt_stats(ctrl2_units_acc), sum(ctrl2_units_acc));
    
    fprintf('\n--- Neuron Types per Animal (Mean +/- SEM) [Total] ---\n');
    fprintf('MSN (Type 1): %s  [Sum: %d]\n', fmt_stats(ctrl2_units_msn), sum(ctrl2_units_msn));
    fprintf('FSN (Type 2): %s  [Sum: %d]\n', fmt_stats(ctrl2_units_fsn), sum(ctrl2_units_fsn));
    fprintf('TAN (Type 3): %s  [Sum: %d]\n', fmt_stats(ctrl2_units_tan), sum(ctrl2_units_tan));
    fprintf('Unclassified: %s  [Sum: %d]\n', fmt_stats(ctrl2_units_unclass), sum(ctrl2_units_unclass));
    
    fprintf('\n--- Behavior & Global Firing Rates ---\n');
    fprintf('Total Trials per Session : %s\n\n', fmt_stats(ctrl2_total_trials));
    
    fprintf('--- Global Firing Rates by Area (Hz, Mean +/- SEM across neurons) ---\n');
    fprintf('DMS Units : %s\n', fmt_fr(ctrl2_fr_dms));
    fprintf('DLS Units : %s\n', fmt_fr(ctrl2_fr_dls));
    fprintf('ACC Units : %s\n\n', fmt_fr(ctrl2_fr_acc));
    
    fprintf('--- Global Firing Rates by Putative Cell Type (Hz, Mean +/- SEM) ---\n');
    fprintf('MSN (Type 1): %s\n', fmt_fr(ctrl2_fr_msn));
    fprintf('FSN (Type 2): %s\n', fmt_fr(ctrl2_fr_fsn));
    fprintf('TAN (Type 3): %s\n', fmt_fr(ctrl2_fr_tan));
    fprintf('==============================================================================\n\n');
end