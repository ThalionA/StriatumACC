%% CCA_Striatum_Spatial_Final_Optimized.m
% Spatial CCA Analysis for Striatal/ACC Dataset
%
% DESCRIPTION:
% Performs Canonical Correlation Analysis (CCA) to quantify shared 
% variance and lead/lag relationships (Precession) between DMS, DLS, and ACC.
% Integrates with the custom cfg-based data loading pipeline.
%
% OUTPUTS:
% - Incremental saving of .mat files
% - SVG Figures with rmANOVA + paired Real vs Shuff tests + Filtered Networks

%% 1. CONFIGURATION & PARAMETERS
clear; clc; close all;

% --- Data Files & Analysis Selection ---
cfg.task_data_file = "preprocessed_data.mat";
cfg.control_data_file = "preprocessed_data_control.mat";

% --- Analysis Selection ---
cfg.analysis_mode = 'task_only'; 
cfg.areas_to_include = {'DMS', 'DLS', 'ACC'}; 

% Define mapping from area names to field names in the data struct
cfg.area_field_map = containers.Map(...
    {'DMS', 'DLS', 'ACC'}, ...
    {'is_dms', 'is_dls', 'is_acc'} ...
    );

% --- Processing Parameters ---
cfg.control_epoch_method = 'fixed_trial'; 
cfg.control_fixed_ref_trial = 40;
cfg.control_epoch_windows = {1:10, [-10, -1], [1, 10]}; 
cfg.task_lp_zscore_threshold = -2;
cfg.task_lp_window_length = 10; 
cfg.task_lp_min_consecutive = 7;

% --- CCA / PCA Parameters ---
pca_selection_method = 'variance'; 
pca_variance_threshold = 70;       
n_components_reduced = 3;          
num_ccs_analyze = 3;               
n_trials_window = -3:3;            
n_bins_window = -3:3;              
n_shuffles = 250;                   
max_shift_bins = 2;                
min_units_per_region = 5;
TRUNCATE_AT_DISENGAGEMENT = true;

% --- Plotting / Spatial Parameters ---
landmarks = [20, 25];              

% --- Save Path ---
save_dir = '/Users/theoamvr/Desktop/Experiments/StriatumACC/Striatum project/CCA_Results/';
if ~exist(save_dir, 'dir'), mkdir(save_dir); end
current_date = datestr(now, 'yyyy_mm_dd');
save_path = fullfile(save_dir, sprintf('Striatum_CCA_Results_%s.mat', current_date));

% --- Parallel Pool ---
if isempty(gcp('nocreate'))
    parpool; 
end

%% 2. DATA LOADING & FILTERING
fprintf('--- Loading and Filtering Data ---\n');
if isfile(cfg.task_data_file)
    fprintf('Loading Task data from: %s\n', cfg.task_data_file);
    loaded_data = load(cfg.task_data_file, "preprocessed_data");
    
    task_data_raw = filterDataByArea(loaded_data.preprocessed_data, cfg.areas_to_include, cfg.area_field_map);
    fprintf('  Filtered Task data to %d areas: %s\n', numel(cfg.areas_to_include), strjoin(cfg.areas_to_include, ', '));
    
    fprintf('Processing Task data to find learning points...\n');
    [task_data, learning_points_task, avg_learning_point, aligned_lick_errors] = processTaskData(task_data_raw, cfg);
    fprintf('  Task data processed for inclusion in CCA analysis.\n');
else
    error('Required Task data file not found: %s', cfg.task_data_file);
end

if isempty(task_data)
    error('No valid task data remaining after filtering and processing.');
end

n_animals = size(task_data, 2);
area_pairs_to_analyze = {'DMS', 'DLS'; 'DMS', 'ACC'; 'DLS', 'ACC'};
n_pairs = size(area_pairs_to_analyze, 1);

% Format Learning Points to vector
learning_points = nan(1, n_animals);
for i=1:n_animals
    if ~isempty(learning_points_task{i})
        learning_points(i) = learning_points_task{i};
    end
end

is_learner = ~isnan(learning_points);
if any(is_learner)
    mean_lp = round(mean(learning_points(is_learner)));
else
    mean_lp = NaN; 
end
analysis_lp = learning_points;
analysis_lp(~is_learner) = mean_lp; 
fprintf('Identified %d Learners and %d Non-Learners. Mean LP used for yoking: %d\n', sum(is_learner), sum(~is_learner), mean_lp);

%% 3. INITIALIZE RESULTS STRUCTURE
group_results = struct('pair_name', cell(n_pairs, 1), ...
                       'all_bins_corr', cell(n_pairs, 1), ...
                       'all_bins_corr_shuff', cell(n_pairs, 1), ...
                       'all_bins_precession_idx', cell(n_pairs, 1), ...
                       'all_bins_precession_idx_shuff', cell(n_pairs, 1), ...
                       'all_bins_precession_curve', cell(n_pairs, 1), ...
                       'all_bins_precession_curve_shuff', cell(n_pairs, 1), ...
                       'trial_corr_early', cell(n_pairs, 1), ...
                       'trial_corr_pre', cell(n_pairs, 1), ...
                       'trial_corr_post', cell(n_pairs, 1), ...
                       'trial_corr_early_shuff', cell(n_pairs, 1), ...
                       'trial_corr_pre_shuff', cell(n_pairs, 1), ...
                       'trial_corr_post_shuff', cell(n_pairs, 1), ...
                       'trial_precession_early_idx', cell(n_pairs, 1), ... 
                       'trial_precession_pre_idx', cell(n_pairs, 1), ...   
                       'trial_precession_post_idx', cell(n_pairs, 1), ...
                       'trial_precession_early_idx_shuff', cell(n_pairs, 1), ...
                       'trial_precession_pre_idx_shuff', cell(n_pairs, 1), ...
                       'trial_precession_post_idx_shuff', cell(n_pairs, 1), ...
                       'trial_precession_early_curve', cell(n_pairs, 1), ... 
                       'trial_precession_pre_curve', cell(n_pairs, 1), ...   
                       'trial_precession_post_curve', cell(n_pairs, 1), ...
                       'trial_precession_early_curve_shuff', cell(n_pairs, 1), ...
                       'trial_precession_pre_curve_shuff', cell(n_pairs, 1), ...
                       'trial_precession_post_curve_shuff', cell(n_pairs, 1));
                   
for ipair = 1:n_pairs
    group_results(ipair).pair_name = sprintf('%s-%s', area_pairs_to_analyze{ipair, 1}, area_pairs_to_analyze{ipair, 2});
end

%% 4. MAIN ANALYSIS LOOP
% Bundle current parameters for tracking and comparison
current_config = struct(...
    'pca_selection_method', pca_selection_method, ...
    'pca_variance_threshold', pca_variance_threshold, ...
    'n_components_reduced', n_components_reduced, ...
    'num_ccs_analyze', num_ccs_analyze, ...
    'max_shift_bins', max_shift_bins, ...
    'min_units_per_region', min_units_per_region, ...
    'n_shuffles', n_shuffles);

existing_files = dir(fullfile(save_dir, 'Striatum_CCA_Results_*.mat'));
if ~isempty(existing_files)
    [~, latest_idx] = max([existing_files.datenum]);
    load_target = fullfile(save_dir, existing_files(latest_idx).name);
    
    vars_in_file = whos('-file', load_target);
    if ismember('saved_config', {vars_in_file.name})
        load(load_target, 'group_results', 'is_learner', 'analysis_lp', 'saved_config');
        fprintf('Found existing results. Loading: %s\n', existing_files(latest_idx).name);
    else
        load(load_target, 'group_results', 'is_learner', 'analysis_lp');
        fprintf('Found existing results (Legacy). Loading: %s\n', existing_files(latest_idx).name);
    end
else
    fprintf('\n--- Starting CCA Analysis ---\n');
    n_shifts = 2 * max_shift_bins + 1;
    for ianimal = 1:n_animals
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        
        try
            % --- A. Truncate Data ---
            animal_data_raw = task_data(ianimal).spatial_binned_fr_all; 
            [~, n_bins, num_trials_raw] = size(animal_data_raw);
            
            if TRUNCATE_AT_DISENGAGEMENT
                diseng_point = min([task_data(ianimal).change_point_mean, num_trials_raw]);
                if isempty(diseng_point) || isnan(diseng_point)
                    diseng_point = num_trials_raw;
                end
                animal_data = animal_data_raw(:, :, 1:diseng_point);
            else
                animal_data = animal_data_raw;
            end
            num_trials = size(animal_data, 3);
            
            % --- B. PCA Reduction per Area ---
            AreaActivity = struct();
            areas_to_check = cfg.areas_to_include;
            
            for ia = 1:length(areas_to_check)
                area_name = areas_to_check{ia};
                idx_field = cfg.area_field_map(area_name);
                
                if isfield(task_data(ianimal), idx_field)
                    u_idx = logical(task_data(ianimal).(idx_field));
                    if sum(u_idx) < min_units_per_region, continue; end
                    
                    area_dat = animal_data(u_idx, :, :);
                    reshaped_dat = reshape(area_dat, sum(u_idx), [])'; 
                    
                    if size(reshaped_dat, 2) >= 2 
                        [~, scores, ~, ~, explained] = pca(reshaped_dat);
                        
                        if strcmp(pca_selection_method, 'variance')
                            cum_var = cumsum(explained);
                            n_comps = find(cum_var >= pca_variance_threshold, 1);
                            if isempty(n_comps), n_comps = size(scores, 2); end
                        else
                            n_comps = n_components_reduced;
                        end
                        
                        n_comps = max(n_comps, num_ccs_analyze); 
                        n_comps = min(n_comps, size(scores, 2));
                        
                        reduced = reshape(scores(:, 1:n_comps)', n_comps, n_bins, num_trials);
                        AreaActivity.(area_name).data = reduced;
                        AreaActivity.(area_name).n_comps = n_comps;
                    end
                end
            end
            
            % --- C. CCA Analysis Loop ---
            for ipair = 1:n_pairs
                a1 = area_pairs_to_analyze{ipair, 1};
                a2 = area_pairs_to_analyze{ipair, 2};
                
                if ~isfield(AreaActivity, a1) || ~isfield(AreaActivity, a2), continue; end
                
                d1 = AreaActivity.(a1).data; d2 = AreaActivity.(a2).data;
                nc1 = AreaActivity.(a1).n_comps; nc2 = AreaActivity.(a2).n_comps;
                
                cca_tr = nan(num_ccs_analyze, num_trials);
                cca_tr_shuff = nan(num_ccs_analyze, num_trials); 
                prec_tr_idx = nan(num_ccs_analyze, num_trials);
                prec_tr_idx_shuff = nan(num_ccs_analyze, num_trials); 
                prec_tr_curve = nan(num_ccs_analyze, n_shifts, num_trials);
                prec_tr_curve_shuff = nan(num_ccs_analyze, n_shifts, num_trials);
                
                cca_bin = nan(num_ccs_analyze, n_bins);
                cca_bin_shuff = nan(num_ccs_analyze, n_bins); 
                prec_bin_idx = nan(num_ccs_analyze, n_bins);
                prec_bin_idx_shuff = nan(num_ccs_analyze, n_bins);
                prec_bin_curve = nan(num_ccs_analyze, n_shifts, n_bins);
                prec_bin_curve_shuff = nan(num_ccs_analyze, n_shifts, n_bins);
                
                % =========================================================
                % ANALYSIS 1: TRIAL-WISE
                % =========================================================
                for t = 1:num_trials
                    win = t + n_trials_window; win = win(win>=1 & win<=num_trials);
                    if isempty(win), continue; end
                    
                    D1_local = d1(:, :, win); D2_local = d2(:, :, win);
                    
                    nan_bins_1 = squeeze(any(isnan(D1_local), [1, 3]));
                    nan_bins_2 = squeeze(any(isnan(D2_local), [1, 3]));
                    valid_bins = ~nan_bins_1 & ~nan_bins_2;
                    if sum(valid_bins) < max(nc1, nc2) + 5, continue; end
                    
                    D1_local = D1_local(:, valid_bins, :);
                    D2_local = D2_local(:, valid_bins, :);
                    
                    x = reshape(D1_local, nc1, []); y = reshape(D2_local, nc2, []);
                    
                    [~,~,r] = canoncorr(x', y');
                    cca_tr(1:min(length(r), num_ccs_analyze), t) = r(1:min(length(r), num_ccs_analyze));
                    [c_curve, c_idx] = calc_precession(D1_local, D2_local, max_shift_bins, num_ccs_analyze, nc1, nc2);
                    prec_tr_curve(:, :, t) = c_curve;
                    prec_tr_idx(:, t) = c_idx;
                    
                    r_sh_iter = nan(num_ccs_analyze, n_shuffles);
                    p_sh_idx_iter = nan(num_ccs_analyze, n_shuffles);
                    p_sh_curve_iter = nan(num_ccs_analyze, n_shifts, n_shuffles);
                    n_bins_local = size(D1_local, 2);
                    
                    parfor ish = 1:n_shuffles
                         perm_idx = randperm(n_bins_local);
                         D1_s = D1_local(:, perm_idx, :); x_s = reshape(D1_s, nc1, []);
                         [~,~,rs] = canoncorr(x_s', y');
                         r_sh_iter(:, ish) = rs(1:min(length(rs), num_ccs_analyze));
                         [c_curve_sh, c_idx_sh] = calc_precession(D1_s, D2_local, max_shift_bins, num_ccs_analyze, nc1, nc2);
                         p_sh_curve_iter(:, :, ish) = c_curve_sh;
                         p_sh_idx_iter(:, ish) = c_idx_sh;
                    end
                    cca_tr_shuff(:, t) = mean(r_sh_iter, 2, 'omitnan');
                    prec_tr_curve_shuff(:, :, t) = mean(p_sh_curve_iter, 3, 'omitnan');
                    prec_tr_idx_shuff(:, t) = mean(p_sh_idx_iter, 2, 'omitnan');
                end
                
                % =========================================================
                % ANALYSIS 2: BIN-WISE
                % =========================================================
                for b = 1:n_bins
                    win = b + n_bins_window; win = win(win>=1 & win<=n_bins);
                    if isempty(win), continue; end
                    
                    D1_local = d1(:, win, :); D2_local = d2(:, win, :);
                    
                    nan_trials_1 = squeeze(any(isnan(D1_local), [1, 2]));
                    nan_trials_2 = squeeze(any(isnan(D2_local), [1, 2]));
                    valid_trials = ~nan_trials_1 & ~nan_trials_2;
                    if sum(valid_trials) < max(nc1, nc2) + 5, continue; end
                    
                    D1_local = D1_local(:, :, valid_trials);
                    D2_local = D2_local(:, :, valid_trials);
                    
                    x = reshape(D1_local, nc1, []); y = reshape(D2_local, nc2, []);
                    
                    [~,~,r] = canoncorr(x', y');
                    cca_bin(1:min(length(r), num_ccs_analyze), b) = r(1:min(length(r), num_ccs_analyze));
                    [c_curve, c_idx] = calc_precession(D1_local, D2_local, max_shift_bins, num_ccs_analyze, nc1, nc2);
                    prec_bin_curve(:, :, b) = c_curve;
                    prec_bin_idx(:, b) = c_idx;
                    
                    r_sh_iter = nan(num_ccs_analyze, n_shuffles);
                    p_sh_idx_iter = nan(num_ccs_analyze, n_shuffles);
                    p_sh_curve_iter = nan(num_ccs_analyze, n_shifts, n_shuffles);
                    n_tr_local = size(D1_local, 3);
                    
                    parfor ish = 1:n_shuffles
                         perm_idx = randperm(n_tr_local);
                         D1_s = D1_local(:, :, perm_idx); x_s = reshape(D1_s, nc1, []);
                         [~,~,rs] = canoncorr(x_s', y');
                         r_sh_iter(:, ish) = rs(1:min(length(rs), num_ccs_analyze));
                         [c_curve_sh, c_idx_sh] = calc_precession(D1_s, D2_local, max_shift_bins, num_ccs_analyze, nc1, nc2);
                         p_sh_curve_iter(:, :, ish) = c_curve_sh;
                         p_sh_idx_iter(:, ish) = c_idx_sh;
                    end
                    cca_bin_shuff(:, b) = mean(r_sh_iter, 2, 'omitnan');
                    prec_bin_curve_shuff(:, :, b) = mean(p_sh_curve_iter, 3, 'omitnan');
                    prec_bin_idx_shuff(:, b) = mean(p_sh_idx_iter, 2, 'omitnan');
                end
                
                % --- D. Store Results ---
                group_results(ipair).all_bins_corr{ianimal} = cca_bin;
                group_results(ipair).all_bins_corr_shuff{ianimal} = cca_bin_shuff;
                group_results(ipair).all_bins_precession_idx{ianimal} = prec_bin_idx;
                group_results(ipair).all_bins_precession_idx_shuff{ianimal} = prec_bin_idx_shuff; 
                group_results(ipair).all_bins_precession_curve{ianimal} = prec_bin_curve;
                group_results(ipair).all_bins_precession_curve_shuff{ianimal} = prec_bin_curve_shuff; 
                
                lp = analysis_lp(ianimal);
                if ~isnan(lp) && lp > 10 && (lp + 9) <= num_trials
                    idx_early = 1:10; idx_pre = (lp - 10) : (lp - 1); idx_post = lp : (lp + 9);
                    get_cols_2d = @(data, cols) data(:, cols);
                    get_cols_3d = @(data, cols) data(:, :, cols);
                    
                    group_results(ipair).trial_corr_early{ianimal} = get_cols_2d(cca_tr, idx_early);
                    group_results(ipair).trial_corr_pre{ianimal}   = get_cols_2d(cca_tr, idx_pre);
                    group_results(ipair).trial_corr_post{ianimal}  = get_cols_2d(cca_tr, idx_post);
                    group_results(ipair).trial_corr_early_shuff{ianimal} = get_cols_2d(cca_tr_shuff, idx_early);
                    group_results(ipair).trial_corr_pre_shuff{ianimal}   = get_cols_2d(cca_tr_shuff, idx_pre);
                    group_results(ipair).trial_corr_post_shuff{ianimal}  = get_cols_2d(cca_tr_shuff, idx_post);
                    
                    group_results(ipair).trial_precession_early_idx{ianimal} = get_cols_2d(prec_tr_idx, idx_early);
                    group_results(ipair).trial_precession_pre_idx{ianimal}   = get_cols_2d(prec_tr_idx, idx_pre);
                    group_results(ipair).trial_precession_post_idx{ianimal}  = get_cols_2d(prec_tr_idx, idx_post);
                    group_results(ipair).trial_precession_early_idx_shuff{ianimal} = get_cols_2d(prec_tr_idx_shuff, idx_early);
                    group_results(ipair).trial_precession_pre_idx_shuff{ianimal}   = get_cols_2d(prec_tr_idx_shuff, idx_pre);
                    group_results(ipair).trial_precession_post_idx_shuff{ianimal}  = get_cols_2d(prec_tr_idx_shuff, idx_post);
                    
                    group_results(ipair).trial_precession_early_curve{ianimal} = get_cols_3d(prec_tr_curve, idx_early);
                    group_results(ipair).trial_precession_pre_curve{ianimal}   = get_cols_3d(prec_tr_curve, idx_pre);
                    group_results(ipair).trial_precession_post_curve{ianimal}  = get_cols_3d(prec_tr_curve, idx_post);
                    group_results(ipair).trial_precession_early_curve_shuff{ianimal} = get_cols_3d(prec_tr_curve_shuff, idx_early);
                    group_results(ipair).trial_precession_pre_curve_shuff{ianimal}   = get_cols_3d(prec_tr_curve_shuff, idx_pre);
                    group_results(ipair).trial_precession_post_curve_shuff{ianimal}  = get_cols_3d(prec_tr_curve_shuff, idx_post);
                end
            end 
        catch ME
            fprintf('Error processing animal %d: %s\n', ianimal, ME.message);
        end
    end
    saved_config = current_config;
    save(save_path, 'group_results', 'is_learner', 'analysis_lp', 'saved_config', '-v7.3');
    fprintf('\nAnalysis Complete. Final Save...\n');
end

%% 5. PLOTTING (Summary Figures)
fprintf('Generating Plots...\n');
lags = -max_shift_bins:max_shift_bins;
n_total_animals = length(is_learner); 
for ipair = 1:n_pairs
    fields = fieldnames(group_results);
    for f = 1:length(fields)
        if iscell(group_results(ipair).(fields{f})) && length(group_results(ipair).(fields{f})) < n_total_animals
            group_results(ipair).(fields{f}){n_total_animals} = [];
        end
    end
end

% --- B1. Epoch Correlation (Bar Graph with Mixed rmANOVA) ---
fprintf('\n--- Epoch Correlation Mixed rmANOVA Results ---\n');
figure('Name', 'Epoch Correlation (Learners vs Non)', 'Color', 'w', 'Position', [100 100 1400 800]);
tiledlayout('flow');
for ipair = 1:n_pairs
    e_vals = extract_animal_means(group_results(ipair).trial_corr_early, 1);
    p_vals = extract_animal_means(group_results(ipair).trial_corr_pre, 1);
    x_vals = extract_animal_means(group_results(ipair).trial_corr_post, 1);
    if all(isnan(e_vals)), continue; end
    nexttile;
    fprintf('\nPair: %s\n', group_results(ipair).pair_name);
    plot_grouped_bars_with_rmanova(e_vals, p_vals, x_vals, is_learner, group_results(ipair).pair_name, 'Correlation (CC1)', false);
end
save_to_svg(fullfile(save_dir, 'Epoch_Corr_Bars_Split'));

% --- B2. Epoch Precession Index (Bar Graph with Mixed rmANOVA) ---
fprintf('\n--- Epoch Precession Index Mixed rmANOVA Results ---\n');
figure('Name', 'Epoch Precession (Learners vs Non)', 'Color', 'w', 'Position', [100 100 1400 800]);
tiledlayout('flow');
for ipair = 1:n_pairs
    e_vals = extract_animal_means(group_results(ipair).trial_precession_early_idx, 1);
    p_vals = extract_animal_means(group_results(ipair).trial_precession_pre_idx, 1);
    x_vals = extract_animal_means(group_results(ipair).trial_precession_post_idx, 1);
    if all(isnan(e_vals)), continue; end
    nexttile;
    fprintf('\nPair: %s\n', group_results(ipair).pair_name);
    plot_grouped_bars_with_rmanova(e_vals, p_vals, x_vals, is_learner, group_results(ipair).pair_name, 'Precession Index', true);
end
save_to_svg(fullfile(save_dir, 'Epoch_Precession_Bars_Split'));

%% 5B. COMBINED CONTINUOUS CURVES, ERROR BARS & NETWORKS (REAL VS SHUFFLED)
fprintf('\n--- Generating Combined Curves, Error Bars & Networks (%d CCs) ---\n', num_ccs_analyze);

% Define layout for Striatum (DMS, DLS, ACC)
layout_def.names = {'DMS', 'DLS', 'ACC'};
layout_def.x     = [3.0, 7.0, 5.0];
layout_def.y     = [3.0, 3.0, 7.0];

for g_idx = 1:2
    if g_idx == 1
        mask = is_learner; group_label = 'Learners';
        c_real_e = [0.4 0.6 0.8]; % Light Teal (Naive)
        c_real_x = [0.1 0.3 0.6]; % Dark Teal (Expert)
    else
        mask = ~is_learner; group_label = 'Non-Learners';
        c_real_e = [0.9 0.5 0.3]; % Light Rust (Naive)
        c_real_x = [0.7 0.2 0.1]; % Dark Rust (Expert)
    end
    c_shuff_e = [0.7 0.7 0.7]; % Light Gray
    c_shuff_x = [0.4 0.4 0.4]; % Dark Gray
    
    if sum(mask) == 0, continue; end
    
    for cc = 1:num_ccs_analyze
        fig_cc_comb = figure('Name', sprintf('Combined_CC_%d_%s', cc, group_label), 'Color', 'w', 'Position', [100 100 1000 200*n_pairs]);
        t_cc_comb = tiledlayout(n_pairs, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        fig_pi_comb = figure('Name', sprintf('Combined_Prec_%d_%s', cc, group_label), 'Color', 'w', 'Position', [150 150 1000 200*n_pairs]);
        t_pi_comb = tiledlayout(n_pairs, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        net_e_cc = nan(n_pairs, 1); net_x_cc = nan(n_pairs, 1);
        net_e_ifi = nan(n_pairs, 1); net_x_ifi = nan(n_pairs, 1);
        net_e_cc_pval = nan(n_pairs, 1); net_x_cc_pval = nan(n_pairs, 1);
        net_e_ifi_pval = nan(n_pairs, 1); net_x_ifi_pval = nan(n_pairs, 1);
        
        for ipair = 1:n_pairs
            pair_name = group_results(ipair).pair_name;
            title_str = sprintf('%s (CC%d)', pair_name, cc);
            
            e_cc_r = extract_animal_means(group_results(ipair).trial_corr_early, cc);
            e_cc_s = extract_animal_means(group_results(ipair).trial_corr_early_shuff, cc);
            x_cc_r = extract_animal_means(group_results(ipair).trial_corr_post, cc);
            x_cc_s = extract_animal_means(group_results(ipair).trial_corr_post_shuff, cc);
            
            e_cc_tr_r = extract_epoch_trials_cc(group_results(ipair).trial_corr_early, n_animals, cc);
            e_cc_tr_s = extract_epoch_trials_cc(group_results(ipair).trial_corr_early_shuff, n_animals, cc);
            x_cc_tr_r = extract_epoch_trials_cc(group_results(ipair).trial_corr_post, n_animals, cc);
            x_cc_tr_s = extract_epoch_trials_cc(group_results(ipair).trial_corr_post_shuff, n_animals, cc);
            
            e_pi_r = extract_animal_means(group_results(ipair).trial_precession_early_idx, cc);
            e_pi_s = extract_animal_means(group_results(ipair).trial_precession_early_idx_shuff, cc);
            x_pi_r = extract_animal_means(group_results(ipair).trial_precession_post_idx, cc);
            x_pi_s = extract_animal_means(group_results(ipair).trial_precession_post_idx_shuff, cc);
            
            e_pi_tr_r = extract_epoch_trials_cc(group_results(ipair).trial_precession_early_idx, n_animals, cc);
            e_pi_tr_s = extract_epoch_trials_cc(group_results(ipair).trial_precession_early_idx_shuff, n_animals, cc);
            x_pi_tr_r = extract_epoch_trials_cc(group_results(ipair).trial_precession_post_idx, n_animals, cc);
            x_pi_tr_s = extract_epoch_trials_cc(group_results(ipair).trial_precession_post_idx_shuff, n_animals, cc);
            
            n_e = sum(~isnan(e_cc_r(mask)) & ~isnan(e_cc_s(mask)));
            n_x = sum(~isnan(x_cc_r(mask)) & ~isnan(x_cc_s(mask)));
            annotated_title = sprintf('%s (n_{naive}=%d, n_{expert}=%d)', title_str, n_e, n_x);
            
            if n_e > 2, [~, p_cc_e] = ttest(e_cc_r(mask), e_cc_s(mask)); else, p_cc_e = nan; end
            if n_x > 2, [~, p_cc_x] = ttest(x_cc_r(mask), x_cc_s(mask)); else, p_cc_x = nan; end
            
            n_e_pi = sum(~isnan(e_pi_r(mask)) & ~isnan(e_pi_s(mask)));
            n_x_pi = sum(~isnan(x_pi_r(mask)) & ~isnan(x_pi_s(mask)));
            if n_e_pi > 2, [~, p_ifi_e] = ttest(e_pi_r(mask), e_pi_s(mask)); else, p_ifi_e = nan; end
            if n_x_pi > 2, [~, p_ifi_x] = ttest(x_pi_r(mask), x_pi_s(mask)); else, p_ifi_x = nan; end
            
            net_e_cc(ipair) = mean(e_cc_r(mask), 'omitnan');
            net_x_cc(ipair) = mean(x_cc_r(mask), 'omitnan');
            net_e_ifi(ipair) = mean(e_pi_r(mask), 'omitnan');
            net_x_ifi(ipair) = mean(x_pi_r(mask), 'omitnan');
            net_e_cc_pval(ipair) = p_cc_e;
            net_x_cc_pval(ipair) = p_cc_x;
            net_e_ifi_pval(ipair) = p_ifi_e;
            net_x_ifi_pval(ipair) = p_ifi_x;
            
            % --- Render CC Combined ---
            figure(fig_cc_comb); 
            nexttile(t_cc_comb);
            plot_continuous_trials_with_shuff(e_cc_tr_r(mask,:), e_cc_tr_s(mask,:), x_cc_tr_r(mask,:), x_cc_tr_s(mask,:), sprintf('%s Trial Curve', annotated_title), 'Correlation (CC)', c_real_e, c_real_x, c_shuff_e, c_shuff_x);
            if ipair ~= n_pairs, xlabel(''); end
            
            nexttile(t_cc_comb);
            plot_real_shuff_bars_with_stats(e_cc_r(mask), e_cc_s(mask), x_cc_r(mask), x_cc_s(mask), sprintf('%s Averages', annotated_title), 'Correlation (CC)', c_real_e, c_real_x, c_shuff_e, c_shuff_x);
            
            % --- Render Precession Combined ---
            figure(fig_pi_comb); 
            nexttile(t_pi_comb);
            plot_continuous_trials_with_shuff(e_pi_tr_r(mask,:), e_pi_tr_s(mask,:), x_pi_tr_r(mask,:), x_pi_tr_s(mask,:), sprintf('%s Trial Curve', annotated_title), 'Precession Index', c_real_e, c_real_x, c_shuff_e, c_shuff_x);
            if ipair ~= n_pairs, xlabel(''); end
            
            nexttile(t_pi_comb);
            plot_real_shuff_bars_with_stats(e_pi_r(mask), e_pi_s(mask), x_pi_r(mask), x_pi_s(mask), sprintf('%s Averages', annotated_title), 'Precession Index', c_real_e, c_real_x, c_shuff_e, c_shuff_x);
        end
        
        L = gobjects(4,1);
        L(1) = plot(nan, nan, 'Color', c_real_e, 'LineWidth', 2); hold on;
        L(2) = plot(nan, nan, 'Color', c_real_x, 'LineWidth', 2);
        L(3) = plot(nan, nan, 'Color', c_shuff_e, 'LineWidth', 2, 'LineStyle', '--');
        L(4) = plot(nan, nan, 'Color', c_shuff_x, 'LineWidth', 2, 'LineStyle', '--');
        
        figure(fig_cc_comb);
        lgd1 = legend(L, {'Real Naive', 'Real Expert', 'Shuff Naive', 'Shuff Expert'}, 'Orientation', 'horizontal', 'NumColumns', 4);
        lgd1.Layout.Tile = 'north';
        save_to_svg(fullfile(save_dir, sprintf('Combined_CC_%d_%s', cc, group_label)));
        
        figure(fig_pi_comb);
        lgd2 = legend(L, {'Real Naive', 'Real Expert', 'Shuff Naive', 'Shuff Expert'}, 'Orientation', 'horizontal', 'NumColumns', 4);
        lgd2.Layout.Tile = 'north';
        save_to_svg(fullfile(save_dir, sprintf('Combined_Prec_%d_%s', cc, group_label)));
        
        % --- Render Filtered Network for this CC and Group ---
        fig_net = figure('Name', sprintf('Network_CC%d_%s', cc, group_label), 'Color', 'w', 'Position', [200 200 1000 500]);
        t_net = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
        
        global_max_cc = max([net_e_cc; net_x_cc], [], 'omitnan');
        global_max_ifi = max(abs([net_e_ifi; net_x_ifi]), [], 'omitnan');
        if isempty(global_max_cc) || global_max_cc == 0 || isnan(global_max_cc), global_max_cc = 1; end
        if isempty(global_max_ifi) || global_max_ifi == 0 || isnan(global_max_ifi), global_max_ifi = 1; end
        
        nexttile(t_net);
        plot_dual_network(area_pairs_to_analyze, net_e_cc, net_e_ifi, net_e_cc_pval, net_e_ifi_pval, sprintf('%s: Naive (CC%d)', group_label, cc), layout_def, global_max_cc, global_max_ifi);
        
        nexttile(t_net);
        plot_dual_network(area_pairs_to_analyze, net_x_cc, net_x_ifi, net_x_cc_pval, net_x_ifi_pval, sprintf('%s: Expert (CC%d)', group_label, cc), layout_def, global_max_cc, global_max_ifi);
        
        save_to_svg(fullfile(save_dir, sprintf('Network_Filtered_CC%d_%s', cc, group_label)));
    end
end

%% 6. LOCAL HELPERS
function plot_grouped_bars_with_rmanova(e_vals, p_vals, x_vals, is_learner, pair_name, y_label, test_zero)
    L_e = e_vals(is_learner); L_p = p_vals(is_learner); L_x = x_vals(is_learner);
    NL_e = e_vals(~is_learner); NL_p = p_vals(~is_learner); NL_x = x_vals(~is_learner);
    
    means_L = [mean(L_e, 'omitnan'), mean(L_p, 'omitnan'), mean(L_x, 'omitnan')];
    sems_L  = [std(L_e, 'omitnan')/sqrt(sum(~isnan(L_e))), std(L_p, 'omitnan')/sqrt(sum(~isnan(L_p))), std(L_x, 'omitnan')/sqrt(sum(~isnan(L_x)))];
    
    means_NL = [mean(NL_e, 'omitnan'), mean(NL_p, 'omitnan'), mean(NL_x, 'omitnan')];
    sems_NL  = [std(NL_e, 'omitnan')/sqrt(sum(~isnan(NL_e))), std(NL_p, 'omitnan')/sqrt(sum(~isnan(NL_p))), std(NL_x, 'omitnan')/sqrt(sum(~isnan(NL_x)))];
    
    hold on;
    b = bar(1:3, [means_L; means_NL]', 'grouped');
    b(1).FaceColor = [0.85 0.33 0.1]; 
    b(2).FaceColor = [0.6 0.6 0.6];   
    
    x_offset = [b(1).XEndPoints; b(2).XEndPoints]';
    errorbar(x_offset(:,1), means_L, sems_L, 'k.', 'LineWidth', 1.5);
    errorbar(x_offset(:,2), means_NL, sems_NL, 'k.', 'LineWidth', 1.5);
    
    yline(0, '-k', 'LineWidth', 0.5);
    xticks(1:3); xticklabels({'Early', 'Pre', 'Post'});
    title(pair_name); ylabel(y_label);
    
    yl = ylim; offset = (yl(2) - yl(1)) * 0.05; 
    min_y_needed = yl(1); max_y_needed = yl(2);
    
    if test_zero
        p_L = nan(1,3); p_NL = nan(1,3);
        if sum(~isnan(L_e)) > 2, [~, p_L(1)] = ttest(L_e, 0); end
        if sum(~isnan(L_p)) > 2, [~, p_L(2)] = ttest(L_p, 0); end
        if sum(~isnan(L_x)) > 2, [~, p_L(3)] = ttest(L_x, 0); end
        
        if sum(~isnan(NL_e)) > 2, [~, p_NL(1)] = ttest(NL_e, 0); end
        if sum(~isnan(NL_p)) > 2, [~, p_NL(2)] = ttest(NL_p, 0); end
        if sum(~isnan(NL_x)) > 2, [~, p_NL(3)] = ttest(NL_x, 0); end
        
        for idx = 1:3
            if p_L(idx) < 0.05
                if means_L(idx) >= 0
                    y_pos = means_L(idx) + sems_L(idx) + offset; vert_align = 'bottom';
                else
                    y_pos = means_L(idx) - sems_L(idx) - offset; vert_align = 'top'; 
                end
                if y_pos > max_y_needed, max_y_needed = y_pos + offset; end
                if y_pos < min_y_needed, min_y_needed = y_pos - offset; end
                text(x_offset(idx, 1), y_pos, '*', 'FontSize', 18, 'HorizontalAlignment', 'center', 'VerticalAlignment', vert_align, 'Color', [0.85 0.33 0.1]);
            end
            if p_NL(idx) < 0.05
                if means_NL(idx) >= 0
                    y_pos = means_NL(idx) + sems_NL(idx) + offset; vert_align = 'bottom';
                else
                    y_pos = means_NL(idx) - sems_NL(idx) - offset; vert_align = 'top'; 
                end
                if y_pos > max_y_needed, max_y_needed = y_pos + offset; end
                if y_pos < min_y_needed, min_y_needed = y_pos - offset; end
                text(x_offset(idx, 2), y_pos, '*', 'FontSize', 18, 'HorizontalAlignment', 'center', 'VerticalAlignment', vert_align, 'Color', [0.4 0.4 0.4]);
            end
        end
    end
    
    ylim([min(yl(1), min_y_needed), max(yl(2), max_y_needed)]); 
    
    n_anim = length(e_vals);
    Y = [e_vals; p_vals; x_vals];
    G_epoch = [ones(n_anim,1); 2*ones(n_anim,1); 3*ones(n_anim,1)];
    G_group = [is_learner; is_learner; is_learner];
    
    valid_mask = ~isnan(Y);
    if sum(valid_mask) > 3
        [p_anova, tbl, stats] = anovan(Y(valid_mask), {G_epoch(valid_mask), G_group(valid_mask)}, ...
            'model', 'interaction', 'varnames', {'Epoch', 'LearnerGroup'}, 'display', 'off');
        
        disp(cell2table(tbl(2:end,:), 'VariableNames', tbl(1,:)));
        
        p_epoch = p_anova(1);
        p_group = p_anova(2);
        p_interaction = p_anova(3);
        
        sig_text = {'ANOVA:'};
        has_sig = false;
        if p_epoch < 0.05, sig_text{end+1} = sprintf('Epoch: p=%.3f', p_epoch); has_sig = true; end
        if p_group < 0.05, sig_text{end+1} = sprintf('Group: p=%.3f', p_group); has_sig = true; end
        if p_interaction < 0.05, sig_text{end+1} = sprintf('Ep x Grp: p=%.3f', p_interaction); has_sig = true; end
        
        if ~has_sig, sig_text{end+1} = 'n.s.'; end
        
        curr_yl = ylim; curr_xl = xlim;
        y_range = curr_yl(2) - curr_yl(1);
        ylim([curr_yl(1), curr_yl(2) + y_range*0.25]); 
        new_yl = ylim;
        
        text(curr_xl(1) + 0.03*(curr_xl(2)-curr_xl(1)), new_yl(2) - 0.03*(new_yl(2)-new_yl(1)), ...
             strjoin(sig_text, '\n'), 'VerticalAlignment', 'top', ...
             'FontSize', 9, 'EdgeColor', 'k', 'BackgroundColor', 'w', 'Margin', 3);
        
        if p_interaction < 0.05
            fprintf('  -> Significant Interaction (p=%.4f). Post-hoc pairwise:\n', p_interaction);
            [c, ~, ~, nms] = multcompare(stats, 'Dimension', [1 2], 'Display', 'off');
            for mc = 1:size(c,1)
                if c(mc, 6) < 0.05
                    fprintf('     %s vs %s: p = %.4f\n', nms{c(mc,1)}, nms{c(mc,2)}, c(mc,6));
                end
            end
        else
            if p_epoch < 0.05
                fprintf('  -> Significant Main Effect of Epoch (p=%.4f). Post-hoc pairwise:\n', p_epoch);
                [c, ~, ~, nms] = multcompare(stats, 'Dimension', 1, 'Display', 'off');
                for mc = 1:size(c,1)
                    if c(mc, 6) < 0.05
                        fprintf('     %s vs %s: p = %.4f\n', nms{c(mc,1)}, nms{c(mc,2)}, c(mc,6));
                    end
                end
            end
        end
    end
end

function plot_real_shuff_bars_with_stats(e_r, e_s, x_r, x_s, title_str, y_label, c_re, c_rx, c_se, c_sx)
    means = [mean(e_r, 'omitnan'), mean(e_s, 'omitnan'); mean(x_r, 'omitnan'), mean(x_s, 'omitnan')];
    sems  = [std(e_r, 'omitnan')/sqrt(sum(~isnan(e_r))), std(e_s, 'omitnan')/sqrt(sum(~isnan(e_s))); ...
             std(x_r, 'omitnan')/sqrt(sum(~isnan(x_r))), std(x_s, 'omitnan')/sqrt(sum(~isnan(x_s)))];
             
    hold on;
    b = bar(1:2, means, 'grouped');
    
    b(1).FaceColor = 'flat';
    b(1).CData = [c_re; c_rx]; 
    
    b(2).FaceColor = 'flat';
    b(2).CData = [c_se; c_sx];
    
    x_offset = [b(1).XEndPoints; b(2).XEndPoints]';
    errorbar(x_offset(:,1), means(:,1), sems(:,1), 'k.', 'LineWidth', 1.5);
    errorbar(x_offset(:,2), means(:,2), sems(:,2), 'k.', 'LineWidth', 1.5);
    
    yline(0, '-k', 'LineWidth', 0.5);
    xticks(1:2); xticklabels({'Naive', 'Expert'});
    ylabel(y_label); title(title_str);
    
    valid_e = ~isnan(e_r) & ~isnan(e_s);
    valid_x = ~isnan(x_r) & ~isnan(x_s);
    valid_ep = ~isnan(e_r) & ~isnan(x_r); 
    
    p_e_shuff = nan; p_x_shuff = nan; p_epoch = nan;
    if sum(valid_e) > 2, [~, p_e_shuff] = ttest(e_r(valid_e), e_s(valid_e)); end
    if sum(valid_x) > 2, [~, p_x_shuff] = ttest(x_r(valid_x), x_s(valid_x)); end
    if sum(valid_ep) > 2, [~, p_epoch] = ttest(x_r(valid_ep), e_r(valid_ep)); end
    
    yl = ylim; 
    if yl(2) == 0 && yl(1) == 0, yl = [-1 1]; end 
    offset = (yl(2) - yl(1)) * 0.08;
    max_y = max([means(1,:) + sems(1,:), means(2,:) + sems(2,:)]);
    
    if p_e_shuff < 0.05
        plot(x_offset(1,:), [1 1]*(max_y+offset), '-k');
        text(mean(x_offset(1,:)), max_y+offset*1.4, '*', 'FontSize', 14, 'HorizontalAlignment', 'center');
    end
    if p_x_shuff < 0.05
        plot(x_offset(2,:), [1 1]*(max_y+offset), '-k');
        text(mean(x_offset(2,:)), max_y+offset*1.4, '*', 'FontSize', 14, 'HorizontalAlignment', 'center');
    end
    
    text_y = max_y + offset * 3;
    if p_epoch < 0.05
        text(1.5, text_y, sprintf('Naive vs Expert: p = %.3f', p_epoch), 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    else
        text(1.5, text_y, 'Naive vs Expert: n.s.', 'HorizontalAlignment', 'center', 'FontSize', 9);
    end
    ylim([min(yl(1), 0 - offset), text_y + offset*2]);
end

function plot_continuous_trials_with_shuff(e_r, e_s, x_r, x_s, title_str, y_label, c_re, c_rx, c_se, c_sx)
    hold on;
    plot_epoch_line(1:10, e_s, c_se, '--'); 
    plot_epoch_line(11:20, x_s, c_sx, '--'); 
    plot_epoch_line(1:10, e_r, c_re, '-');  
    plot_epoch_line(11:20, x_r, c_rx, '-');  
    
    xline(10.5, 'k:'); 
    yline(0, '-k', 'LineWidth', 0.5);
    
    xticks([5, 15]); 
    xticklabels({'Naive', 'Expert'});
    xlim([1 20]);
    title(title_str); ylabel(y_label);
end

function plot_dual_network(pairs, cc_vals, ifi_vals, pval_cc, pval_ifi, titleStr, layout, max_cc, max_ifi)
    regions = unique(pairs(:)); 
    sources = []; targets = []; 
    weights_cc = []; weights_ifi = []; weights_pval_ifi = [];
    
    n_edges = size(pairs, 1);
    for i = 1:n_edges
        u = find(strcmp(regions, pairs{i, 1})); 
        v = find(strcmp(regions, pairs{i, 2}));
        
        cc = cc_vals(i);
        ifi = ifi_vals(i);
        p_cc = pval_cc(i);
        p_ifi = pval_ifi(i);
        
        if isnan(cc) || isnan(p_cc) || p_cc >= 0.05
            continue; 
        end
        
        if ifi >= 0
            s = u; t = v; 
        else
            s = v; t = u; 
        end
    
        sources = [sources; s]; 
        targets = [targets; t]; 
        weights_cc = [weights_cc; cc];
        weights_ifi = [weights_ifi; abs(ifi)];
        weights_pval_ifi = [weights_pval_ifi; p_ifi];
    end
    
    NodeTable = table(regions, 'VariableNames', {'Name'});
    x_coords = zeros(1, numel(regions)); y_coords = zeros(1, numel(regions));
    
    for i = 1:numel(regions)
        idx = find(strcmp(layout.names, regions{i}));
        if ~isempty(idx), x_coords(i) = layout.x(idx); y_coords(i) = layout.y(idx); end
    end
    
    if isempty(sources)
        G_empty = graph([], [], [], NodeTable);
        p_empty = plot(G_empty, 'XData', x_coords, 'YData', y_coords);
        p_empty.NodeColor = [0.85 0.85 0.85]; p_empty.MarkerSize = 30; p_empty.NodeFontSize = 12; p_empty.NodeFontWeight = 'bold';
        p_empty.EdgeColor = 'none';
        axis off; xlim([0 10]); ylim([0 10]); 
        title(titleStr, 'FontSize', 14, 'FontWeight', 'bold');
        return;
    end
    
    G_un = graph(sources, targets, weights_cc, NodeTable);
    p_un = plot(G_un, 'XData', x_coords, 'YData', y_coords);
    p_un.NodeColor = [0.85 0.85 0.85]; p_un.MarkerSize = 30; p_un.NodeFontSize = 12; p_un.NodeFontWeight = 'bold';
    p_un.EdgeColor = [0.85 0.85 0.85]; 
    
    norm_cc_un = G_un.Edges.Weight / max_cc;
    p_un.LineWidth = 1 + (norm_cc_un * 8); 
    hold on;
    
    sig_idx = weights_pval_ifi < 0.05;
    
    if any(sig_idx)
        sig_sources = sources(sig_idx);
        sig_targets = targets(sig_idx);
        sig_cc      = weights_cc(sig_idx);
        sig_ifi     = weights_ifi(sig_idx);
        
        EdgeTable_dir = table([sig_sources, sig_targets], sig_cc, sig_ifi, ...
            'VariableNames', {'EndNodes', 'Weight', 'IFI'});
            
        G_dir = digraph(EdgeTable_dir, NodeTable);
        
        p_dir = plot(G_dir, 'XData', x_coords, 'YData', y_coords);
        p_dir.NodeColor = 'none'; 
        p_dir.EdgeLabel = {};     
        p_dir.NodeLabel = {};     
        
        norm_cc_dir = G_dir.Edges.Weight / max_cc;
        p_dir.LineWidth = 1 + (norm_cc_dir * 8); 
        
        norm_ifi_dir = min(G_dir.Edges.IFI / max_ifi, 1); 
        base_color = [0.7 0.7 0.7];   
        target_color = [0.1 0.4 0.6]; 
        
        edge_colors_dir = zeros(length(norm_ifi_dir), 3);
        for i = 1:length(norm_ifi_dir)
            edge_colors_dir(i, :) = base_color + (target_color - base_color) * norm_ifi_dir(i);
        end
        
        p_dir.EdgeColor = edge_colors_dir; 
        p_dir.ArrowSize = 15;
        
        for i = 1:numedges(G_dir)
            s_name = G_dir.Edges.EndNodes(i, 1);
            t_name = G_dir.Edges.EndNodes(i, 2);
            
            s_idx = findnode(G_dir, s_name);
            t_idx = findnode(G_dir, t_name);
            
            x1 = x_coords(s_idx); y1 = y_coords(s_idx);
            x2 = x_coords(t_idx); y2 = y_coords(t_idx);
            
            x_mid = (x1 + x2) / 2;
            y_mid = (y1 + y2) / 2;
            
            dx = x2 - x1; dy = y2 - y1;
            len = sqrt(dx^2 + dy^2);
            nx = -dy / len * 0.15; 
            ny = dx / len * 0.15;  
            
            text(x_mid + nx, y_mid + ny, '*', 'FontSize', 22, 'Color', edge_colors_dir(i, :), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold');
        end
    end
    
    axis off; 
    xlim([0 10]); ylim([0 10]);
    title(titleStr, 'FontSize', 14, 'FontWeight', 'bold');
end

function [c_curve, p_idx] = calc_precession(D1, D2, max_shift, num_ccs, nc1, nc2)
    shifts = -max_shift : max_shift;
    r_shifts = zeros(num_ccs, length(shifts));
    
    for is = 1:length(shifts)
        shift = shifts(is);
        if shift < 0      
             s1 = D1(:, 1:end+shift, :); s2 = D2(:, 1-shift:end, :);
        elseif shift > 0  
             s1 = D1(:, 1+shift:end, :); s2 = D2(:, 1:end-shift, :);
        else
             s1 = D1; s2 = D2;
        end
        
        sx = reshape(s1, nc1, []);
        sy = reshape(s2, nc2, []);
        
        if size(sx,2) > max(nc1, nc2) + 2
             [~,~,rs] = canoncorr(sx', sy');
             r_shifts(1:min(length(rs), num_ccs), is) = rs(1:min(length(rs), num_ccs));
        end
    end
    
    c_curve = r_shifts;
    idx_neg = 1:max_shift; 
    idx_pos = max_shift+2 : length(shifts);
    p_idx = nan(num_ccs, 1);
    
    for cc = 1:num_ccs
        neg_val = mean(r_shifts(cc, idx_neg), 'omitnan');
        pos_val = mean(r_shifts(cc, idx_pos), 'omitnan');
        if (neg_val + pos_val) > 0.001 
            p_idx(cc) = (neg_val - pos_val) / (neg_val + pos_val);
        else
            p_idx(cc) = nan;
        end
    end
end

function plot_epoch_line(x_range, data_mat, color, linestyle)
    valid_mask = sum(~isnan(data_mat), 2) > 0;
    valid_data = data_mat(valid_mask, :);
    if isempty(valid_data), return; end
    
    mu = mean(valid_data, 1, 'omitnan');
    se = std(valid_data, 0, 1, 'omitnan') ./ sqrt(size(valid_data, 1));
    
    if strcmp(linestyle, '--')
        plot(x_range, mu, 'Color', color, 'LineStyle', linestyle, 'LineWidth', 2);
        plot(x_range, mu+se, 'Color', color, 'LineStyle', ':', 'LineWidth', 0.5);
        plot(x_range, mu-se, 'Color', color, 'LineStyle', ':', 'LineWidth', 0.5);
    else
        shadedErrorBar_local(x_range, mu, se, color);
    end
end

function mat = extract_epoch_trials_cc(cell_data, n_animals, cc_idx)
    mat = nan(n_animals, 10);
    for i = 1:n_animals
        d = cell_data{i};
        if ~isempty(d) && size(d, 1) >= cc_idx && size(d, 2) == 10
            mat(i, :) = d(cc_idx, :); 
        end
    end
end

function raw_means = extract_animal_means(cell_data, cc_idx)
    n_animals = length(cell_data); raw_means = nan(n_animals, 1);
    for i = 1:n_animals
        if ~isempty(cell_data{i}), raw_means(i) = mean(cell_data{i}(cc_idx, :), 2, 'omitnan'); end
    end
end

function save_to_svg(fig_name)
    fig = gcf; set(fig, 'Renderer', 'painters'); 
    fprintf('Saving %s.svg...\n', fig_name);
    try print(fig, '-dsvg', [fig_name '.svg']); catch, print(fig, '-dpng', [fig_name '.png']); end
end

function ok = check_dims(X, Y, min_samples)
    if isempty(X) || isempty(Y), ok = false; return; end
    n_samples = size(X, 2); if any(isnan(X(:))) || any(isnan(Y(:))), ok = false; return; end
    ok = n_samples > min_samples; 
end

function shadedErrorBar_local(x, y, err, color)
    fill([x, fliplr(x)], [y+err, fliplr(y-err)], color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold on; plot(x, y, 'Color', color, 'LineWidth', 1.5);
end