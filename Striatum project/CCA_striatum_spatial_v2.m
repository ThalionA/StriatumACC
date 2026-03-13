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
% - SVG Figures with ANOVA + multiple comparisons (sigstar) + 1-sample t-tests

%% 1. CONFIGURATION & PARAMETERS
clear; clc; close all;

% --- Data Files & Analysis Selection ---
cfg.task_data_file = "preprocessed_data.mat";
cfg.control_data_file = "preprocessed_data_control.mat";

% --- Analysis Selection ---
cfg.analysis_mode = 'task_only'; % Options: 'task_only', 'control_only', 'task_and_control'
cfg.areas_to_include = {'DMS', 'DLS', 'ACC'}; % List areas to keep (e.g., {'DMS', 'DLS', 'ACC'})
% Define mapping from area names to field names in the data struct
cfg.area_field_map = containers.Map(...
    {'DMS', 'DLS', 'ACC'}, ...
    {'is_dms', 'is_dls', 'is_acc'} ...
    );

% --- Processing Parameters ---
cfg.control_epoch_method = 'fixed_trial'; % How to align control trials
cfg.control_fixed_ref_trial = 40;
cfg.control_epoch_windows = {1:10, [-10, -1], [1, 10]}; % {early_range, pre_lp_offset, post_lp_offset} relative to reference point
cfg.task_lp_zscore_threshold = -2;
cfg.task_lp_window_length = 10; % movsum uses [0, N-1], so 10 means current + next 9
cfg.task_lp_min_consecutive = 7;

% --- CCA / PCA Parameters ---
pca_selection_method = 'variance'; % 'fixed' or 'variance'
pca_variance_threshold = 60;       % % variance to explain
n_components_reduced = 3;          % Fallback/fixed PC count
num_ccs_analyze = 1;               % Number of CCA components to analyze
n_trials_window = -3:3;            % Sliding window for Trial-wise analysis
n_bins_window = -3:3;              % Sliding window for Bin-wise analysis
n_shuffles = 50;                   % Number of shuffles
max_shift_bins = 3;                % Max shift for precession
min_units_per_region = 3;
TRUNCATE_AT_DISENGAGEMENT = true;

% --- Plotting / Spatial Parameters ---
landmarks = [20, 25];              % Task event bins

% --- Save Path ---
save_dir = './CCA_Results/';
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
    
    % Use your custom filter function
    task_data_raw = filterDataByArea(loaded_data.preprocessed_data, cfg.areas_to_include, cfg.area_field_map);
    fprintf('  Filtered Task data to %d areas: %s\n', numel(cfg.areas_to_include), strjoin(cfg.areas_to_include, ', '));
    
    % Use your custom processing function to get learning points and formatted data
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

%% 3. INITIALIZE RESULTS STRUCTURE
group_results = struct('pair_name', cell(n_pairs, 1), ...
                       'all_bins_corr', cell(n_pairs, 1), ...
                       'all_bins_corr_shuff', cell(n_pairs, 1), ...
                       'all_bins_precession', cell(n_pairs, 1), ...
                       'all_bins_precession_shuff', cell(n_pairs, 1), ...
                       ...
                       'trial_corr_early', cell(n_pairs, 1), ...
                       'trial_corr_pre', cell(n_pairs, 1), ...
                       'trial_corr_post', cell(n_pairs, 1), ...
                       ...
                       'trial_corr_early_shuff', cell(n_pairs, 1), ...
                       'trial_corr_pre_shuff', cell(n_pairs, 1), ...
                       'trial_corr_post_shuff', cell(n_pairs, 1), ...
                       ...
                       'trial_precession_early', cell(n_pairs, 1), ... 
                       'trial_precession_pre', cell(n_pairs, 1), ...   
                       'trial_precession_post', cell(n_pairs, 1), ...
                       ...
                       'trial_precession_early_shuff', cell(n_pairs, 1), ...
                       'trial_precession_pre_shuff', cell(n_pairs, 1), ...
                       'trial_precession_post_shuff', cell(n_pairs, 1));
                   
for ipair = 1:n_pairs
    group_results(ipair).pair_name = sprintf('%s-%s', area_pairs_to_analyze{ipair, 1}, area_pairs_to_analyze{ipair, 2});
end

%% 4. MAIN ANALYSIS LOOP
fprintf('\n--- Starting CCA Analysis ---\n');

for ianimal = 1:n_animals
    fprintf('\nProcessing Animal %d/%d...\n', ianimal, n_animals);
    
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
            
            if ~isfield(AreaActivity, a1) || ~isfield(AreaActivity, a2)
                continue; 
            end
            
            d1 = AreaActivity.(a1).data;
            d2 = AreaActivity.(a2).data;
            nc1 = AreaActivity.(a1).n_comps;
            nc2 = AreaActivity.(a2).n_comps;
            
            % Preallocate Local Results
            cca_tr = nan(num_ccs_analyze, num_trials);
            cca_tr_shuff = nan(num_ccs_analyze, num_trials); 
            prec_tr = nan(num_ccs_analyze, num_trials);
            prec_tr_shuff = nan(num_ccs_analyze, num_trials); 
            
            cca_bin = nan(num_ccs_analyze, n_bins);
            cca_bin_shuff = nan(num_ccs_analyze, n_bins); 
            prec_bin = nan(num_ccs_analyze, n_bins);
            prec_bin_shuff = nan(num_ccs_analyze, n_bins);
            
            % =========================================================
            % ANALYSIS 1: TRIAL-WISE
            % =========================================================
            for t = 1:num_trials
                win = t + n_trials_window;
                win = win(win>=1 & win<=num_trials);
                if isempty(win), continue; end
                
                D1_local = d1(:, :, win); 
                D2_local = d2(:, :, win);
                
                nan_bins_1 = squeeze(any(isnan(D1_local), [1, 3]));
                nan_bins_2 = squeeze(any(isnan(D2_local), [1, 3]));
                valid_bins = ~nan_bins_1 & ~nan_bins_2;
                
                if sum(valid_bins) < max(nc1, nc2) + 5, continue; end
                
                D1_local = D1_local(:, valid_bins, :);
                D2_local = D2_local(:, valid_bins, :);
                
                x = reshape(D1_local, nc1, []);
                y = reshape(D2_local, nc2, []);
                
                [~,~,r] = canoncorr(x', y');
                cca_tr(1:min(length(r), num_ccs_analyze), t) = r(1:min(length(r), num_ccs_analyze));
                prec_tr(:, t) = calc_precession(D1_local, D2_local, max_shift_bins, num_ccs_analyze, nc1, nc2);
                
                r_sh_iter = nan(num_ccs_analyze, n_shuffles);
                p_sh_iter = nan(num_ccs_analyze, n_shuffles);
                n_bins_local = size(D1_local, 2);
                
                parfor ish = 1:n_shuffles
                     perm_idx = randperm(n_bins_local);
                     D1_s = D1_local(:, perm_idx, :); 
                     x_s = reshape(D1_s, nc1, []);
                     
                     [~,~,rs] = canoncorr(x_s', y');
                     r_sh_iter(:, ish) = rs(1:min(length(rs), num_ccs_analyze));
                     p_sh_iter(:, ish) = calc_precession(D1_s, D2_local, max_shift_bins, num_ccs_analyze, nc1, nc2);
                end
                cca_tr_shuff(:, t) = mean(r_sh_iter, 2, 'omitnan');
                prec_tr_shuff(:, t) = mean(p_sh_iter, 2, 'omitnan');
            end
            
            % =========================================================
            % ANALYSIS 2: BIN-WISE
            % =========================================================
            for b = 1:n_bins
                win = b + n_bins_window;
                win = win(win>=1 & win<=n_bins);
                if isempty(win), continue; end
                
                D1_local = d1(:, win, :);
                D2_local = d2(:, win, :);
                
                nan_trials_1 = squeeze(any(isnan(D1_local), [1, 2]));
                nan_trials_2 = squeeze(any(isnan(D2_local), [1, 2]));
                valid_trials = ~nan_trials_1 & ~nan_trials_2;
                
                if sum(valid_trials) < max(nc1, nc2) + 5, continue; end
                
                D1_local = D1_local(:, :, valid_trials);
                D2_local = D2_local(:, :, valid_trials);
                
                x = reshape(D1_local, nc1, []);
                y = reshape(D2_local, nc2, []);
                
                [~,~,r] = canoncorr(x', y');
                cca_bin(1:min(length(r), num_ccs_analyze), b) = r(1:min(length(r), num_ccs_analyze));
                prec_bin(:, b) = calc_precession(D1_local, D2_local, max_shift_bins, num_ccs_analyze, nc1, nc2);
                
                r_sh_iter = nan(num_ccs_analyze, n_shuffles);
                p_sh_iter = nan(num_ccs_analyze, n_shuffles);
                n_tr_local = size(D1_local, 3);
                
                parfor ish = 1:n_shuffles
                     perm_idx = randperm(n_tr_local);
                     D1_s = D1_local(:, :, perm_idx);
                     x_s = reshape(D1_s, nc1, []);
                     
                     [~,~,rs] = canoncorr(x_s', y');
                     r_sh_iter(:, ish) = rs(1:min(length(rs), num_ccs_analyze));
                     p_sh_iter(:, ish) = calc_precession(D1_s, D2_local, max_shift_bins, num_ccs_analyze, nc1, nc2);
                end
                cca_bin_shuff(:, b) = mean(r_sh_iter, 2, 'omitnan');
                prec_bin_shuff(:, b) = mean(p_sh_iter, 2, 'omitnan');
            end
            
            % --- D. Store Results ---
            group_results(ipair).all_bins_corr{ianimal} = cca_bin;
            group_results(ipair).all_bins_corr_shuff{ianimal} = cca_bin_shuff;
            group_results(ipair).all_bins_precession{ianimal} = prec_bin;
            group_results(ipair).all_bins_precession_shuff{ianimal} = prec_bin_shuff; 
            
            lp = learning_points(ianimal);
            if ~isnan(lp) && lp > 10 && (lp + 9) <= num_trials
                idx_early = 1:10;
                idx_pre   = (lp - 10) : (lp - 1);
                idx_post  = lp : (lp + 9);
                get_cols = @(data, cols) data(:, cols);
                
                group_results(ipair).trial_corr_early{ianimal} = get_cols(cca_tr, idx_early);
                group_results(ipair).trial_corr_pre{ianimal}   = get_cols(cca_tr, idx_pre);
                group_results(ipair).trial_corr_post{ianimal}  = get_cols(cca_tr, idx_post);
                
                group_results(ipair).trial_corr_early_shuff{ianimal} = get_cols(cca_tr_shuff, idx_early);
                group_results(ipair).trial_corr_pre_shuff{ianimal}   = get_cols(cca_tr_shuff, idx_pre);
                group_results(ipair).trial_corr_post_shuff{ianimal}  = get_cols(cca_tr_shuff, idx_post);
                
                group_results(ipair).trial_precession_early{ianimal} = get_cols(prec_tr, idx_early);
                group_results(ipair).trial_precession_pre{ianimal}   = get_cols(prec_tr, idx_pre);
                group_results(ipair).trial_precession_post{ianimal}  = get_cols(prec_tr, idx_post);
                
                group_results(ipair).trial_precession_early_shuff{ianimal} = get_cols(prec_tr_shuff, idx_early);
                group_results(ipair).trial_precession_pre_shuff{ianimal}   = get_cols(prec_tr_shuff, idx_pre);
                group_results(ipair).trial_precession_post_shuff{ianimal}  = get_cols(prec_tr_shuff, idx_post);
            end
        end 
        save(save_path, 'group_results', '-v7.3');
    catch ME
        fprintf('Error processing animal %d: %s\n', ianimal, ME.message);
    end
end
fprintf('\nAnalysis Complete. Final Save...\n');
save(save_path, 'group_results', '-v7.3');

%% 5. PLOTTING (Summary Figures)
fprintf('Generating Plots...\n');

all_bin_sizes = cellfun(@(x) size(x, 2), group_results(1).all_bins_corr);
n_bins_plot = mode(all_bin_sizes(all_bin_sizes > 0));

% --- A. Spatial Correlation (Bin-wise) ---
figure('Name', 'Group Spatial Correlation', 'Color', 'w', 'Position', [100 100 1200 400]);
tiledlayout('flow');
for ipair = 1:n_pairs
    nexttile
    [mu, se] = aggregate_cells(group_results(ipair).all_bins_corr, n_bins_plot, 1); 
    [mu_s, ~] = aggregate_cells(group_results(ipair).all_bins_corr_shuff, n_bins_plot, 1);
    
    if isempty(mu), continue; end
    
    plot(1:n_bins_plot, mu_s, 'Color', [0.5 0.5 0.5], 'LineWidth', 1); hold on;
    shadedErrorBar_local(1:n_bins_plot, mu, se, 'b');
    
    xline(landmarks(:)); 
    title(group_results(ipair).pair_name);
    xlabel('Spatial Bin'); ylabel('Correlation (CC1)');
    xlim([1 n_bins_plot]);
end
linkaxes; save_to_svg(fullfile(save_dir, 'Spatial_Corr_Group'));

% --- B. Spatial Precession ---
figure('Name', 'Group Spatial Precession', 'Color', 'w', 'Position', [100 100 1200 400]);
tiledlayout('flow');
for ipair = 1:n_pairs
    nexttile
    [mu, se] = aggregate_cells(group_results(ipair).all_bins_precession, n_bins_plot, 1);
    [mu_s, ~] = aggregate_cells(group_results(ipair).all_bins_precession_shuff, n_bins_plot, 1); 
    
    if isempty(mu), continue; end
    
    plot(1:n_bins_plot, mu_s, 'Color', [0.6 0.6 0.6], 'LineWidth', 1, 'LineStyle', '--'); hold on;
    shadedErrorBar_local(1:n_bins_plot, mu, se, 'r');
    
    xline(landmarks(:)); 
    yline(0, '--k');
    title(group_results(ipair).pair_name);
    xlim([1 n_bins_plot]); ylim([-0.2 0.2]);
    ylabel('Precession Idx (CC1)');
end
linkaxes; save_to_svg(fullfile(save_dir, 'Spatial_Precession_Group'));

% --- C. Trial Correlation (Epochs) ---
figure('Name', 'Group Epoch Correlation', 'Color', 'w', 'Position', [100 100 1200 400]);
tiledlayout('flow');
for ipair = 1:n_pairs
    e_vals = extract_animal_means(group_results(ipair).trial_corr_early, 1);
    p_vals = extract_animal_means(group_results(ipair).trial_corr_pre, 1);
    x_vals = extract_animal_means(group_results(ipair).trial_corr_post, 1);
    
    es_vals = extract_animal_means(group_results(ipair).trial_corr_early_shuff, 1);
    ps_vals = extract_animal_means(group_results(ipair).trial_corr_pre_shuff, 1);
    xs_vals = extract_animal_means(group_results(ipair).trial_corr_post_shuff, 1);
    
    if all(isnan(e_vals)), continue; end
    
    nexttile
    means = [mean(e_vals, 'omitnan'), mean(p_vals, 'omitnan'), mean(x_vals, 'omitnan')];
    sems  = [std(e_vals, 'omitnan')/sqrt(sum(~isnan(e_vals))), ...
             std(p_vals, 'omitnan')/sqrt(sum(~isnan(p_vals))), ...
             std(x_vals, 'omitnan')/sqrt(sum(~isnan(x_vals)))];
    shuffs = [mean(es_vals, 'omitnan'), mean(ps_vals, 'omitnan'), mean(xs_vals, 'omitnan')];
    
    bar(1:3, means, 'FaceColor', [0.8 0.8 0.8]); hold on;
    errorbar(1:3, means, sems, 'k.', 'LineWidth', 1.5);
    plot(1:3, shuffs, 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'LineStyle', '--');
    
    xticklabels({'Naive', 'Intermediate', 'Expert'});
    title(group_results(ipair).pair_name);
    ylabel('Correlation (CC1)');
    
    y_stats = [e_vals; p_vals; x_vals];
    g_stats = [ones(length(e_vals),1); 2*ones(length(p_vals),1); 3*ones(length(x_vals),1)];
    valid_mask = ~isnan(y_stats);
    
    if sum(valid_mask) > 3
        [pval_anova, ~, stats] = anova1(y_stats(valid_mask), g_stats(valid_mask), 'off');
        if pval_anova < 0.05
            c = multcompare(stats, 'Display', 'off');
            pairs = {}; pvals_mc = [];
            for k = 1:size(c,1)
                if c(k,6) < 0.05
                    pairs{end+1} = [c(k,1), c(k,2)];
                    pvals_mc(end+1) = c(k,6);
                end
            end
            if ~isempty(pairs), sigstar(pairs, pvals_mc); end
        end
    end
end
linkaxes; save_to_svg(fullfile(save_dir, 'Trial_Corr_Epochs'));

% --- D. Trial Precession (Epochs) ---
figure('Name', 'Group Epoch Precession', 'Color', 'w', 'Position', [100 100 1200 400]);
tiledlayout('flow');
for ipair = 1:n_pairs
    e_vals = extract_animal_means(group_results(ipair).trial_precession_early, 1);
    p_vals = extract_animal_means(group_results(ipair).trial_precession_pre, 1);
    x_vals = extract_animal_means(group_results(ipair).trial_precession_post, 1);
    
    es_vals = extract_animal_means(group_results(ipair).trial_precession_early_shuff, 1);
    ps_vals = extract_animal_means(group_results(ipair).trial_precession_pre_shuff, 1);
    xs_vals = extract_animal_means(group_results(ipair).trial_precession_post_shuff, 1);
    
    if all(isnan(e_vals)), continue; end
    
    nexttile
    means = [mean(e_vals, 'omitnan'), mean(p_vals, 'omitnan'), mean(x_vals, 'omitnan')];
    sems  = [std(e_vals, 'omitnan')/sqrt(sum(~isnan(e_vals))), ...
             std(p_vals, 'omitnan')/sqrt(sum(~isnan(p_vals))), ...
             std(x_vals, 'omitnan')/sqrt(sum(~isnan(x_vals)))];
    shuffs = [mean(es_vals, 'omitnan'), mean(ps_vals, 'omitnan'), mean(xs_vals, 'omitnan')];
    
    bar(1:3, means, 'FaceColor', [0.8 0.5 0.5]); hold on;
    errorbar(1:3, means, sems, 'k.', 'LineWidth', 1.5);
    plot(1:3, shuffs, 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'LineStyle', '--');
    yline(0, '-k', 'LineWidth', 0.5);
    
    xticklabels({'Naive', 'Intermediate', 'Expert'});
    title(group_results(ipair).pair_name);
    ylabel('Precession Index');
    
    % --- Statistics 1: ANOVA & multcompare ---
    y_stats = [e_vals; p_vals; x_vals];
    g_stats = [ones(length(e_vals),1); 2*ones(length(p_vals),1); 3*ones(length(x_vals),1)];
    valid_mask = ~isnan(y_stats);
    
    if sum(valid_mask) > 3
        [pval_anova, ~, stats] = anova1(y_stats(valid_mask), g_stats(valid_mask), 'off');
        if pval_anova < 0.05
            c = multcompare(stats, 'Display', 'off');
            pairs = {}; pvals_mc = [];
            for k = 1:size(c,1)
                if c(k,6) < 0.05
                    pairs{end+1} = [c(k,1), c(k,2)];
                    pvals_mc(end+1) = c(k,6);
                end
            end
            if ~isempty(pairs), sigstar(pairs, pvals_mc); end
        end
    end

    % --- Statistics 2: One-Sample T-Tests against 0 ---
    p_zero = nan(1, 3);
    if sum(~isnan(e_vals)) > 2, [~, p_zero(1)] = ttest(e_vals, 0); end
    if sum(~isnan(p_vals)) > 2, [~, p_zero(2)] = ttest(p_vals, 0); end
    if sum(~isnan(x_vals)) > 2, [~, p_zero(3)] = ttest(x_vals, 0); end
    
    yl = ylim; 
    offset = (yl(2) - yl(1)) * 0.02; 
    min_y_needed = yl(1); 
    
    for idx = 1:3
        if p_zero(idx) < 0.05
            if means(idx) >= 0
                y_pos = means(idx) + sems(idx) + offset;
                vert_align = 'bottom';
            else
                y_pos = means(idx) - sems(idx) - offset;
                vert_align = 'top';
                if y_pos < min_y_needed, min_y_needed = y_pos - offset; end
            end
            text(idx, y_pos, '*', 'FontSize', 18, 'HorizontalAlignment', 'center', 'VerticalAlignment', vert_align);
        end
    end
    if min_y_needed < yl(1), ylim([min_y_needed, yl(2)]); end
end
linkaxes; save_to_svg(fullfile(save_dir, 'Trial_Precession_Epochs'));

%% E. NETWORK VISUALIZATION (Directed Graph)
fprintf('Generating Network Graphs...\n');
% Layout mapped to typical coronal/sagittal abstraction for Striatum/ACC
layout_def.names = {'ACC', 'DMS', 'DLS'};
layout_def.x     = [5.0,   3.0,   7.0];
layout_def.y     = [8.0,   4.0,   4.0];

net_data.early_cc  = zeros(n_pairs, 1);
net_data.early_ifi = zeros(n_pairs, 1);
net_data.post_cc   = zeros(n_pairs, 1);
net_data.post_ifi  = zeros(n_pairs, 1);

for ipair = 1:n_pairs
    e_cc = extract_animal_means(group_results(ipair).trial_corr_early, 1);
    net_data.early_cc(ipair) = mean(e_cc, 'omitnan');
    
    e_ifi = extract_animal_means(group_results(ipair).trial_precession_early, 1);
    net_data.early_ifi(ipair) = mean(e_ifi, 'omitnan');
    
    p_cc = extract_animal_means(group_results(ipair).trial_corr_post, 1);
    net_data.post_cc(ipair) = mean(p_cc, 'omitnan');
    
    p_ifi = extract_animal_means(group_results(ipair).trial_precession_post, 1);
    net_data.post_ifi(ipair) = mean(p_ifi, 'omitnan');
end

figure('Name', 'Network Connectivity', 'Color', 'w', 'Position', [100 100 1600 500]);
t = tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'none');

nexttile(t);
plot_directed_network(area_pairs_to_analyze, net_data.early_cc, net_data.early_ifi, 'Naive', false, layout_def);
nexttile(t);
plot_directed_network(area_pairs_to_analyze, net_data.post_cc, net_data.post_ifi, 'Expert', false, layout_def);

diff_cc  = net_data.post_cc - net_data.early_cc;
diff_ifi = net_data.post_ifi - net_data.early_ifi;
nexttile(t);
plot_directed_network(area_pairs_to_analyze, diff_cc, diff_ifi, 'Difference (Expert - Naive)', true, layout_def);

save_to_svg(fullfile(save_dir, 'Network_Connectivity_Group'));
fprintf('Done.\n');

% --- LOCAL PLOTTING FUNCTION ---
function plot_directed_network(pairs, cc_vals, ifi_vals, titleStr, isDiff, layout)
    regions = unique(pairs(:));
    sources = []; targets = []; weights = []; edge_colors = [];
    n_edges = size(pairs, 1);
    
    for i = 1:n_edges
        u = find(strcmp(regions, pairs{i, 1}));
        v = find(strcmp(regions, pairs{i, 2}));
        cc = cc_vals(i);
        ifi = ifi_vals(i);
        
        if isnan(cc) || isnan(ifi), continue; end
        
        if ifi >= 0, s = u; t = v; else, s = v; t = u; end
        
        sources = [sources; s];
        targets = [targets; t];
        weights = [weights; abs(cc)];
        
        if isDiff
            if cc >= 0, edge_colors = [edge_colors; 0.85, 0.33, 0.10]; 
            else, edge_colors = [edge_colors; 0.00, 0.45, 0.74]; end
        else
            edge_colors = [edge_colors; 0.2, 0.2, 0.2]; 
        end
    end
    
    G = digraph(sources, targets, weights, numel(regions));
    p = plot(G);
    
    x_coords = zeros(1, numel(regions)); y_coords = zeros(1, numel(regions));
    for i = 1:numel(regions)
        idx = find(strcmp(layout.names, regions{i}));
        if ~isempty(idx), x_coords(i) = layout.x(idx); y_coords(i) = layout.y(idx);
        else, x_coords(i) = 0; y_coords(i) = 0; end
    end
    p.XData = x_coords; p.YData = y_coords;
    
    axis equal; axis manual; xlim([0 10]); ylim([0 11]); 
    labelnode(p, 1:numel(regions), regions);
    p.NodeColor = [0.9 0.9 0.9]; p.MarkerSize = 24; p.NodeFontSize = 14; p.NodeFontWeight = 'bold';
    
    max_w = max(G.Edges.Weight);
    if isempty(max_w) || max_w == 0, norm_w = zeros(size(G.Edges.Weight));
    else, norm_w = G.Edges.Weight / max_w; end
    
    p.LineWidth = (norm_w * 7) + 0.5; 
    p.EdgeColor = edge_colors; p.ArrowSize = 15;
    if ~isDiff, p.EdgeAlpha = 0.8; end
    
    axis off; title(titleStr, 'FontSize', 16, 'FontWeight', 'bold');
end

%% 6. LOCAL HELPERS
function p_idx = calc_precession(D1, D2, max_shift, num_ccs, nc1, nc2)
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

function [mu, se] = aggregate_cells(cell_data, n_bins, cc_idx)
    valid_data = cell_data(~cellfun(@isempty, cell_data));
    n_animals = length(valid_data);
    if n_animals == 0, mu = []; se = []; return; end
    
    stack = nan(n_animals, n_bins);
    for i = 1:n_animals
        d = valid_data{i};
        if size(d, 2) == n_bins, stack(i, :) = d(cc_idx, :); end
    end
    mu = mean(stack, 1, 'omitnan');
    se = std(stack, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(stack), 1));
end

function raw_means = extract_animal_means(cell_data, cc_idx)
    n_animals = length(cell_data);
    raw_means = nan(n_animals, 1);
    for i = 1:n_animals
        if ~isempty(cell_data{i})
            raw_means(i) = mean(cell_data{i}(cc_idx, :), 2, 'omitnan');
        end
    end
end

function save_to_svg(fig_name)
    fig = gcf;
    set(fig, 'Renderer', 'painters'); 
    fprintf('Saving %s.svg...\n', fig_name);
    try print(fig, '-dsvg', [fig_name '.svg']);
    catch, print(fig, '-dpng', [fig_name '.png']); end
end

function shadedErrorBar_local(x, y, err, color)
    fill([x, fliplr(x)], [y+err, fliplr(y-err)], color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold on; plot(x, y, 'Color', color, 'LineWidth', 1.5);
end