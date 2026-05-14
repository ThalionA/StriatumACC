%% 1. Configuration & Setup
clear; clc; close all;
cfg = project_cfg();
cfg.data_file = cfg.task_data_file;
cfg.save_file = 'processed_data/shannon_mi_results.mat';
cfg.regions   = cfg.areas;
cfg.colors    = mat2cell(cfg.area_colors, ones(size(cfg.area_colors,1),1), 3)';

% --- Information Theory Parameters (script-specific) ---
% Binning scheme (discretize_zero_aware): bin 1 always holds exact zeros;
% the remaining (n-1) bins hold equipopulated non-zero quantiles. With
% mi_*_bins = 2 the analysis is therefore a binary "no-spike vs spike"
% (or "no-lick vs lick") MI — the smallest non-degenerate scheme for
% sparse signals.
cfg.mi_neural_bins   = 2;
cfg.mi_behav_bins    = 2;
cfg.mi_pool_win      = 5;  % Number of spatial bins to pool per window
cfg.mi_pool_shift    = 1;  % Shift step for the moving window
cfg.min_valid_trials = 8;  % Strict minimum N to compute valid discrete probability distributions
cfg.lp_min_pass      = cfg.lp_min_consecutive;  % alias for legacy code paths below

% Number of trial-permutation shuffles used to estimate the MI null. Inherits
% project default; override here if a single script needs more/fewer.
if ~isfield(cfg, 'n_shuffles') || isempty(cfg.n_shuffles)
    cfg.n_shuffles = 25;
end

% Reproducibility: seed the global RNG once. Per-worker RNG is reseeded as
% rng(cfg.seed + ianimal, 'twister') inside each parfor iteration so that
% shuffle nulls are deterministic regardless of worker count or scheduling.
rng(cfg.seed, 'twister');

%% 2. Data Loading & Preprocessing
fprintf('--- Loading Data ---\n');
load(cfg.data_file, 'preprocessed_data');
n_animals = numel(preprocessed_data);
clean_data = struct();

for ianimal = 1:n_animals
    n_trials_raw = size(preprocessed_data(ianimal).spatial_binned_fr_all, 3);
    diseng_point = preprocessed_data(ianimal).change_point_mean;
    if isnan(diseng_point) || isempty(diseng_point)
        diseng_point = n_trials_raw;
    end
    n_trials = min(n_trials_raw, diseng_point);
    
    licks = preprocessed_data(ianimal).spatial_binned_data.licks(1:n_trials, :);
    durations = preprocessed_data(ianimal).spatial_binned_data.durations(1:n_trials, :);
    durations(durations == 0) = eps; 
    
    lick_rate = licks ./ durations;
    velocity = (4 * 1.25) ./ durations; 
    
    % --- Robust Learning Point (LP) Calculation ---
    % Refactored 2026-05-07. find_learning_points returns the START of the
    % qualifying window (the project-wide convention). MI v2 historically
    % uses the END of that window as its LP, so we shift here to preserve
    % the existing analysis windows.
    if isfield(preprocessed_data(ianimal), 'learning_point') && ~isempty(preprocessed_data(ianimal).learning_point)
        lp = preprocessed_data(ianimal).learning_point;
    else
        lp_cfg = struct('lp_z_threshold', cfg.lp_z_threshold, ...
                        'lp_window',      cfg.lp_window, ...
                        'lp_min_consecutive', cfg.lp_min_pass);
        lps_start = find_learning_points(preprocessed_data(ianimal), lp_cfg);
        if isnan(lps_start)
            lp = NaN;
        else
            lp = lps_start + cfg.lp_window - 1; % MI v2 convention: end-of-window
        end
    end
    
    n_bins_actual = min(size(preprocessed_data(ianimal).spatial_binned_fr_all, 2), cfg.max_bin);
    
    clean_data(ianimal).neural = preprocessed_data(ianimal).spatial_binned_fr_all(:, 1:n_bins_actual, 1:n_trials);
    clean_data(ianimal).lick_rate = lick_rate(:, 1:n_bins_actual)'; 
    clean_data(ianimal).velocity = velocity(:, 1:n_bins_actual)';   
    clean_data(ianimal).lp = lp;
    clean_data(ianimal).n_trials = n_trials;
    clean_data(ianimal).n_bins = n_bins_actual;
    % Pull every area mask via is_area_safe so areas added to project_cfg
    % (CA1, DG, ...) flow through automatically and missing columns on
    % older animals fall back to all-false instead of raising.
    for i_area = 1:numel(cfg.areas)
        area_name = cfg.areas{i_area};
        clean_data(ianimal).(['is_' lower(area_name)]) = ...
            is_area_safe(preprocessed_data(ianimal), area_name);
    end
end

%% 3. Core Functions for Information Theory
% Fast Shannon Mutual Information I(X;Y) with Miller-Madow Bias Correction
calc_mi = @(x, y, nx, ny) compute_mi_mm(x, y, nx, ny);

%% 4. Calculate Pooled Shannon Mutual Information
% Cache key: any change to these fields invalidates the saved .mat.
mi_cache_keys = {'mi_neural_bins','mi_behav_bins','mi_pool_win', ...
                 'mi_pool_shift','min_valid_trials','n_shuffles', ...
                 'max_bin','seed','behav_targets','regions'};
[mi_cache_ok, mi_cache_reason] = check_cache(cfg.save_file, cfg, mi_cache_keys);
if mi_cache_ok
    fprintf('Loading existing MI results...\n');
    load(cfg.save_file, 'mi_results');
else
    if exist(cfg.save_file, 'file')
        fprintf('Cache invalid (%s); recomputing.\n', mi_cache_reason);
    end
    fprintf('--- Computing Shannon Mutual Information (Pooled) ---\n');
    % Preallocate so parfor can write into mi_results(ianimal) safely.
    mi_results(n_animals).win_centers = [];
    if isempty(gcp('nocreate'))
        try, parpool; catch, end %#ok<CTCH>
    end

    cfg_par = cfg; %#ok<NASGU>  % broadcast value to workers
    parfor ianimal = 1:n_animals
        % Per-worker deterministic seed: depends on cfg.seed and animal id
        % only, so results are reproducible across runs/worker counts.
        rng(cfg_par.seed + ianimal, 'twister');
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;

        win_starts = 1 : cfg_par.mi_pool_shift : (n_bins - cfg_par.mi_pool_win + 1);
        n_windows = length(win_starts);
        win_centers = win_starts + floor(cfg_par.mi_pool_win / 2);
        mi_results(ianimal).win_centers = win_centers;

        epochs = epochs_from_lp(lp, n_trials);

        for i_targ = 1:numel(cfg_par.behav_targets)
            target_name = cfg_par.behav_targets{i_targ};
            Y_all = clean_data(ianimal).(target_name);

            for i_reg = 1:numel(cfg_par.regions)
                region = cfg_par.regions{i_reg};
                idx_reg = clean_data(ianimal).(['is_' lower(region)]);
                if sum(idx_reg) == 0; continue; end

                reg_neural = clean_data(ianimal).neural(idx_reg, :, :);
                n_units = size(reg_neural, 1);

                for i_ep = 1:3
                    tr_idx = epochs{i_ep};
                    if length(tr_idx) < cfg_par.min_valid_trials
                        mi_results(ianimal).(region).(target_name).epoch(i_ep).real         = nan(n_units, n_windows);
                        mi_results(ianimal).(region).(target_name).epoch(i_ep).shuffle_mean = nan(n_units, n_windows);
                        mi_results(ianimal).(region).(target_name).epoch(i_ep).shuffle_p95  = nan(n_units, n_windows);
                        continue;
                    end

                    Y_ep = Y_all(:, tr_idx);
                    X_ep = reg_neural(:, :, tr_idx);
                    n_tr_ep = length(tr_idx);
                    mi_real_mat = nan(n_units, n_windows);
                    mi_null_mean = nan(n_units, n_windows);
                    mi_null_p95  = nan(n_units, n_windows);

                    % Pre-generate trial permutations so every unit in this
                    % window sees the same null. Identity is implicit (real).
                    for w = 1:n_windows
                        b_idx = win_starts(w) : (win_starts(w) + cfg_par.mi_pool_win - 1);
                        Y_win = Y_ep(b_idx, :);
                        Y_disc_2d = reshape(discretize_zero_aware(Y_win(:), cfg_par.mi_behav_bins), size(Y_win));
                        if sum(~isnan(Y_disc_2d(:))) < cfg_par.min_valid_trials; continue; end

                        perms = zeros(n_tr_ep, cfg_par.n_shuffles);
                        for s = 1:cfg_par.n_shuffles
                            perms(:, s) = randperm(n_tr_ep);
                        end

                        for u = 1:n_units
                            X_win = squeeze(X_ep(u, b_idx, :));
                            X_disc_2d = reshape(discretize_zero_aware(X_win(:), cfg_par.mi_neural_bins), size(X_win));

                            X_flat = X_disc_2d(:); Y_flat = Y_disc_2d(:);
                            valid_mask = ~isnan(X_flat) & ~isnan(Y_flat);
                            if sum(valid_mask) < cfg_par.min_valid_trials; continue; end

                            mi_real_mat(u, w) = calc_mi(X_flat(valid_mask), Y_flat(valid_mask), ...
                                cfg_par.mi_neural_bins, cfg_par.mi_behav_bins);

                            null_vals = nan(1, cfg_par.n_shuffles);
                            for s = 1:cfg_par.n_shuffles
                                Y_shuf_flat = reshape(Y_disc_2d(:, perms(:, s)), [], 1);
                                vm = ~isnan(X_flat) & ~isnan(Y_shuf_flat);
                                if sum(vm) < cfg_par.min_valid_trials; continue; end
                                null_vals(s) = calc_mi(X_flat(vm), Y_shuf_flat(vm), ...
                                    cfg_par.mi_neural_bins, cfg_par.mi_behav_bins);
                            end
                            mi_null_mean(u, w) = mean(null_vals, 'omitnan');
                            mi_null_p95(u, w)  = prctile(null_vals, 95);
                        end
                    end
                    mi_results(ianimal).(region).(target_name).epoch(i_ep).real         = mi_real_mat;
                    mi_results(ianimal).(region).(target_name).epoch(i_ep).shuffle_mean = mi_null_mean;
                    mi_results(ianimal).(region).(target_name).epoch(i_ep).shuffle_p95  = mi_null_p95;
                end
            end
        end
    end
    cache_meta = cache_meta_from_cfg(cfg, mi_cache_keys); %#ok<NASGU>
    save(cfg.save_file, 'mi_results', 'cache_meta', '-v7.3');
    fprintf('Mutual Information computation complete.\n');
end

%% 5. Plotting: Spatial Evolution of Information Encoding
fprintf('--- Plotting MI Spatial Profiles ---\n');
epoch_names = {'Naive (Trials 1-10)', 'Pre-LP (-10 to -1)', 'Post-LP (+1 to +10)'};
for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    figure('Position', [100 150 1500 500], 'Color', 'w', 'Name', sprintf('Shannon MI - %s', target));
    t_mi = tiledlayout(1, 3, 'Padding', 'compact');
    
    for i_ep = 1:3
        ax = nexttile(t_mi); hold(ax, 'on');
        legend_handles = [];
        legend_labels  = {};
        plot_x_all = [];

        for i_reg = 1:numel(cfg.regions)
            region = cfg.regions{i_reg};
            mi_curves_real = [];
            mi_curves_null = [];
            common_x = [];

            for ianimal = 1:n_animals
                if isfield(mi_results(ianimal), region) && isfield(mi_results(ianimal).(region), target)
                    if length(mi_results(ianimal).(region).(target).epoch) >= i_ep
                        ep_data = mi_results(ianimal).(region).(target).epoch(i_ep);
                        real_mat = ep_data.real;
                        common_x = mi_results(ianimal).win_centers;
                        if ~isempty(real_mat)
                            mi_curves_real = [mi_curves_real; mean(real_mat, 1, 'omitnan')];
                            if isfield(ep_data, 'shuffle_mean') && ~isempty(ep_data.shuffle_mean)
                                mi_curves_null = [mi_curves_null; mean(ep_data.shuffle_mean, 1, 'omitnan')];
                            end
                        end
                    end
                end
            end

            if ~isempty(mi_curves_real)
                mu_real = mean(mi_curves_real, 1, 'omitnan');
                se_real = std(mi_curves_real, 0, 1, 'omitnan') ./ ...
                          sqrt(sum(~isnan(mi_curves_real), 1));
                h = shadedErrorBar(common_x, mu_real, se_real, 'lineprops', {'Color', cfg.colors{i_reg}, 'LineWidth', 2.5});
                legend_handles(end+1) = h.mainLine;
                legend_labels{end+1}  = region;
                plot_x_all = [plot_x_all, common_x];

                if ~isempty(mi_curves_null)
                    mu_null = mean(mi_curves_null, 1, 'omitnan');
                    plot(common_x, mu_null, '--', 'Color', cfg.colors{i_reg}, 'LineWidth', 1, 'HandleVisibility', 'off');
                end
            end
        end

        xline(cfg.target_rz_bin, 'r-', 'Reward Zone', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
        if ~isempty(plot_x_all)
            xlim([min(plot_x_all) max(plot_x_all)]);
        end

        title(sprintf('%s', epoch_names{i_ep}), 'FontSize', 14);
        if i_ep == 1; ylabel('Corrected MI (Bits, dashed = trial-shuffle null)', 'FontWeight', 'bold', 'FontSize', 12); end
        if i_ep == 2; xlabel('Spatial Bin (Window Center)', 'FontSize', 12); end
        if i_ep == 3 && ~isempty(legend_handles)
            legend(legend_handles, legend_labels, 'Location', 'northwest');
        end
        box on; grid on;
    end
    linkaxes
    sgtitle(sprintf('Task Encoding: Single-Unit Shannon Information (%s)', strrep(target, '_', ' ')), 'FontSize', 16);
end

%% 6. Calculate Cross-Spatial (Lagged) Mutual Information (Neural -> Behavior)
fprintf('--- Computing Cross-Spatial Mutual Information ---\n');
cfg.mi_max_lag = 15;
cross_mi_file = 'cross_spatial_mi_results.mat';
cross_mi_keys = [mi_cache_keys, {'mi_max_lag'}];
[xmi_ok, xmi_reason] = check_cache(cross_mi_file, cfg, cross_mi_keys);
if xmi_ok
    fprintf('Loading existing Cross-Spatial MI results...\n');
    load(cross_mi_file, 'cross_mi_results');
else
    if exist(cross_mi_file, 'file')
        fprintf('Cache invalid (%s); recomputing.\n', xmi_reason);
    end
    cross_mi_results(n_animals).win_centers = [];
    if isempty(gcp('nocreate'))
        try, parpool; catch, end %#ok<CTCH>
    end

    cfg_par = cfg; %#ok<NASGU>
    parfor ianimal = 1:n_animals
        rng(cfg_par.seed + ianimal, 'twister');
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;

        win_starts = 1 : cfg_par.mi_pool_shift : (n_bins - cfg_par.mi_pool_win + 1);
        n_windows = length(win_starts);
        cross_mi_results(ianimal).win_centers = win_starts + floor(cfg_par.mi_pool_win / 2);

        epochs = epochs_from_lp(lp, n_trials);

        for i_targ = 1:numel(cfg_par.behav_targets)
            target_name = cfg_par.behav_targets{i_targ};
            Y_all = clean_data(ianimal).(target_name);

            for i_reg = 1:numel(cfg_par.regions)
                region = cfg_par.regions{i_reg};
                idx_reg = clean_data(ianimal).(['is_' lower(region)]);
                if sum(idx_reg) == 0; continue; end

                reg_neural = clean_data(ianimal).neural(idx_reg, :, :);
                n_units = size(reg_neural, 1);

                for i_ep = 1:3
                    tr_idx = epochs{i_ep};
                    if length(tr_idx) < cfg_par.min_valid_trials; continue; end

                    Y_ep = Y_all(:, tr_idx);
                    X_ep = reg_neural(:, :, tr_idx);
                    n_tr_ep = length(tr_idx);
                    mi_real_mat = nan(n_units, n_windows, n_windows);
                    mi_null_mean = nan(n_units, n_windows, n_windows);
                    mi_null_p95  = nan(n_units, n_windows, n_windows);

                    % Pre-discretise X per (unit, source-window) to avoid
                    % redundant work inside the (target, source) loop.
                    X_disc_cache = cell(n_units, n_windows);

                    for w_targ = 1:n_windows
                        b_idx_targ = win_starts(w_targ) : (win_starts(w_targ) + cfg_par.mi_pool_win - 1);
                        Y_win = Y_ep(b_idx_targ, :);
                        Y_disc_2d = reshape(discretize_zero_aware(Y_win(:), cfg_par.mi_behav_bins), size(Y_win));
                        if sum(~isnan(Y_disc_2d(:))) < cfg_par.min_valid_trials; continue; end

                        perms = zeros(n_tr_ep, cfg_par.n_shuffles);
                        for s = 1:cfg_par.n_shuffles
                            perms(:, s) = randperm(n_tr_ep);
                        end

                        w_source_start = max(1, w_targ - cfg_par.mi_max_lag);
                        for w_source = w_source_start:w_targ
                            b_idx_source = win_starts(w_source) : (win_starts(w_source) + cfg_par.mi_pool_win - 1);

                            for u = 1:n_units
                                if isempty(X_disc_cache{u, w_source})
                                    X_win = squeeze(X_ep(u, b_idx_source, :));
                                    X_disc_cache{u, w_source} = ...
                                        reshape(discretize_zero_aware(X_win(:), cfg_par.mi_neural_bins), size(X_win));
                                end
                                X_disc_2d = X_disc_cache{u, w_source};

                                X_flat = X_disc_2d(:); Y_flat = Y_disc_2d(:);
                                valid_mask = ~isnan(X_flat) & ~isnan(Y_flat);
                                if sum(valid_mask) < cfg_par.min_valid_trials; continue; end

                                mi_real_mat(u, w_targ, w_source) = calc_mi(X_flat(valid_mask), Y_flat(valid_mask), ...
                                    cfg_par.mi_neural_bins, cfg_par.mi_behav_bins);

                                null_vals = nan(1, cfg_par.n_shuffles);
                                for s = 1:cfg_par.n_shuffles
                                    Y_shuf_flat = reshape(Y_disc_2d(:, perms(:, s)), [], 1);
                                    vm = ~isnan(X_flat) & ~isnan(Y_shuf_flat);
                                    if sum(vm) < cfg_par.min_valid_trials; continue; end
                                    null_vals(s) = calc_mi(X_flat(vm), Y_shuf_flat(vm), ...
                                        cfg_par.mi_neural_bins, cfg_par.mi_behav_bins);
                                end
                                mi_null_mean(u, w_targ, w_source) = mean(null_vals, 'omitnan');
                                mi_null_p95(u, w_targ, w_source)  = prctile(null_vals, 95);
                            end
                        end
                    end
                    cross_mi_results(ianimal).(region).(target_name).epoch(i_ep).real         = mi_real_mat;
                    cross_mi_results(ianimal).(region).(target_name).epoch(i_ep).shuffle_mean = mi_null_mean;
                    cross_mi_results(ianimal).(region).(target_name).epoch(i_ep).shuffle_p95  = mi_null_p95;
                end
            end
        end
    end
    cache_meta = cache_meta_from_cfg(cfg, cross_mi_keys); %#ok<NASGU>
    save(cross_mi_file, 'cross_mi_results', 'cache_meta', '-v7.3');
    fprintf('Cross-Spatial MI computation complete.\n');
end

%% 7. Plotting: Spatial Lag Profiles by Zone
fprintf('--- Plotting MI Lags by Target Zone ---\n');
zone_defs = {
    'Early Corridor (Bins 10-15)', @(x) x >= 10 & x <= 15;
    'Visual Landmark (Bins 20-25)', @(x) x >= 20 & x <= 25;
    'Reward Zone (Bins 25-30)',    @(x) x >= 25 & x <= 30;
};
zone_colors = {[0.5 0.5 0.5], [0.9290 0.6940 0.1250], [0.4660 0.6740 0.1880]};

for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    for i_reg = 1:numel(cfg.regions)
        region = cfg.regions{i_reg};
        
        figure('Position', [100 100 1200 400], 'Color', 'w', 'Name', sprintf('Lag Profiles %s - %s', region, target));
        t_zones = tiledlayout(1, 3, 'Padding', 'compact');
        
        for i_ep = 1:3
            ax = nexttile(t_zones); hold(ax, 'on');
            legend_handles = [];
            legend_labels  = {};

            for z = 1:size(zone_defs, 1)
                lag_curves = [];
                for ianimal = 1:n_animals
                    if isfield(cross_mi_results(ianimal), region) && isfield(cross_mi_results(ianimal).(region), target)
                        if length(cross_mi_results(ianimal).(region).(target).epoch) >= i_ep
                            centers = cross_mi_results(ianimal).win_centers;
                            real_mat = cross_mi_results(ianimal).(region).(target).epoch(i_ep).real;

                            if isempty(real_mat); continue; end
                            target_idxs = find(zone_defs{z, 2}(centers));
                            if isempty(target_idxs); continue; end

                            mouse_lag_profile = nan(length(target_idxs), cfg.mi_max_lag + 1);
                            for t_idx = 1:length(target_idxs)
                                wy = target_idxs(t_idx);
                                for lag = 0:cfg.mi_max_lag
                                    wx = wy - lag;
                                    if wx >= 1
                                        mouse_lag_profile(t_idx, lag + 1) = mean(real_mat(:, wy, wx), 1, 'omitnan');
                                    end
                                end
                            end
                            lag_curves = [lag_curves; mean(mouse_lag_profile, 1, 'omitnan')];
                        end
                    end
                end

                if ~isempty(lag_curves)
                    mu = mean(lag_curves, 1, 'omitnan');
                    se = std(lag_curves, 0, 1, 'omitnan') ./ ...
                         sqrt(sum(~isnan(lag_curves), 1));
                    h = shadedErrorBar(0:cfg.mi_max_lag, mu, se, 'lineprops', {'Color', zone_colors{z}, 'LineWidth', 2});
                    legend_handles(end+1) = h.mainLine;
                    legend_labels{end+1}  = zone_defs{z, 1};
                end
            end

            yline(0, 'k--');
            xlim([0 cfg.mi_max_lag]);
            title(epoch_names{i_ep}, 'FontSize', 13);
            if i_ep == 1; ylabel('Corrected MI (Bits)'); end
            xlabel('Spatial Lag (Target Bin - Source Bin)');
            if i_ep == 3 && ~isempty(legend_handles)
                legend(legend_handles, legend_labels, 'Location', 'northeast');
            end
            box on; grid on;
        end
        sgtitle(sprintf('Predictive Horizon by Spatial Zone: %s encoding %s', region, strrep(target, '_', ' ')), 'FontSize', 16);
    end
end

%% 8. Calculate Cross-Area Mutual Information (Information Flow using PC1)
fprintf('--- Computing Cross-Area Mutual Information ---\n');
cross_area_file = 'cross_area_mi_results.mat';
% Script-local subset of cfg.area_pairs. project_cfg lists all 15 pairs
% across the 6 areas, but the cross-area MI / MMI plots (sections 9 + 12)
% expect 6. Edit this list, not project_cfg, if you want to add/remove
% pairs from the MI flow analysis.
cfg.area_pairs = {'ACC', 'DMS'; 'ACC', 'DLS'; 'DMS', 'DLS'; ...
                  'V1',  'DMS'; 'V1',  'DLS'; 'V1',  'ACC'};
cross_area_keys = [mi_cache_keys, {'area_pairs'}];
[xa_ok, xa_reason] = check_cache(cross_area_file, cfg, cross_area_keys);
if xa_ok
    fprintf('Loading existing Cross-Area MI results...\n');
    load(cross_area_file, 'area_mi_results');
else
    if exist(cross_area_file, 'file')
        fprintf('Cache invalid (%s); recomputing.\n', xa_reason);
    end
    area_mi_results(n_animals).win_centers = [];
    if isempty(gcp('nocreate'))
        try, parpool; catch, end %#ok<CTCH>
    end

    cfg_par = cfg; %#ok<NASGU>
    parfor ianimal = 1:n_animals
        rng(cfg_par.seed + ianimal, 'twister');
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;

        win_starts = 1 : cfg_par.mi_pool_shift : (n_bins - cfg_par.mi_pool_win + 1);
        n_windows = length(win_starts);
        area_mi_results(ianimal).win_centers = win_starts + floor(cfg_par.mi_pool_win / 2);

        epochs = epochs_from_lp(lp, n_trials);

        % Fit PC1 per area ONCE on all trials so that the same coordinate
        % system is used across epochs (otherwise MI on PC1 in one epoch
        % is incommensurate with MI on PC1 in another).
        pc1_by_area = struct();
        for i_area = 1:numel(cfg_par.areas)
            an = cfg_par.areas{i_area};
            idx_a = clean_data(ianimal).(['is_' lower(an)]);
            if sum(idx_a) == 0
                pc1_by_area.(an) = [];
            else
                pc1_by_area.(an) = extract_pc1(clean_data(ianimal).neural(idx_a, :, :));
            end
        end

        for p = 1:size(cfg_par.area_pairs, 1)
            regA = cfg_par.area_pairs{p, 1}; regB = cfg_par.area_pairs{p, 2};
            pair_name = sprintf('%s_%s', regA, regB);
            popA_all = pc1_by_area.(regA); popB_all = pc1_by_area.(regB);
            if isempty(popA_all) || isempty(popB_all); continue; end

            for i_ep = 1:3
                tr_idx = epochs{i_ep};
                if length(tr_idx) < cfg_par.min_valid_trials; continue; end

                popA = popA_all(:, tr_idx);
                popB = popB_all(:, tr_idx);
                n_tr_ep = length(tr_idx);

                % Pre-discretise A and B per source-window once.
                discA_cache = cell(1, n_windows);
                discB_cache = cell(1, n_windows);
                for w = 1:n_windows
                    bw = win_starts(w) : (win_starts(w) + cfg_par.mi_pool_win - 1);
                    winA = popA(bw, :);
                    winB = popB(bw, :);
                    discA_cache{w} = reshape(discretize_zero_aware(winA(:), cfg_par.mi_neural_bins), size(winA));
                    discB_cache{w} = reshape(discretize_zero_aware(winB(:), cfg_par.mi_neural_bins), size(winB));
                end

                perms = zeros(n_tr_ep, cfg_par.n_shuffles);
                for s = 1:cfg_par.n_shuffles
                    perms(:, s) = randperm(n_tr_ep);
                end

                mi_mat       = nan(n_windows, n_windows);
                mi_null_mean = nan(n_windows, n_windows);
                mi_null_p95  = nan(n_windows, n_windows);

                for wA = 1:n_windows
                    discA_2d = discA_cache{wA};
                    for wB = 1:n_windows
                        discB_2d = discB_cache{wB};
                        flatA = discA_2d(:); flatB = discB_2d(:);
                        valid_mask = ~isnan(flatA) & ~isnan(flatB);
                        if sum(valid_mask) < cfg_par.min_valid_trials; continue; end

                        mi_mat(wA, wB) = calc_mi(flatA(valid_mask), flatB(valid_mask), ...
                            cfg_par.mi_neural_bins, cfg_par.mi_neural_bins);

                        null_vals = nan(1, cfg_par.n_shuffles);
                        for s = 1:cfg_par.n_shuffles
                            flatB_shuf = reshape(discB_2d(:, perms(:, s)), [], 1);
                            vm = ~isnan(flatA) & ~isnan(flatB_shuf);
                            if sum(vm) < cfg_par.min_valid_trials; continue; end
                            null_vals(s) = calc_mi(flatA(vm), flatB_shuf(vm), ...
                                cfg_par.mi_neural_bins, cfg_par.mi_neural_bins);
                        end
                        mi_null_mean(wA, wB) = mean(null_vals, 'omitnan');
                        mi_null_p95(wA, wB)  = prctile(null_vals, 95);
                    end
                end
                area_mi_results(ianimal).(pair_name).epoch(i_ep).real         = mi_mat;
                area_mi_results(ianimal).(pair_name).epoch(i_ep).shuffle_mean = mi_null_mean;
                area_mi_results(ianimal).(pair_name).epoch(i_ep).shuffle_p95  = mi_null_p95;
            end
        end
    end
    cache_meta = cache_meta_from_cfg(cfg, cross_area_keys); %#ok<NASGU>
    save(cross_area_file, 'area_mi_results', 'cache_meta', '-v7.3');
    fprintf('Cross-Area MI computation complete.\n');
end

%% 9. Plotting: Cross-Area Information Flow
fprintf('--- Plotting Cross-Area Information Flow ---\n');
cfg.area_max_lag = 10;
lags_to_plot = -cfg.area_max_lag : cfg.area_max_lag;
figure('Position', [100 600 1500 500], 'Color', 'w', 'Name', 'Cross-Area Mutual Information');
t_flow = tiledlayout(1, 3, 'Padding', 'compact');
pair_colors = lines(size(cfg.area_pairs, 1));

for i_ep = 1:3
    ax = nexttile(t_flow); hold(ax, 'on');
    legend_handles = [];
    legend_labels  = {};

    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1}; regB = cfg.area_pairs{p, 2};
        pair_name = sprintf('%s_%s', regA, regB);

        lag_curves      = [];
        lag_curves_null = [];
        for ianimal = 1:n_animals
            if isfield(area_mi_results(ianimal), pair_name)
                pair_data = area_mi_results(ianimal).(pair_name);
                if isstruct(pair_data) && isfield(pair_data, 'epoch') && length(pair_data.epoch) >= i_ep
                    mat = pair_data.epoch(i_ep).real;
                    if isempty(mat); continue; end

                    mouse_lags = nan(1, length(lags_to_plot));
                    for l_idx = 1:length(lags_to_plot)
                        mouse_lags(l_idx) = mean(diag(mat, -lags_to_plot(l_idx)), 'omitnan');
                    end
                    lag_curves = [lag_curves; mouse_lags];

                    if isfield(pair_data.epoch(i_ep), 'shuffle_mean') && ~isempty(pair_data.epoch(i_ep).shuffle_mean)
                        nm = pair_data.epoch(i_ep).shuffle_mean;
                        mouse_lags_n = nan(1, length(lags_to_plot));
                        for l_idx = 1:length(lags_to_plot)
                            mouse_lags_n(l_idx) = mean(diag(nm, -lags_to_plot(l_idx)), 'omitnan');
                        end
                        lag_curves_null = [lag_curves_null; mouse_lags_n];
                    end
                end
            end
        end

        if ~isempty(lag_curves)
            mu = mean(lag_curves, 1, 'omitnan');
            se = std(lag_curves, 0, 1, 'omitnan') ./ ...
                 sqrt(sum(~isnan(lag_curves), 1));
            h = shadedErrorBar(lags_to_plot, mu, se, 'lineprops', {'Color', pair_colors(p,:), 'LineWidth', 2});
            legend_handles(end+1) = h.mainLine;
            legend_labels{end+1}  = sprintf('%s \\leftrightarrow %s', regA, regB);

            if ~isempty(lag_curves_null)
                mu_null = mean(lag_curves_null, 1, 'omitnan');
                plot(lags_to_plot, mu_null, '--', 'Color', pair_colors(p,:), 'LineWidth', 1, 'HandleVisibility', 'off');
            end
        end
    end

    xline(0, 'k--', 'LineWidth', 1.5);
    xlim([-cfg.area_max_lag, cfg.area_max_lag]);

    title(epoch_names{i_ep}, 'FontSize', 14);
    if i_ep == 1; ylabel('Corrected PC1 MI (Bits, dashed = trial-shuffle null)', 'FontWeight', 'bold'); end
    xlabel('Spatial Lag (Bins)');
    if i_ep == 3
        text(0.1, -0.15, sprintf('\\leftarrow %s leads', 'Area B'), 'Units', 'normalized', 'HorizontalAlignment', 'left');
        text(0.9, -0.15, sprintf('%s leads \\rightarrow', 'Area A'), 'Units', 'normalized', 'HorizontalAlignment', 'right');
        if ~isempty(legend_handles)
            legend(legend_handles, legend_labels, 'Location', 'northwest');
        end
    end
    box on; grid on;
end
sgtitle('Cross-Area Information Flow Across Learning', 'FontSize', 16);

%% 10. Calculate Minimum Mutual Information (MMI Bound for PID)
fprintf('--- Computing MMI Shared Information Bound ---\n');
cfg.pid_max_lag = 10;
pid_file = 'pid_shared_info_results.mat';
pid_keys = [mi_cache_keys, {'area_pairs', 'pid_max_lag'}];
[pid_ok, pid_reason] = check_cache(pid_file, cfg, pid_keys);
if pid_ok
    fprintf('Loading existing MMI results...\n');
    load(pid_file, 'pid_results');
else
    if exist(pid_file, 'file')
        fprintf('Cache invalid (%s); recomputing.\n', pid_reason);
    end
    pid_results(n_animals).win_centers = [];
    if isempty(gcp('nocreate'))
        try, parpool; catch, end %#ok<CTCH>
    end

    cfg_par = cfg; %#ok<NASGU>
    parfor ianimal = 1:n_animals
        rng(cfg_par.seed + ianimal, 'twister');
        fprintf('Processing Animal %d/%d for MMI...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;

        win_starts = 1 : cfg_par.mi_pool_shift : (n_bins - cfg_par.mi_pool_win + 1);
        n_windows = length(win_starts);
        pid_results(ianimal).win_centers = win_starts + floor(cfg_par.mi_pool_win / 2);

        epochs = epochs_from_lp(lp, n_trials);

        % PC1 per area, fit once on all trials so the coordinate system is
        % shared across epochs.
        pc1_by_area = struct();
        for i_area = 1:numel(cfg_par.areas)
            an = cfg_par.areas{i_area};
            idx_a = clean_data(ianimal).(['is_' lower(an)]);
            if sum(idx_a) == 0
                pc1_by_area.(an) = [];
            else
                pc1_by_area.(an) = extract_pc1(clean_data(ianimal).neural(idx_a, :, :));
            end
        end

        for i_targ = 1:numel(cfg_par.behav_targets)
            target_name = cfg_par.behav_targets{i_targ};
            Y_all = clean_data(ianimal).(target_name);

            for i_ep = 1:3
                tr_idx = epochs{i_ep};
                if length(tr_idx) < cfg_par.min_valid_trials; continue; end

                for p = 1:size(cfg_par.area_pairs, 1)
                    regA = cfg_par.area_pairs{p, 1}; regB = cfg_par.area_pairs{p, 2};
                    pair_name = sprintf('%s_to_%s', regA, regB);

                    popA_all = pc1_by_area.(regA); popB_all = pc1_by_area.(regB);
                    if isempty(popA_all) || isempty(popB_all); continue; end

                    popA = popA_all(:, tr_idx);
                    popB = popB_all(:, tr_idx);
                    behav = Y_all(:, tr_idx);
                    n_tr_ep = length(tr_idx);

                    % Pre-discretise all three streams per window.
                    A_disc_cache = cell(1, n_windows);
                    B_disc_cache = cell(1, n_windows);
                    F_disc_cache = cell(1, n_windows);
                    for w = 1:n_windows
                        bw = win_starts(w) : (win_starts(w) + cfg_par.mi_pool_win - 1);
                        A_disc_cache{w} = reshape(discretize_zero_aware(reshape(popA(bw,:),[],1), cfg_par.mi_neural_bins), [length(bw), n_tr_ep]);
                        B_disc_cache{w} = reshape(discretize_zero_aware(reshape(popB(bw,:),[],1), cfg_par.mi_neural_bins), [length(bw), n_tr_ep]);
                        F_disc_cache{w} = reshape(discretize_zero_aware(reshape(behav(bw,:),[],1), cfg_par.mi_behav_bins),  [length(bw), n_tr_ep]);
                    end

                    perms = zeros(n_tr_ep, cfg_par.n_shuffles);
                    for s = 1:cfg_par.n_shuffles
                        perms(:, s) = randperm(n_tr_ep);
                    end

                    n_lags = 2 * cfg_par.pid_max_lag + 1;
                    si_mat       = nan(n_windows, n_lags);
                    si_null_mean = nan(n_windows, n_lags);
                    si_null_p95  = nan(n_windows, n_lags);

                    for w_targ = 1:n_windows
                        F_disc_2d = F_disc_cache{w_targ};
                        B_disc_2d = B_disc_cache{w_targ};
                        F_flat = F_disc_2d(:);
                        B_flat = B_disc_2d(:);

                        for lag_val = -cfg_par.pid_max_lag : cfg_par.pid_max_lag
                            lag_idx = lag_val + cfg_par.pid_max_lag + 1;
                            w_source = w_targ - lag_val;
                            if w_source < 1 || w_source > n_windows; continue; end

                            A_disc_2d = A_disc_cache{w_source};
                            A_flat = A_disc_2d(:);

                            valid = ~isnan(A_flat) & ~isnan(B_flat) & ~isnan(F_flat);
                            if sum(valid) < cfg_par.min_valid_trials; continue; end

                            I_A_F = calc_mi(A_flat(valid), F_flat(valid), cfg_par.mi_neural_bins, cfg_par.mi_behav_bins);
                            I_B_F = calc_mi(B_flat(valid), F_flat(valid), cfg_par.mi_neural_bins, cfg_par.mi_behav_bins);
                            si_mat(w_targ, lag_idx) = min(I_A_F, I_B_F);

                            % Null: permute behaviour trial labels.
                            null_vals = nan(1, cfg_par.n_shuffles);
                            for s = 1:cfg_par.n_shuffles
                                F_shuf_flat = reshape(F_disc_2d(:, perms(:, s)), [], 1);
                                vm = ~isnan(A_flat) & ~isnan(B_flat) & ~isnan(F_shuf_flat);
                                if sum(vm) < cfg_par.min_valid_trials; continue; end
                                I_A_F_n = calc_mi(A_flat(vm), F_shuf_flat(vm), cfg_par.mi_neural_bins, cfg_par.mi_behav_bins);
                                I_B_F_n = calc_mi(B_flat(vm), F_shuf_flat(vm), cfg_par.mi_neural_bins, cfg_par.mi_behav_bins);
                                null_vals(s) = min(I_A_F_n, I_B_F_n);
                            end
                            si_null_mean(w_targ, lag_idx) = mean(null_vals, 'omitnan');
                            si_null_p95(w_targ, lag_idx)  = prctile(null_vals, 95);
                        end
                    end
                    pid_results(ianimal).(pair_name).(target_name).epoch(i_ep).real         = si_mat;
                    pid_results(ianimal).(pair_name).(target_name).epoch(i_ep).shuffle_mean = si_null_mean;
                    pid_results(ianimal).(pair_name).(target_name).epoch(i_ep).shuffle_p95  = si_null_p95;
                end
            end
        end
    end
    cache_meta = cache_meta_from_cfg(cfg, pid_keys); %#ok<NASGU>
    save(pid_file, 'pid_results', 'cache_meta', '-v7.3');
    fprintf('MMI Shared Information computation complete.\n');
end

%% 11. Plotting: Bidirectional MMI Shared Information Bound
fprintf('--- Plotting Space-Lagged MMI Heatmaps ---\n');
lag_vector = -cfg.pid_max_lag : cfg.pid_max_lag;
for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    
    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1}; regB = cfg.area_pairs{p, 2}; 
        pair_name = sprintf('%s_to_%s', regA, regB);
        
        figure('Position', [100 150 1400 450], 'Color', 'w', 'Name', sprintf('MMI Heatmap %s - %s', pair_name, target));
        t_pid = tiledlayout(1, 3, 'Padding', 'compact');
        
        for i_ep = 1:3
            ax = nexttile(t_pid);
            si_all = [];
            for ianimal = 1:n_animals
                if isfield(pid_results(ianimal), pair_name)
                    pair_data = pid_results(ianimal).(pair_name);
                    if isstruct(pair_data) && isfield(pair_data, target) && length(pair_data.(target).epoch) >= i_ep
                        real_mat = pair_data.(target).epoch(i_ep).real;
                        if ~isempty(real_mat)
                            si_all = cat(3, si_all, real_mat);
                        end
                    end
                end
            end
            
            if ~isempty(si_all)
                mu_si = mean(si_all, 3, 'omitnan');
                mask = ~isnan(mu_si);
                % NaN-aware Gaussian smoothing: smooth signal*mask and mask
                % separately, then divide. This prevents NaN-bordered cells
                % from being pulled toward zero (which the previous
                % zero-fill + symmetric-pad approach did at edges).
                sigma = 0.8;
                signal_zeroed = mu_si;
                signal_zeroed(~mask) = 0;
                num = imgaussfilt(signal_zeroed, sigma, ...
                    'FilterDomain', 'spatial', 'Padding', 'symmetric');
                den = imgaussfilt(double(mask),  sigma, ...
                    'FilterDomain', 'spatial', 'Padding', 'symmetric');
                den(den < 1e-12) = nan;
                smoothed_si = num ./ den;
                smoothed_si(~mask) = nan;

                imagesc(1:size(smoothed_si,1), lag_vector, smoothed_si');
                colormap(gca, 'magma'); 
                
                if i_ep == 3; cb = colorbar; cb.Label.String = 'MMI Upper Bound (Bits)'; end
                
                hold on;
                yline(0, 'w--', 'LineWidth', 1); 
                xline(cfg.target_rz_bin, 'w:', 'LineWidth', 1.5); 
                hold off;
                
                axis xy; 
                xlim([1 size(smoothed_si,1)]);
                ylim([-cfg.pid_max_lag, cfg.pid_max_lag]);
                
                title(epoch_names{i_ep}, 'FontSize', 13);
                xlabel('Spatial Bin (Behavior)');
                
                if i_ep == 1
                    ylabel({'Spatial Lag (Bins)', sprintf('(- = %s Leads | + = %s Leads)', regB, regA)}, 'FontWeight', 'bold'); 
                end
            end
        end
        sgtitle(sprintf('Redundancy Upper Bound (MMI): %s / %s encoding %s', regA, regB, strrep(target, '_', ' ')), 'FontSize', 16);
    end
end

%% 12. Plotting: Summary Quantifications of MMI Asymmetry
fprintf('--- Plotting Summary MMI Quantifications ---\n');
for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    figure('Position', [100 650 1500 800], 'Color', 'w', 'Name', sprintf('MMI Summary - %s', target));
    t_sum = tiledlayout(2, 3, 'Padding', 'compact');
    epoch_cols = {[0.298, 0.447, 0.690], [0.867, 0.518, 0.322], [0.333, 0.776, 0.333]}; 
    
    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1}; regB = cfg.area_pairs{p, 2};
        pair_name = sprintf('%s_to_%s', regA, regB);
        ax = nexttile(t_sum); hold(ax, 'on');
        legend_handles = [];
        legend_labels  = {};

        for i_ep = 1:3
            mouse_lag_curves = [];
            for ianimal = 1:n_animals
                if isfield(pid_results(ianimal), pair_name)
                    pair_data = pid_results(ianimal).(pair_name);
                    if isstruct(pair_data) && isfield(pair_data, target) && length(pair_data.(target).epoch) >= i_ep
                        real_mat = pair_data.(target).epoch(i_ep).real;
                        if ~isempty(real_mat)
                            mouse_lag_curves = [mouse_lag_curves; mean(real_mat, 1, 'omitnan')];
                        end
                    end
                end
            end

            if ~isempty(mouse_lag_curves)
                mu = mean(mouse_lag_curves, 1, 'omitnan');
                se = std(mouse_lag_curves, 0, 1, 'omitnan') ./ ...
                     sqrt(sum(~isnan(mouse_lag_curves), 1));
                h = shadedErrorBar(lag_vector, mu, se, 'lineprops', {'-', 'Color', epoch_cols{i_ep}, 'LineWidth', 2});
                legend_handles(end+1) = h.mainLine;
                legend_labels{end+1}  = epoch_names{i_ep};
            end
        end

        xline(0, 'k--', 'LineWidth', 1);
        xlim([-cfg.pid_max_lag, cfg.pid_max_lag]);
        title(sprintf('%s \\leftrightarrow %s', regA, regB), 'FontSize', 14);
        if p == 1; ylabel('Mean MMI (Bits)'); end
        xlabel(sprintf('\\leftarrow %s Leads      %s Leads \\rightarrow', regB, regA));
        if p == 3 && ~isempty(legend_handles)
            legend(legend_handles, legend_labels, 'Location', 'best');
        end
        box on; grid on;
    end

    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1}; regB = cfg.area_pairs{p, 2};
        pair_name = sprintf('%s_to_%s', regA, regB);
        ax = nexttile(t_sum); hold(ax, 'on');

        asym_data_A = nan(3, n_animals); asym_data_B = nan(3, n_animals);

        for i_ep = 1:3
            for ianimal = 1:n_animals
                if isfield(pid_results(ianimal), pair_name)
                    pair_data = pid_results(ianimal).(pair_name);
                    if isstruct(pair_data) && isfield(pair_data, target) && length(pair_data.(target).epoch) >= i_ep
                        real_mat = pair_data.(target).epoch(i_ep).real;
                        if ~isempty(real_mat)
                            mean_lags = mean(real_mat, 1, 'omitnan');
                            asym_data_B(i_ep, ianimal) = mean(mean_lags(lag_vector < 0), 'omitnan');
                            asym_data_A(i_ep, ianimal) = mean(mean_lags(lag_vector > 0), 'omitnan');
                        end
                    end
                end
            end
        end

        n_A = sum(~isnan(asym_data_A), 2);
        n_B = sum(~isnan(asym_data_B), 2);
        mu_A = mean(asym_data_A, 2, 'omitnan'); se_A = std(asym_data_A, 0, 2, 'omitnan') ./ sqrt(max(n_A, 1));
        mu_B = mean(asym_data_B, 2, 'omitnan'); se_B = std(asym_data_B, 0, 2, 'omitnan') ./ sqrt(max(n_B, 1));

        b = bar(1:3, [mu_A, mu_B], 'grouped');
        b(1).FaceColor = [0.2 0.6 0.8]; b(2).FaceColor = [0.8 0.4 0.2];

        x_A = b(1).XEndPoints; x_B = b(2).XEndPoints;
        errorbar(x_A, mu_A, se_A, 'k', 'linestyle', 'none', 'LineWidth', 1.5);
        errorbar(x_B, mu_B, se_B, 'k', 'linestyle', 'none', 'LineWidth', 1.5);

        xticks(1:3); xticklabels({'Naive', 'Pre-LP', 'Post-LP'});
        if p == 1; ylabel('Net Capacity Bounds (Bits)'); end
        title('Information Dominance (MMI)', 'FontSize', 12);
        legend({sprintf('%s Leads', regA), sprintf('%s Leads', regB)}, 'Location', 'northwest');
        box on; grid on;
    end
    sgtitle(sprintf('Summary: Directionality of Redundancy Bounds (%s)', strrep(target, '_', ' ')), 'FontSize', 16);
end

%% --- Helper Functions ---

function out = discretize_zero_aware(x, n_bins)
    % 1. Isolate strict zeros, equipopulate non-zeros to preserve sparsity structure
    out = nan(size(x));
    is_valid = ~isnan(x);
    if sum(is_valid) == 0; return; end
    
    is_zero = abs(x) < 1e-8 & is_valid;
    out(is_zero) = 1;
    
    nz_idx = find(~is_zero & is_valid);
    if isempty(nz_idx); return; end
    
    nz_vals = x(nz_idx);
    nz_vals = nz_vals + randn(size(nz_vals)) * 1e-10; % Break exact ties
    
    if n_bins > 1 && length(nz_vals) >= n_bins
        edges = [-inf, prctile(nz_vals, linspace(100/(n_bins-1), 100 - 100/(n_bins-1), n_bins-2)), inf];
        out(nz_idx) = discretize(nz_vals, edges) + 1;
    else
        out(nz_idx) = 2;
    end
end

function I_corrected = compute_mi_mm(x, y, nx, ny)
    % Fast MI calculation with Panzeri-Treves / Miller-Madow asymptotic bias correction
    valid = x >= 1 & x <= nx & y >= 1 & y <= ny;
    x = x(valid); y = y(valid);
    N = length(x);
    if N == 0
        I_corrected = nan; return; 
    end
    
    joint_counts = accumarray([x(:), y(:)], 1, [nx, ny]);
    P_xy = joint_counts / N;
    P_x = sum(P_xy, 2);
    P_y = sum(P_xy, 1);
    
    R_x = sum(P_x > 0);
    R_y = sum(P_y > 0);
    R_xy = sum(P_xy(:) > 0);
    bias = (R_xy - R_x - R_y + 1) / (2 * N * log(2));
    
    I_raw = 0;
    [r, c] = find(P_xy > 0);
    for k = 1:length(r)
        px = P_x(r(k)); py = P_y(c(k)); pxy = P_xy(r(k), c(k));
        I_raw = I_raw + pxy * log2(pxy / (px * py));
    end
    
    I_corrected = max(0, I_raw - bias); % Bound at 0 to prevent negative info on pure noise
end

function pc1_proj = extract_pc1(neural_tensor)
    % Reduces Unit dimension to a macroscopic PC1 trace.
    % Input:  neural_tensor [Units x Bins x Trials]
    % Output: pc1_proj      [Bins x Trials]
    %
    % NaN handling: uses pca's pairwise covariance estimator so missing
    % samples are excluded per-pair instead of being mean-imputed (which
    % the previous version did and which biases variance toward zero
    % and can flip the sign of PC1). The PC1 score is then computed by
    % projecting the centred data onto the resulting coefficient vector.
    [n_u, n_b, n_t] = size(neural_tensor);
    if n_u == 1
        pc1_proj = reshape(neural_tensor, n_b, n_t); return;
    end

    X2D = reshape(neural_tensor, n_u, n_b * n_t)';   % [(Bins*Trials) x Units]

    try
        coeff = pca(X2D, 'NumComponents', 1, 'Rows', 'pairwise');
    catch
        coeff = [];
    end
    if isempty(coeff)
        pc1_proj = nan(n_b, n_t); return;
    end

    mu  = mean(X2D, 1, 'omitnan');
    Xc  = X2D - mu;
    score1 = nan(size(Xc, 1), 1);
    valid_rows = ~any(isnan(Xc), 2);
    score1(valid_rows) = Xc(valid_rows, :) * coeff;
    pc1_proj = reshape(score1, n_b, n_t);
end

function epochs = epochs_from_lp(lp, n_trials)
    % Naive / Pre-LP / Post-LP indices using the v2 "end-of-window LP"
    % convention. Partial epochs are kept (clipped to [1, n_trials])
    % rather than dropped; downstream `min_valid_trials` checks decide
    % whether a partial epoch is large enough to use.
    epochs = cell(1, 3);
    epochs{1} = 1:min(10, n_trials);
    if ~isnan(lp)
        ep2 = (lp-10):(lp-1); epochs{2} = ep2(ep2 >= 1);
        ep3 = (lp+1):(lp+10); epochs{3} = ep3(ep3 <= n_trials);
    else
        epochs{2} = []; epochs{3} = [];
    end
end

function meta = cache_meta_from_cfg(cfg, keys)
    % Snapshot the cfg fields that change the on-disk computation, so we
    % can detect stale caches when a parameter changes.
    meta = struct();
    for k = 1:numel(keys)
        f = keys{k};
        if isfield(cfg, f)
            meta.(f) = cfg.(f);
        end
    end
end

function [is_valid, reason] = check_cache(filename, cfg, keys)
    % Returns true iff the cache file exists, contains `cache_meta`, and
    % every monitored field matches the current cfg. Any missing or
    % differing field invalidates the cache.
    is_valid = false; reason = '';
    if ~exist(filename, 'file')
        reason = 'no cache file';
        return;
    end
    info = whos('-file', filename);
    has_meta = any(strcmp({info.name}, 'cache_meta'));
    if ~has_meta
        reason = 'cache predates parameter tracking';
        return;
    end
    S = load(filename, 'cache_meta');
    cm = S.cache_meta;
    expected = cache_meta_from_cfg(cfg, keys);
    f_exp = fieldnames(expected);
    for i = 1:numel(f_exp)
        f = f_exp{i};
        if ~isfield(cm, f) || ~isequaln(cm.(f), expected.(f))
            reason = sprintf('field "%s" changed', f);
            return;
        end
    end
    is_valid = true;
end