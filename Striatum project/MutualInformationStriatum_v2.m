%% 1. Configuration & Setup
clear; clc; close all;
cfg.data_file = 'preprocessed_data.mat';
cfg.save_file = 'shannon_mi_results.mat';
cfg.regions = {'DMS', 'DLS', 'ACC'};
cfg.colors = {[0 0.4470 0.7410], [0.4660 0.6740 0.1880], [0.8500 0.3250 0.0980]};
cfg.behav_targets = {'lick_rate', 'velocity'};

% --- Information Theory Parameters ---
cfg.mi_neural_bins   = 2;  % Number of bins (Bin 1 reserved for exact zeros)
cfg.mi_behav_bins    = 2;  % Number of bins (Bin 1 reserved for exact zeros)
cfg.mi_pool_win      = 5;  % Number of spatial bins to pool per window
cfg.mi_pool_shift    = 1;  % Shift step for the moving window
cfg.min_valid_trials = 8; % Strict minimum N to compute valid discrete probability distributions
cfg.max_bin          = 30; % Truncate spatial analysis to RZ + 5 bins
cfg.target_rz_bin    = 25; % Reward zone start bin
cfg.lp_z_threshold   = -2; % Z-score threshold for lick precision
cfg.lp_window        = 10; % Size of the sliding window
cfg.lp_min_pass      = 7;  % Number of trials within the window that must pass the threshold

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
    if isfield(preprocessed_data(ianimal), 'learning_point') && ~isempty(preprocessed_data(ianimal).learning_point)
        lp = preprocessed_data(ianimal).learning_point;
    else
        z_err = preprocessed_data(ianimal).zscored_lick_errors(1:n_trials);
        lp = NaN; 
        for t = 1 : (n_trials - cfg.lp_window + 1)
            window_idx = t : (t + cfg.lp_window - 1);
            passed_trials = sum(z_err(window_idx) <= cfg.lp_z_threshold);
            if passed_trials >= cfg.lp_min_pass
                lp = window_idx(end); 
                break; 
            end
        end
    end
    
    n_bins_actual = min(size(preprocessed_data(ianimal).spatial_binned_fr_all, 2), cfg.max_bin);
    
    clean_data(ianimal).neural = preprocessed_data(ianimal).spatial_binned_fr_all(:, 1:n_bins_actual, 1:n_trials);
    clean_data(ianimal).lick_rate = lick_rate(:, 1:n_bins_actual)'; 
    clean_data(ianimal).velocity = velocity(:, 1:n_bins_actual)';   
    clean_data(ianimal).lp = lp;
    clean_data(ianimal).n_trials = n_trials;
    clean_data(ianimal).n_bins = n_bins_actual;
    clean_data(ianimal).is_dms = preprocessed_data(ianimal).is_dms;
    clean_data(ianimal).is_dls = preprocessed_data(ianimal).is_dls;
    clean_data(ianimal).is_acc = preprocessed_data(ianimal).is_acc;
end

%% 3. Core Functions for Information Theory
% Fast Shannon Mutual Information I(X;Y) with Miller-Madow Bias Correction
calc_mi = @(x, y, nx, ny) compute_mi_mm(x, y, nx, ny);

%% 4. Calculate Pooled Shannon Mutual Information
if exist(cfg.save_file, 'file')
    fprintf('Loading existing MI results...\n');
    load(cfg.save_file, 'mi_results');
else
    fprintf('--- Computing Shannon Mutual Information (Pooled) ---\n');
    mi_results = struct();
    
    for ianimal = 1:n_animals
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;
        
        win_starts = 1 : cfg.mi_pool_shift : (n_bins - cfg.mi_pool_win + 1);
        n_windows = length(win_starts);
        win_centers = win_starts + floor(cfg.mi_pool_win / 2);
        mi_results(ianimal).win_centers = win_centers;
        
        epochs = cell(1, 3);
        epochs{1} = 1:min(10, n_trials); 
        if ~isnan(lp)
            ep2 = (lp-10):(lp-1); epochs{2} = ep2(ep2 >= 1); 
            ep3 = (lp+1):(lp+10); epochs{3} = ep3(ep3 <= n_trials); 
        else
            epochs{2} = []; epochs{3} = [];
        end
        
        for i_targ = 1:numel(cfg.behav_targets)
            target_name = cfg.behav_targets{i_targ};
            Y_all = clean_data(ianimal).(target_name);
            
            for i_reg = 1:numel(cfg.regions)
                region = cfg.regions{i_reg};
                idx_reg = clean_data(ianimal).(['is_' lower(region)]);
                if sum(idx_reg) == 0; continue; end
                
                reg_neural = clean_data(ianimal).neural(idx_reg, :, :);
                n_units = size(reg_neural, 1);
                
                for i_ep = 1:3
                    tr_idx = epochs{i_ep};
                    if length(tr_idx) < cfg.min_valid_trials 
                        mi_results(ianimal).(region).(target_name).epoch(i_ep).real = nan(n_units, n_windows);
                        continue;
                    end
                    
                    Y_ep = Y_all(:, tr_idx);
                    X_ep = reg_neural(:, :, tr_idx);
                    mi_real_mat = nan(n_units, n_windows);
                    
                    for w = 1:n_windows
                        b_idx = win_starts(w) : (win_starts(w) + cfg.mi_pool_win - 1);
                        Y_win = Y_ep(b_idx, :); Y_flat = Y_win(:);
                        
                        Y_disc_full = discretize_zero_aware(Y_flat, cfg.mi_behav_bins);
                        if sum(~isnan(Y_disc_full)) < cfg.min_valid_trials; continue; end 
                        
                        for u = 1:n_units
                            X_win = squeeze(X_ep(u, b_idx, :)); X_flat = X_win(:);
                            X_disc_full = discretize_zero_aware(X_flat, cfg.mi_neural_bins);
                            
                            valid_mask = ~isnan(X_disc_full) & ~isnan(Y_disc_full);
                            if sum(valid_mask) < cfg.min_valid_trials; continue; end
                            
                            X_val = X_disc_full(valid_mask);
                            Y_val = Y_disc_full(valid_mask);
                            
                            mi_real_mat(u, w) = calc_mi(X_val, Y_val, cfg.mi_neural_bins, cfg.mi_behav_bins);
                        end
                    end
                    mi_results(ianimal).(region).(target_name).epoch(i_ep).real = mi_real_mat;
                end
            end
        end
    end
    save(cfg.save_file, 'mi_results', '-v7.3');
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
        
        for i_reg = 1:numel(cfg.regions)
            region = cfg.regions{i_reg};
            mi_curves_real = [];
            common_x = [];
            
            for ianimal = 1:n_animals
                if isfield(mi_results(ianimal), region) && isfield(mi_results(ianimal).(region), target)
                    if length(mi_results(ianimal).(region).(target).epoch) >= i_ep
                        real_mat = mi_results(ianimal).(region).(target).epoch(i_ep).real;
                        common_x = mi_results(ianimal).win_centers;
                        if ~isempty(real_mat)
                            mouse_mean_real = mean(real_mat, 1, 'omitnan');
                            mi_curves_real = [mi_curves_real; mouse_mean_real];
                        end
                    end
                end
            end
            
            if ~isempty(mi_curves_real)
                mu_real = mean(mi_curves_real, 1, 'omitnan');
                se_real = std(mi_curves_real, 0, 1, 'omitnan') ./ sqrt(size(mi_curves_real, 1));
                h = shadedErrorBar(common_x, mu_real, se_real, 'lineprops', {'Color', cfg.colors{i_reg}, 'LineWidth', 2.5});
                legend_handles(end+1) = h.mainLine;
            end
        end
        
        xline(cfg.target_rz_bin, 'r-', 'Reward Zone', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
        xlim([min(common_x) max(common_x)]);
        
        title(sprintf('%s', epoch_names{i_ep}), 'FontSize', 14);
        if i_ep == 1; ylabel('Corrected MI (Bits)', 'FontWeight', 'bold', 'FontSize', 12); end
        if i_ep == 2; xlabel('Spatial Bin (Window Center)', 'FontSize', 12); end
        if i_ep == 3; legend(legend_handles, cfg.regions, 'Location', 'northwest'); end
        box on; grid on;
    end
    linkaxes
    sgtitle(sprintf('Task Encoding: Single-Unit Shannon Information (%s)', strrep(target, '_', ' ')), 'FontSize', 16);
end

%% 6. Calculate Cross-Spatial (Lagged) Mutual Information (Neural -> Behavior)
fprintf('--- Computing Cross-Spatial Mutual Information ---\n');
cfg.mi_max_lag = 15;
cross_mi_file = 'cross_spatial_mi_results.mat';
if exist(cross_mi_file, 'file')
    fprintf('Loading existing Cross-Spatial MI results...\n');
    load(cross_mi_file, 'cross_mi_results');
else
    cross_mi_results = struct();
    for ianimal = 1:n_animals
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;
        
        win_starts = 1 : cfg.mi_pool_shift : (n_bins - cfg.mi_pool_win + 1);
        n_windows = length(win_starts);
        cross_mi_results(ianimal).win_centers = win_starts + floor(cfg.mi_pool_win / 2);
        
        epochs = cell(1, 3);
        epochs{1} = 1:min(10, n_trials); 
        if ~isnan(lp)
            ep2 = (lp-10):(lp-1); epochs{2} = ep2(ep2 >= 1); 
            ep3 = (lp+1):(lp+10); epochs{3} = ep3(ep3 <= n_trials); 
        else
            epochs{2} = []; epochs{3} = [];
        end
        
        for i_targ = 1:numel(cfg.behav_targets)
            target_name = cfg.behav_targets{i_targ};
            Y_all = clean_data(ianimal).(target_name);
            
            for i_reg = 1:numel(cfg.regions)
                region = cfg.regions{i_reg};
                idx_reg = clean_data(ianimal).(['is_' lower(region)]);
                if sum(idx_reg) == 0; continue; end
                
                reg_neural = clean_data(ianimal).neural(idx_reg, :, :);
                n_units = size(reg_neural, 1);
                
                for i_ep = 1:3
                    tr_idx = epochs{i_ep};
                    if length(tr_idx) < cfg.min_valid_trials; continue; end
                    
                    Y_ep = Y_all(:, tr_idx);
                    X_ep = reg_neural(:, :, tr_idx);
                    mi_real_mat = nan(n_units, n_windows, n_windows);
                    
                    for w_targ = 1:n_windows
                        b_idx_targ = win_starts(w_targ) : (win_starts(w_targ) + cfg.mi_pool_win - 1);
                        Y_win = Y_ep(b_idx_targ, :); Y_flat = Y_win(:);
                        
                        Y_disc_full = discretize_zero_aware(Y_flat, cfg.mi_behav_bins);
                        if sum(~isnan(Y_disc_full)) < cfg.min_valid_trials; continue; end 
                        
                        w_source_start = max(1, w_targ - cfg.mi_max_lag);
                        for w_source = w_source_start:w_targ
                            b_idx_source = win_starts(w_source) : (win_starts(w_source) + cfg.mi_pool_win - 1);
                            
                            for u = 1:n_units
                                X_win = squeeze(X_ep(u, b_idx_source, :)); X_flat = X_win(:);
                                X_disc_full = discretize_zero_aware(X_flat, cfg.mi_neural_bins);
                                
                                valid_mask = ~isnan(X_disc_full) & ~isnan(Y_disc_full);
                                if sum(valid_mask) < cfg.min_valid_trials; continue; end
                                
                                X_val = X_disc_full(valid_mask);
                                Y_val = Y_disc_full(valid_mask);
                                
                                mi_real_mat(u, w_targ, w_source) = calc_mi(X_val, Y_val, cfg.mi_neural_bins, cfg.mi_behav_bins);
                            end
                        end
                    end
                    cross_mi_results(ianimal).(region).(target_name).epoch(i_ep).real = mi_real_mat;
                end
            end
        end
    end
    save(cross_mi_file, 'cross_mi_results', '-v7.3');
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
                    se = std(lag_curves, 0, 1, 'omitnan') ./ sqrt(size(lag_curves, 1));
                    h = shadedErrorBar(0:cfg.mi_max_lag, mu, se, 'lineprops', {'Color', zone_colors{z}, 'LineWidth', 2});
                    legend_handles(end+1) = h.mainLine;
                end
            end
            
            yline(0, 'k--');
            xlim([0 cfg.mi_max_lag]); 
            title(epoch_names{i_ep}, 'FontSize', 13);
            if i_ep == 1; ylabel('Corrected MI (Bits)'); end
            xlabel('Spatial Lag (Target Bin - Source Bin)');
            if i_ep == 3; legend(legend_handles, zone_defs(:,1), 'Location', 'northeast'); end
            box on; grid on;
        end
        sgtitle(sprintf('Predictive Horizon by Spatial Zone: %s encoding %s', region, strrep(target, '_', ' ')), 'FontSize', 16);
    end
end

%% 8. Calculate Cross-Area Mutual Information (Information Flow using PC1)
fprintf('--- Computing Cross-Area Mutual Information ---\n');
cross_area_file = 'cross_area_mi_results.mat';
cfg.area_pairs = {'ACC', 'DMS'; 'ACC', 'DLS'; 'DMS', 'DLS'};
if exist(cross_area_file, 'file')
    fprintf('Loading existing Cross-Area MI results...\n');
    load(cross_area_file, 'area_mi_results');
else
    area_mi_results = struct();
    for ianimal = 1:n_animals
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;
        
        win_starts = 1 : cfg.mi_pool_shift : (n_bins - cfg.mi_pool_win + 1);
        n_windows = length(win_starts);
        area_mi_results(ianimal).win_centers = win_starts + floor(cfg.mi_pool_win / 2);
        
        epochs = cell(1, 3);
        epochs{1} = 1:min(10, n_trials); 
        if ~isnan(lp)
            ep2 = (lp-10):(lp-1); epochs{2} = ep2(ep2 >= 1); 
            ep3 = (lp+1):(lp+10); epochs{3} = ep3(ep3 <= n_trials); 
        else
            epochs{2} = []; epochs{3} = [];
        end
        
        for i_ep = 1:3
            tr_idx = epochs{i_ep};
            if length(tr_idx) < cfg.min_valid_trials; continue; end
            
            for p = 1:size(cfg.area_pairs, 1)
                regA = cfg.area_pairs{p, 1}; regB = cfg.area_pairs{p, 2};
                pair_name = sprintf('%s_%s', regA, regB);
                
                idxA = clean_data(ianimal).(['is_' lower(regA)]);
                idxB = clean_data(ianimal).(['is_' lower(regB)]);
                if sum(idxA) == 0 || sum(idxB) == 0; continue; end
                
                % Extract PC1 Projection as macroscopic state
                popA = extract_pc1(clean_data(ianimal).neural(idxA, :, tr_idx));
                popB = extract_pc1(clean_data(ianimal).neural(idxB, :, tr_idx));
                
                mi_mat = nan(n_windows, n_windows);
                
                for wA = 1:n_windows
                    b_idxA = win_starts(wA) : (win_starts(wA) + cfg.mi_pool_win - 1);
                    winA = popA(b_idxA, :); flatA = winA(:);
                    discA_full = discretize_zero_aware(flatA, cfg.mi_neural_bins);
                    
                    for wB = 1:n_windows
                        b_idxB = win_starts(wB) : (win_starts(wB) + cfg.mi_pool_win - 1);
                        winB = popB(b_idxB, :); flatB = winB(:);
                        discB_full = discretize_zero_aware(flatB, cfg.mi_neural_bins);
                        
                        valid_mask = ~isnan(discA_full) & ~isnan(discB_full);
                        if sum(valid_mask) < cfg.min_valid_trials; continue; end
                        
                        discA = discA_full(valid_mask); discB = discB_full(valid_mask);
                        mi_mat(wA, wB) = calc_mi(discA, discB, cfg.mi_neural_bins, cfg.mi_neural_bins);
                    end
                end
                area_mi_results(ianimal).(pair_name).epoch(i_ep).real = mi_mat;
            end
        end
    end
    save(cross_area_file, 'area_mi_results', '-v7.3');
    fprintf('Cross-Area MI computation complete.\n');
end

%% 9. Plotting: Cross-Area Information Flow
fprintf('--- Plotting Cross-Area Information Flow ---\n');
cfg.area_max_lag = 10; 
lags_to_plot = -cfg.area_max_lag : cfg.area_max_lag;
figure('Position', [100 600 1500 500], 'Color', 'w', 'Name', 'Cross-Area Mutual Information');
t_flow = tiledlayout(1, 3, 'Padding', 'compact');
pair_colors = lines(3);

for i_ep = 1:3
    ax = nexttile(t_flow); hold(ax, 'on');
    legend_handles = [];
    pair_labels = {};
    
    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1}; regB = cfg.area_pairs{p, 2};
        pair_name = sprintf('%s_%s', regA, regB);
        pair_labels{end+1} = sprintf('%s \\rightarrow %s', regA, regB);
        
        lag_curves = [];
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
                end
            end
        end
        
        if ~isempty(lag_curves)
            mu = mean(lag_curves, 1, 'omitnan');
            se = std(lag_curves, 0, 1, 'omitnan') ./ sqrt(size(lag_curves, 1));
            h = shadedErrorBar(lags_to_plot, mu, se, 'lineprops', {'Color', pair_colors(p,:), 'LineWidth', 2});
            legend_handles(end+1) = h.mainLine;
        end
    end
    
    xline(0, 'k--', 'LineWidth', 1.5);
    xlim([-cfg.area_max_lag, cfg.area_max_lag]);
    
    title(epoch_names{i_ep}, 'FontSize', 14);
    if i_ep == 1; ylabel('Corrected PC1 MI (Bits)', 'FontWeight', 'bold'); end
    xlabel('Spatial Lag (Bins)');
    if i_ep == 3
        text(0.1, -0.15, sprintf('\\leftarrow %s leads', 'Area B'), 'Units', 'normalized', 'HorizontalAlignment', 'left');
        text(0.9, -0.15, sprintf('%s leads \\rightarrow', 'Area A'), 'Units', 'normalized', 'HorizontalAlignment', 'right');
        legend(legend_handles, pair_labels, 'Location', 'northwest'); 
    end
    box on; grid on;
end
sgtitle('Cross-Area Information Flow Across Learning', 'FontSize', 16);

%% 10. Calculate Minimum Mutual Information (MMI Bound for PID)
fprintf('--- Computing MMI Shared Information Bound ---\n');
cfg.pid_max_lag = 10; 
pid_file = 'pid_shared_info_results.mat';
if exist(pid_file, 'file')
    fprintf('Loading existing MMI results...\n');
    load(pid_file, 'pid_results');
else
    pid_results = struct();
    for ianimal = 1:n_animals
        fprintf('Processing Animal %d/%d for MMI...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;
        
        win_starts = 1 : cfg.mi_pool_shift : (n_bins - cfg.mi_pool_win + 1);
        n_windows = length(win_starts);
        pid_results(ianimal).win_centers = win_starts + floor(cfg.mi_pool_win / 2);
        
        epochs = cell(1, 3);
        epochs{1} = 1:min(10, n_trials); 
        if ~isnan(lp)
            ep2 = (lp-10):(lp-1); epochs{2} = ep2(ep2 >= 1); 
            ep3 = (lp+1):(lp+10); epochs{3} = ep3(ep3 <= n_trials); 
        else
            epochs{2} = []; epochs{3} = [];
        end
        
        for i_targ = 1:numel(cfg.behav_targets)
            target_name = cfg.behav_targets{i_targ};
            Y_all = clean_data(ianimal).(target_name);
            
            for i_ep = 1:3
                tr_idx = epochs{i_ep};
                if length(tr_idx) < cfg.min_valid_trials; continue; end
                
                for p = 1:size(cfg.area_pairs, 1)
                    regA = cfg.area_pairs{p, 1}; regB = cfg.area_pairs{p, 2}; 
                    pair_name = sprintf('%s_to_%s', regA, regB);
                    
                    idxA = clean_data(ianimal).(['is_' lower(regA)]);
                    idxB = clean_data(ianimal).(['is_' lower(regB)]);
                    if sum(idxA) == 0 || sum(idxB) == 0; continue; end
                    
                    popA = extract_pc1(clean_data(ianimal).neural(idxA, :, tr_idx));
                    popB = extract_pc1(clean_data(ianimal).neural(idxB, :, tr_idx));
                    behav = Y_all(:, tr_idx);
                    
                    n_lags = 2 * cfg.pid_max_lag + 1;
                    si_mat = nan(n_windows, n_lags);
                    
                    for w_targ = 1:n_windows
                        b_idx = win_starts(w_targ) : (win_starts(w_targ) + cfg.mi_pool_win - 1);
                        F_flat = behav(b_idx, :); F_flat = F_flat(:);
                        F_disc_full = discretize_zero_aware(F_flat, cfg.mi_behav_bins);
                        
                        B_flat = popB(b_idx, :); B_flat = B_flat(:);
                        B_disc_full = discretize_zero_aware(B_flat, cfg.mi_neural_bins);
                        
                        for lag_val = -cfg.pid_max_lag : cfg.pid_max_lag
                            lag_idx = lag_val + cfg.pid_max_lag + 1; 
                            w_source = w_targ - lag_val;
                            if w_source < 1 || w_source > n_windows; continue; end 
                            
                            b_idx_lag = win_starts(w_source) : (win_starts(w_source) + cfg.mi_pool_win - 1);
                            A_flat = popA(b_idx_lag, :); A_flat = A_flat(:);
                            A_disc_full = discretize_zero_aware(A_flat, cfg.mi_neural_bins);
                            
                            valid = ~isnan(A_disc_full) & ~isnan(B_disc_full) & ~isnan(F_disc_full);
                            if sum(valid) < cfg.min_valid_trials; continue; end
                            
                            A_val = A_disc_full(valid); B_val = B_disc_full(valid); F_val = F_disc_full(valid);
                            
                            I_A_F = calc_mi(A_val, F_val, cfg.mi_neural_bins, cfg.mi_behav_bins);
                            I_B_F = calc_mi(B_val, F_val, cfg.mi_neural_bins, cfg.mi_behav_bins);
                            si_mat(w_targ, lag_idx) = min(I_A_F, I_B_F); % MMI Bound
                        end
                    end
                    pid_results(ianimal).(pair_name).(target_name).epoch(i_ep).real = si_mat;
                end
            end
        end
    end
    save(pid_file, 'pid_results', '-v7.3');
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
                mu_si(isnan(mu_si)) = 0; 
                smoothed_si = imgaussfilt(mu_si, 0.8, 'FilterDomain', 'spatial', 'padding', 'symmetric');
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
                se = std(mouse_lag_curves, 0, 1, 'omitnan') ./ sqrt(size(mouse_lag_curves, 1));
                h = shadedErrorBar(lag_vector, mu, se, 'lineprops', {'-', 'Color', epoch_cols{i_ep}, 'LineWidth', 2});
                legend_handles(end+1) = h.mainLine;
            end
        end
        
        xline(0, 'k--', 'LineWidth', 1);
        xlim([-cfg.pid_max_lag, cfg.pid_max_lag]);
        title(sprintf('%s \\leftrightarrow %s', regA, regB), 'FontSize', 14);
        if p == 1; ylabel('Mean MMI (Bits)'); end
        xlabel(sprintf('\\leftarrow %s Leads      %s Leads \\rightarrow', regB, regA));
        if p == 3; legend(legend_handles, epoch_names, 'Location', 'best'); end
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
        
        mu_A = mean(asym_data_A, 2, 'omitnan'); se_A = std(asym_data_A, 0, 2, 'omitnan') ./ sqrt(n_animals);
        mu_B = mean(asym_data_B, 2, 'omitnan'); se_B = std(asym_data_B, 0, 2, 'omitnan') ./ sqrt(n_animals);
        
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
    % Reduces Unit dimension to robust macroscopic representation using PCA
    % Input: [Units x Bins x Trials]
    [n_u, n_b, n_t] = size(neural_tensor);
    if n_u == 1
        pc1_proj = squeeze(neural_tensor); return;
    end
    
    X2D = reshape(neural_tensor, n_u, n_b * n_t)'; 
    X2D_c = X2D - mean(X2D, 1, 'omitnan');
    X2D_c(isnan(X2D_c)) = 0; % Fill NaNs at mean for decomposition
    
    [~, score, ~] = pca(X2D_c, 'NumComponents', 1);
    if isempty(score)
        pc1_proj = nan(n_b, n_t);
    else
        pc1_proj = reshape(score(:,1), n_b, n_t);
    end
end