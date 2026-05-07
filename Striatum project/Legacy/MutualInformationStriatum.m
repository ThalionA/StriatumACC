%% 1. Configuration & Setup
% clear; clc; close all;

cfg.data_file = 'preprocessed_data.mat';
cfg.save_file = 'shannon_mi_results.mat';
cfg.regions = {'DMS', 'DLS', 'ACC'};
cfg.colors = {[0 0.4470 0.7410], [0.4660 0.6740 0.1880], [0.8500 0.3250 0.0980]};
cfg.behav_targets = {'lick_rate', 'velocity'};

% --- Information Theory Parameters ---
cfg.mi_neural_bins = 3;  % Number of equipopulated bins for Neural Activity (e.g., 3 = tertiles)
cfg.mi_behav_bins  = 3;  % Number of equipopulated bins for Behavior
cfg.mi_pool_win    = 5;  % Number of spatial bins to pool per window (e.g., 5 bins = 20cm)
cfg.mi_pool_shift  = 1;  % Shift step for the moving window (1 for maximum spatial resolution)
cfg.n_shuffles     = 20; % Number of shuffles to compute the null (chance) MI threshold
cfg.max_bin        = 30; % Truncate spatial analysis to RZ + 5 bins
cfg.target_rz_bin  = 25; % Reward zone start bin
cfg.lp_z_threshold = -2; % Z-score threshold for lick precision
cfg.lp_window      = 10; % Size of the sliding window
cfg.lp_min_pass    = 7;  % Number of trials within the window that must pass the threshold

%% 2. Data Loading & Preprocessing
fprintf('--- Loading Data ---\n');
% load(cfg.data_file, 'preprocessed_data');
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
    % 1. Check if a pre-calculated LP exists to ensure consistency across scripts
    if isfield(preprocessed_data(ianimal), 'learning_point') && ~isempty(preprocessed_data(ianimal).learning_point)
        lp = preprocessed_data(ianimal).learning_point;
    else
        % 2. Calculate using sliding window (No convolution)
        z_err = preprocessed_data(ianimal).zscored_lick_errors(1:n_trials);
        lp = NaN; 
        
        % Slide a window of length cfg.lp_window across the trials
        for t = 1 : (n_trials - cfg.lp_window + 1)
            window_idx = t : (t + cfg.lp_window - 1);
            passed_trials = sum(z_err(window_idx) <= cfg.lp_z_threshold);
            
            % If the window meets the minimum pass criteria, the LP is the END of this window
            if passed_trials >= cfg.lp_min_pass
                lp = window_idx(end); 
                break; % Stop searching once the first successful window is found
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
% Robust Equipopulated Discretization (Handles NaNs safely)
discretize_equipop = @(x, n) discretize(x + randn(size(x))*1e-9, ...
    [-inf, prctile(x(~isnan(x)) + randn(sum(~isnan(x)),1)*1e-9, linspace(100/n, 100 - 100/n, n-1)), inf]);

% Fast Shannon Mutual Information I(X;Y)
calc_mi = @(x, y, nx, ny) compute_mi_core(x, y, nx, ny);

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
        
        % Determine moving window indices
        win_starts = 1 : cfg.mi_pool_shift : (n_bins - cfg.mi_pool_win + 1);
        n_windows = length(win_starts);
        win_centers = win_starts + floor(cfg.mi_pool_win / 2);
        mi_results(ianimal).win_centers = win_centers;
        
        % Define Trial Epochs: Naive, Pre-LP, Post-LP
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
            Y_all = clean_data(ianimal).(target_name); % [Bins x Trials]
            
            for i_reg = 1:numel(cfg.regions)
                region = cfg.regions{i_reg};
                idx_reg = clean_data(ianimal).(['is_' lower(region)]);
                if sum(idx_reg) == 0; continue; end
                
                reg_neural = clean_data(ianimal).neural(idx_reg, :, :); % [Units x Bins x Trials]
                n_units = size(reg_neural, 1);
                
                for i_ep = 1:3
                    tr_idx = epochs{i_ep};
                    if length(tr_idx) < 3 % Need minimum trials to build distributions
                        mi_results(ianimal).(region).(target_name).epoch(i_ep).real = nan(n_units, n_windows);
                        mi_results(ianimal).(region).(target_name).epoch(i_ep).shuff = nan(n_units, n_windows);
                        continue;
                    end
                    
                    Y_ep = Y_all(:, tr_idx);
                    X_ep = reg_neural(:, :, tr_idx);
                    
                    mi_real_mat = nan(n_units, n_windows);
                    mi_shuff_mat = nan(n_units, n_windows);
                    
                    for w = 1:n_windows
                        b_idx = win_starts(w) : (win_starts(w) + cfg.mi_pool_win - 1);
                        
                        % Extract, flatten, and discretize behavior for this window
                        Y_win = Y_ep(b_idx, :);
                        Y_flat = Y_win(:);
                        
                        % Safety check: Need variance and enough data points
                        if sum(~isnan(Y_flat)) < 5 || std(Y_flat, 'omitnan') == 0; continue; end 
                        Y_disc_full = discretize_equipop(Y_flat, cfg.mi_behav_bins);
                        
                        % Compute for each unit
                        for u = 1:n_units
                            X_win = squeeze(X_ep(u, b_idx, :));
                            X_flat = X_win(:);
                            
                            % Safety check for silent or sparse neurons
                            if sum(~isnan(X_flat)) < 5 || std(X_flat, 'omitnan') == 0; continue; end
                            X_disc_full = discretize_equipop(X_flat, cfg.mi_neural_bins);
                            
                            % --- CRITICAL FIX: Extract indices valid in BOTH arrays ---
                            valid_mask = ~isnan(X_disc_full) & ~isnan(Y_disc_full);
                            if sum(valid_mask) < 5; continue; end
                            
                            X_val = X_disc_full(valid_mask);
                            Y_val = Y_disc_full(valid_mask);
                            
                            % Real MI
                            mi_real_mat(u, w) = calc_mi(X_val, Y_val, cfg.mi_neural_bins, cfg.mi_behav_bins);
                            
                            % Shuffled Null (Permute behavior to break temporal/spatial mapping)
                            shuff_vals = zeros(1, cfg.n_shuffles);
                            for s = 1:cfg.n_shuffles
                                shuff_vals(s) = calc_mi(X_val, Y_val(randperm(length(Y_val))), cfg.mi_neural_bins, cfg.mi_behav_bins);
                            end
                            mi_shuff_mat(u, w) = mean(shuff_vals);
                        end
                    end
                    
                    mi_results(ianimal).(region).(target_name).epoch(i_ep).real = mi_real_mat;
                    mi_results(ianimal).(region).(target_name).epoch(i_ep).shuff = mi_shuff_mat;
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
            
            % Collect unit-averaged MI curves across all mice
            mi_curves_real = [];
            mi_curves_shuff = [];
            common_x = [];
            
            for ianimal = 1:n_animals
                if isfield(mi_results(ianimal), region) && isfield(mi_results(ianimal).(region), target)
                    if length(mi_results(ianimal).(region).(target).epoch) >= i_ep
                        real_mat = mi_results(ianimal).(region).(target).epoch(i_ep).real;
                        shuff_mat = mi_results(ianimal).(region).(target).epoch(i_ep).shuff;
                        common_x = mi_results(ianimal).win_centers;
                        
                        if ~isempty(real_mat)
                            % Average across units for this mouse
                            mouse_mean_real = mean(real_mat, 1, 'omitnan');
                            mouse_mean_shuff = mean(shuff_mat, 1, 'omitnan');
                            
                            mi_curves_real = [mi_curves_real; mouse_mean_real];
                            mi_curves_shuff = [mi_curves_shuff; mouse_mean_shuff];
                        end
                    end
                end
            end
            
            if ~isempty(mi_curves_real)
                % Average across mice
                mu_real = mean(mi_curves_real, 1, 'omitnan');
                se_real = std(mi_curves_real, 0, 1, 'omitnan') ./ sqrt(size(mi_curves_real, 1));
                
                mu_shuff = mean(mi_curves_shuff, 1, 'omitnan');
                
                % Plot Real MI
                h = shadedErrorBar(common_x, mu_real, se_real, 'lineprops', {'Color', cfg.colors{i_reg}, 'LineWidth', 2.5});
                legend_handles(end+1) = h.mainLine;
                
                % Plot Null (Chance) MI
                plot(common_x, mu_shuff, '--', 'Color', [cfg.colors{i_reg} 0.5], 'LineWidth', 1.5);
            end
        end
        
        % Formatting
        xline(cfg.target_rz_bin, 'r-', 'Reward Zone', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
        
        xlim([min(common_x) max(common_x)]);
        % ylim([0 0.15]); % Fixed Y-axis (Bits of Info) for comparison across epochs
        
        title(sprintf('%s', epoch_names{i_ep}), 'FontSize', 14);
        if i_ep == 1; ylabel('Mutual Information (Bits)', 'FontWeight', 'bold', 'FontSize', 12); end
        if i_ep == 2; xlabel('Spatial Bin (Window Center)', 'FontSize', 12); end
        
        if i_ep == 3; legend(legend_handles, cfg.regions, 'Location', 'northwest'); end
        box on; grid on;
    end
    linkaxes
    sgtitle(sprintf('Task Encoding: Single-Unit Shannon Information (%s)', strrep(target, '_', ' ')), 'FontSize', 16);
end

%% 6. Calculate Cross-Spatial (Lagged) Mutual Information (Neural -> Behavior)
fprintf('--- Computing Cross-Spatial Mutual Information ---\n');

cfg.mi_max_lag = 15; % Maximum spatial lag to compute (in bins)
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
        win_centers = win_starts + floor(cfg.mi_pool_win / 2);
        cross_mi_results(ianimal).win_centers = win_centers;
        
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
                    if length(tr_idx) < 3; continue; end
                    
                    Y_ep = Y_all(:, tr_idx);
                    X_ep = reg_neural(:, :, tr_idx);
                    
                    % 3D Matrix: [Units x TargetWindow x SourceWindow]
                    mi_real_mat = nan(n_units, n_windows, n_windows);
                    mi_shuff_mat = nan(n_units, n_windows, n_windows);
                    
                    for w_targ = 1:n_windows
                        b_idx_targ = win_starts(w_targ) : (win_starts(w_targ) + cfg.mi_pool_win - 1);
                        Y_win = Y_ep(b_idx_targ, :); Y_flat = Y_win(:);
                        
                        % Ensure sufficient non-NaN variability
                        if sum(~isnan(Y_flat)) < 5 || std(Y_flat, 'omitnan') == 0; continue; end 
                        Y_disc_full = discretize_equipop(Y_flat, cfg.mi_behav_bins);
                        
                        % Only look at current and previous spatial bins (Predictive)
                        w_source_start = max(1, w_targ - cfg.mi_max_lag);
                        
                        for w_source = w_source_start:w_targ
                            b_idx_source = win_starts(w_source) : (win_starts(w_source) + cfg.mi_pool_win - 1);
                            
                            for u = 1:n_units
                                X_win = squeeze(X_ep(u, b_idx_source, :));
                                X_flat = X_win(:);
                                
                                if sum(~isnan(X_flat)) < 5 || std(X_flat, 'omitnan') == 0; continue; end
                                X_disc_full = discretize_equipop(X_flat, cfg.mi_neural_bins);
                                
                                % --- CRITICAL FIX: Extract indices valid in BOTH arrays ---
                                valid_mask = ~isnan(X_disc_full) & ~isnan(Y_disc_full);
                                if sum(valid_mask) < 5; continue; end
                                
                                X_val = X_disc_full(valid_mask);
                                Y_val = Y_disc_full(valid_mask);
                                
                                mi_real_mat(u, w_targ, w_source) = calc_mi(X_val, Y_val, cfg.mi_neural_bins, cfg.mi_behav_bins);
                                
                                shuff_vals = zeros(1, 5); % Reduced shuffles for computational speed on cross-spatial
                                for s = 1:5
                                    shuff_vals(s) = calc_mi(X_val, Y_val(randperm(length(Y_val))), cfg.mi_neural_bins, cfg.mi_behav_bins);
                                end
                                mi_shuff_mat(u, w_targ, w_source) = mean(shuff_vals);
                            end
                        end
                    end
                    cross_mi_results(ianimal).(region).(target_name).epoch(i_ep).real = mi_real_mat;
                    cross_mi_results(ianimal).(region).(target_name).epoch(i_ep).shuff = mi_shuff_mat;
                end
            end
        end
    end
    save(cross_mi_file, 'cross_mi_results', '-v7.3');
    fprintf('Cross-Spatial MI computation complete.\n');
end

%% 7. Plotting: Spatial Lag Profiles by Zone (Early vs Visual vs Reward)
fprintf('--- Plotting MI Lags by Target Zone ---\n');

% Define Zones based on Window Centers
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
                lag_curves = []; % [Mice x Lags]
                
                for ianimal = 1:n_animals
                    if isfield(cross_mi_results(ianimal), region) && isfield(cross_mi_results(ianimal).(region), target)
                        if length(cross_mi_results(ianimal).(region).(target).epoch) >= i_ep
                            centers = cross_mi_results(ianimal).win_centers;
                            real_mat = cross_mi_results(ianimal).(region).(target).epoch(i_ep).real;
                            shuff_mat = cross_mi_results(ianimal).(region).(target).epoch(i_ep).shuff;
                            
                            if isempty(real_mat); continue; end
                            
                            % Find target windows that fall within this zone
                            target_idxs = find(zone_defs{z, 2}(centers));
                            if isempty(target_idxs); continue; end
                            
                            mouse_lag_profile = nan(length(target_idxs), cfg.mi_max_lag + 1);
                            
                            for t_idx = 1:length(target_idxs)
                                wy = target_idxs(t_idx);
                                for lag = 0:cfg.mi_max_lag
                                    wx = wy - lag;
                                    if wx >= 1
                                        % Mean Delta MI across all units for this specific target-source pair
                                        r_delta = mean(real_mat(:, wy, wx) - shuff_mat(:, wy, wx), 1, 'omitnan');
                                        mouse_lag_profile(t_idx, lag + 1) = r_delta;
                                    end
                                end
                            end
                            % Average across target bins within the zone for this mouse
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
            ylim([-0.02 0.1]); % Set strict bounds to observe shifting predictive horizons
            
            title(epoch_names{i_ep}, 'FontSize', 13);
            if i_ep == 1; ylabel('Mean \Delta MI (Bits)'); end
            xlabel('Spatial Lag (Target Bin - Source Bin)');
            if i_ep == 3; legend(legend_handles, zone_defs(:,1), 'Location', 'northeast'); end
            box on; grid on;
        end
        sgtitle(sprintf('Predictive Horizon by Spatial Zone: %s encoding %s', region, strrep(target, '_', ' ')), 'FontSize', 16);
    end
end

%% 8. Calculate Cross-Area Mutual Information (Information Flow)
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
            if length(tr_idx) < 3; continue; end
            
            for p = 1:size(cfg.area_pairs, 1)
                regA = cfg.area_pairs{p, 1};
                regB = cfg.area_pairs{p, 2};
                pair_name = sprintf('%s_%s', regA, regB);
                
                idxA = clean_data(ianimal).(['is_' lower(regA)]);
                idxB = clean_data(ianimal).(['is_' lower(regB)]);
                
                if sum(idxA) == 0 || sum(idxB) == 0; continue; end
                
                % Use Population Average as a proxy for macroscopic state (LFP analog)
                popA = squeeze(mean(clean_data(ianimal).neural(idxA, :, tr_idx), 1, 'omitnan'));
                popB = squeeze(mean(clean_data(ianimal).neural(idxB, :, tr_idx), 1, 'omitnan'));
                
                % [Window A x Window B]
                mi_mat = nan(n_windows, n_windows);
                mi_shuff = nan(n_windows, n_windows);
                
                for wA = 1:n_windows
                    b_idxA = win_starts(wA) : (win_starts(wA) + cfg.mi_pool_win - 1);
                    winA = popA(b_idxA, :); flatA = winA(:);
                    
                    if sum(~isnan(flatA)) < 5 || std(flatA, 'omitnan') == 0; continue; end
                    discA_full = discretize_equipop(flatA, cfg.mi_neural_bins);
                    
                    for wB = 1:n_windows
                        b_idxB = win_starts(wB) : (win_starts(wB) + cfg.mi_pool_win - 1);
                        winB = popB(b_idxB, :); flatB = winB(:);
                        
                        if sum(~isnan(flatB)) < 5 || std(flatB, 'omitnan') == 0; continue; end
                        discB_full = discretize_equipop(flatB, cfg.mi_neural_bins);
                        
                        % --- CRITICAL FIX: Extract indices valid in BOTH regions ---
                        valid_mask = ~isnan(discA_full) & ~isnan(discB_full);
                        if sum(valid_mask) < 5; continue; end
                        
                        discA = discA_full(valid_mask);
                        discB = discB_full(valid_mask);
                        
                        mi_mat(wA, wB) = calc_mi(discA, discB, cfg.mi_neural_bins, cfg.mi_neural_bins);
                        
                        s_vals = zeros(1, 5);
                        for s = 1:5
                            s_vals(s) = calc_mi(discA, discB(randperm(length(discB))), cfg.mi_neural_bins, cfg.mi_neural_bins); 
                        end
                        mi_shuff(wA, wB) = mean(s_vals);
                    end
                end
                
                area_mi_results(ianimal).(pair_name).epoch(i_ep).real = mi_mat;
                area_mi_results(ianimal).(pair_name).epoch(i_ep).shuff = mi_shuff;
            end
        end
    end
    save(cross_area_file, 'area_mi_results', '-v7.3');
    fprintf('Cross-Area MI computation complete.\n');
end

%% 9. Plotting: Cross-Area Information Flow (Lagged MI)
fprintf('--- Plotting Cross-Area Information Flow ---\n');
% Lag = Window A - Window B. 
% Positive Lag means Area A occurs BEFORE Area B.

cfg.area_max_lag = 10; % Max bins to shift between areas
lags_to_plot = -cfg.area_max_lag : cfg.area_max_lag;

figure('Position', [100 600 1500 500], 'Color', 'w', 'Name', 'Cross-Area Mutual Information');
t_flow = tiledlayout(1, 3, 'Padding', 'compact');
pair_colors = lines(3);

for i_ep = 1:3
    ax = nexttile(t_flow); hold(ax, 'on');
    legend_handles = [];
    pair_labels = {};
    
    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1};
        regB = cfg.area_pairs{p, 2};
        pair_name = sprintf('%s_%s', regA, regB);
        pair_labels{end+1} = sprintf('%s \\rightarrow %s', regA, regB);
        
        lag_curves = [];
        for ianimal = 1:n_animals
            % 1. Check if the field exists at all
            if isfield(area_mi_results(ianimal), pair_name)
                pair_data = area_mi_results(ianimal).(pair_name);
                
                % 2. CRITICAL FIX: Ensure it is actually a struct (not an empty []) before indexing
                if isstruct(pair_data) && isfield(pair_data, 'epoch') && length(pair_data.epoch) >= i_ep
                    mat = pair_data.epoch(i_ep).real;
                    mat_shuff = pair_data.epoch(i_ep).shuff;
                    
                    if isempty(mat); continue; end
                    delta_mat = mat - mat_shuff;
                    
                    mouse_lags = nan(1, length(lags_to_plot));
                    for l_idx = 1:length(lags_to_plot)
                        lag = lags_to_plot(l_idx);
                        % A positive lag means row (wA) is strictly greater than col (wB) 
                        % i.e., sub-diagonal
                        mouse_lags(l_idx) = mean(diag(delta_mat, -lag), 'omitnan');
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
    yline(0, 'k-', 'LineWidth', 0.5);
    
    xlim([-cfg.area_max_lag, cfg.area_max_lag]);
    ylim([-0.02 0.15]); % Adjust based on your bit scale
    
    title(epoch_names{i_ep}, 'FontSize', 14);
    if i_ep == 1; ylabel('Population \Delta MI (Bits)', 'FontWeight', 'bold'); end
    xlabel('Spatial Lag (Bins)');
    if i_ep == 3
        % Annotate directional flow
        text(0.1, -0.15, sprintf('\\leftarrow %s leads', 'Area B'), 'Units', 'normalized', 'HorizontalAlignment', 'left');
        text(0.9, -0.15, sprintf('%s leads \\rightarrow', 'Area A'), 'Units', 'normalized', 'HorizontalAlignment', 'right');
        legend(legend_handles, pair_labels, 'Location', 'northwest'); 
    end
    box on; grid on;
end
sgtitle('Cross-Area Information Flow Across Learning', 'FontSize', 16);

%% 10. Calculate Partial Information Decomposition (Shared Information)
fprintf('--- Computing PID Shared Information (Bidirectional Space-Lagged) ---\n');

% Parameters for PID
cfg.pid_max_lag = 10; % Max spatial lag in BOTH directions
pid_file = 'pid_shared_info_results.mat';

if exist(pid_file, 'file')
    fprintf('Loading existing PID results...\n');
    load(pid_file, 'pid_results');
else
    pid_results = struct();
    
    for ianimal = 1:n_animals
        fprintf('Processing Animal %d/%d for PID...\n', ianimal, n_animals);
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
                if length(tr_idx) < 3; continue; end
                
                for p = 1:size(cfg.area_pairs, 1)
                    regA = cfg.area_pairs{p, 1}; 
                    regB = cfg.area_pairs{p, 2}; 
                    pair_name = sprintf('%s_to_%s', regA, regB);
                    
                    idxA = clean_data(ianimal).(['is_' lower(regA)]);
                    idxB = clean_data(ianimal).(['is_' lower(regB)]);
                    if sum(idxA) == 0 || sum(idxB) == 0; continue; end
                    
                    % Macroscopic state proxy
                    popA = squeeze(mean(clean_data(ianimal).neural(idxA, :, tr_idx), 1, 'omitnan'));
                    popB = squeeze(mean(clean_data(ianimal).neural(idxB, :, tr_idx), 1, 'omitnan'));
                    behav = Y_all(:, tr_idx);
                    
                    % 2D Matrix: [Target Window x (2*MaxLag + 1)]
                    n_lags = 2 * cfg.pid_max_lag + 1;
                    si_mat = nan(n_windows, n_lags);
                    si_shuff = nan(n_windows, n_lags);
                    
                    for w_targ = 1:n_windows
                        b_idx = win_starts(w_targ) : (win_starts(w_targ) + cfg.mi_pool_win - 1);
                        
                        F_flat = behav(b_idx, :); F_flat = F_flat(:);
                        if sum(~isnan(F_flat)) < 5 || std(F_flat, 'omitnan') == 0; continue; end
                        F_disc_full = discretize_equipop(F_flat, cfg.mi_behav_bins);
                        
                        B_flat = popB(b_idx, :); B_flat = B_flat(:);
                        if sum(~isnan(B_flat)) < 5 || std(B_flat, 'omitnan') == 0; continue; end
                        B_disc_full = discretize_equipop(B_flat, cfg.mi_neural_bins);
                        
                        % Compute Bidirectional Lags
                        for lag_val = -cfg.pid_max_lag : cfg.pid_max_lag
                            lag_idx = lag_val + cfg.pid_max_lag + 1; 
                            w_source = w_targ - lag_val;
                            
                            % Check bounds for the shifted window
                            if w_source < 1 || w_source > n_windows; continue; end 
                            
                            b_idx_lag = win_starts(w_source) : (win_starts(w_source) + cfg.mi_pool_win - 1);
                            A_flat = popA(b_idx_lag, :); A_flat = A_flat(:);
                            
                            if sum(~isnan(A_flat)) < 5 || std(A_flat, 'omitnan') == 0; continue; end
                            A_disc_full = discretize_equipop(A_flat, cfg.mi_neural_bins);
                            
                            % Strict NaN filtering
                            valid = ~isnan(A_disc_full) & ~isnan(B_disc_full) & ~isnan(F_disc_full);
                            if sum(valid) < 5; continue; end
                            
                            A_val = A_disc_full(valid);
                            B_val = B_disc_full(valid);
                            F_val = F_disc_full(valid);
                            
                            % Calculate Marginal MIs
                            I_A_F = calc_mi(A_val, F_val, cfg.mi_neural_bins, cfg.mi_behav_bins);
                            I_B_F = calc_mi(B_val, F_val, cfg.mi_neural_bins, cfg.mi_behav_bins);
                            
                            % Minimum Mutual Information (MMI) bound for Shared Information
                            si_mat(w_targ, lag_idx) = min(I_A_F, I_B_F);
                            
                            % Shuffle Feature for null distribution
                            s_vals = zeros(1, 5);
                            for s = 1:5
                                F_shuff = F_val(randperm(length(F_val)));
                                s_vals(s) = min(calc_mi(A_val, F_shuff, cfg.mi_neural_bins, cfg.mi_behav_bins), ...
                                                calc_mi(B_val, F_shuff, cfg.mi_neural_bins, cfg.mi_behav_bins));
                            end
                            si_shuff(w_targ, lag_idx) = mean(s_vals);
                        end
                    end
                    pid_results(ianimal).(pair_name).(target_name).epoch(i_ep).real = si_mat;
                    pid_results(ianimal).(pair_name).(target_name).epoch(i_ep).shuff = si_shuff;
                end
            end
        end
    end
    save(pid_file, 'pid_results', '-v7.3');
    fprintf('PID Shared Information computation complete.\n');
end

%% 11. Plotting: Bidirectional Space-Lagged Shared Information (2D Heatmaps)
fprintf('--- Plotting Space-Lagged Shared Information (PID Heatmaps) ---\n');

lag_vector = -cfg.pid_max_lag : cfg.pid_max_lag;
epoch_names = {'Naive (Trials 1-10)', 'Pre-LP (-10 to -1)', 'Post-LP (+1 to +10)'};

for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    
    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1}; 
        regB = cfg.area_pairs{p, 2}; 
        pair_name = sprintf('%s_to_%s', regA, regB);
        
        figure('Position', [100 150 1400 450], 'Color', 'w', 'Name', sprintf('PID Heatmap %s - %s', pair_name, target));
        t_pid = tiledlayout(1, 3, 'Padding', 'compact');
        
        for i_ep = 1:3
            ax = nexttile(t_pid);
            
            delta_si_all = [];
            for ianimal = 1:n_animals
                if isfield(pid_results(ianimal), pair_name)
                    pair_data = pid_results(ianimal).(pair_name);
                    if isstruct(pair_data) && isfield(pair_data, target)
                        targ_data = pair_data.(target);
                        if isstruct(targ_data) && length(targ_data.epoch) >= i_ep
                            real_mat = targ_data.epoch(i_ep).real;
                            shuff_mat = targ_data.epoch(i_ep).shuff;
                            if ~isempty(real_mat)
                                delta_si_all = cat(3, delta_si_all, (real_mat - shuff_mat));
                            end
                        end
                    end
                end
            end
            
            if ~isempty(delta_si_all)
                mu_si = mean(delta_si_all, 3, 'omitnan');
                
                % Optional: Mild 2D smoothing to wash out noise
                mask = ~isnan(mu_si);
                mu_si(isnan(mu_si)) = 0; % Temp fill for smoothing
                smoothed_si = imgaussfilt(mu_si, 0.8, 'FilterDomain', 'spatial', 'padding', 'symmetric');
                smoothed_si(~mask) = nan; 
                
                % Plot 2D Heatmap: X-axis = Target Bin, Y-axis = Spatial Lag
                imagesc(1:size(smoothed_si,1), lag_vector, smoothed_si', [0, 0.05]); 
                colormap(gca, 'magma'); 
                
                if i_ep == 3; cb = colorbar; cb.Label.String = '\Delta Shared Info (Bits)'; end
                
                % Overlay Markers
                hold on;
                yline(0, 'w--', 'LineWidth', 1); % Instantaneous line
                xline(cfg.target_rz_bin, 'w:', 'LineWidth', 1.5); % RZ Start
                hold off;
                
                axis xy; % Origin at bottom left so +lags are up
                xlim([1 size(smoothed_si,1)]);
                ylim([-cfg.pid_max_lag, cfg.pid_max_lag]);
                
                title(epoch_names{i_ep}, 'FontSize', 13);
                xlabel('Spatial Bin (Behavior)');
                
                if i_ep == 1
                    % Fixed multiline label using Cell Array to avoid \n interpreter errors
                    ylabel({'Spatial Lag (Bins)', ...
                            sprintf('(- = %s Leads | + = %s Leads)', regB, regA)}, ...
                            'FontWeight', 'bold'); 
                end
            end
        end
        sgtitle(sprintf('Shared Information (PID): %s / %s encoding %s', regA, regB, strrep(target, '_', ' ')), 'FontSize', 16);
    end
end

%% 12. Plotting: Summary Quantifications of PID (1D Profiles & Asymmetry)
fprintf('--- Plotting Summary PID Quantifications ---\n');

for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    
    figure('Position', [100 650 1500 800], 'Color', 'w', 'Name', sprintf('PID Summary - %s', target));
    t_sum = tiledlayout(2, 3, 'Padding', 'compact');
    epoch_cols = {[0.298, 0.447, 0.690], [0.867, 0.518, 0.322], [0.333, 0.776, 0.333]}; 
    
    % --- Row 1: 1D Directional Flow Profiles (Averaged across corridor) ---
    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1}; 
        regB = cfg.area_pairs{p, 2}; 
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
                        shuff_mat = pair_data.(target).epoch(i_ep).shuff;
                        if ~isempty(real_mat)
                            delta_mat = real_mat - shuff_mat;
                            % Average across spatial dimension (rows) to get 1D curve
                            mouse_lag_curves = [mouse_lag_curves; mean(delta_mat, 1, 'omitnan')];
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
        if p == 1; ylabel('Mean \Delta Shared Info (Bits)'); end
        xlabel(sprintf('\\leftarrow %s Leads      %s Leads \\rightarrow', regB, regA));
        if p == 3; legend(legend_handles, epoch_names, 'Location', 'best'); end
        box on; grid on;
    end
    
    % --- Row 2: Directional Asymmetry Bar Charts ---
    % Quantify Area A leading (Positive Lags) vs Area B leading (Negative Lags)
    for p = 1:size(cfg.area_pairs, 1)
        regA = cfg.area_pairs{p, 1}; 
        regB = cfg.area_pairs{p, 2}; 
        pair_name = sprintf('%s_to_%s', regA, regB);
        
        ax = nexttile(t_sum); hold(ax, 'on');
        
        asym_data_A_leads = nan(3, n_animals);
        asym_data_B_leads = nan(3, n_animals);
        
        for i_ep = 1:3
            for ianimal = 1:n_animals
                if isfield(pid_results(ianimal), pair_name)
                    pair_data = pid_results(ianimal).(pair_name);
                    if isstruct(pair_data) && isfield(pair_data, target) && length(pair_data.(target).epoch) >= i_ep
                        real_mat = pair_data.(target).epoch(i_ep).real;
                        shuff_mat = pair_data.(target).epoch(i_ep).shuff;
                        if ~isempty(real_mat)
                            delta_mat = real_mat - shuff_mat;
                            
                            % Average across all spatial bins
                            mean_lags = mean(delta_mat, 1, 'omitnan');
                            
                            % Sum or Mean of strictly positive vs strictly negative lags
                            asym_data_B_leads(i_ep, ianimal) = mean(mean_lags(lag_vector < 0), 'omitnan');
                            asym_data_A_leads(i_ep, ianimal) = mean(mean_lags(lag_vector > 0), 'omitnan');
                        end
                    end
                end
            end
        end
        
        % Plotting grouped bars
        mu_A = mean(asym_data_A_leads, 2, 'omitnan');
        se_A = std(asym_data_A_leads, 0, 2, 'omitnan') ./ sqrt(n_animals);
        mu_B = mean(asym_data_B_leads, 2, 'omitnan');
        se_B = std(asym_data_B_leads, 0, 2, 'omitnan') ./ sqrt(n_animals);
        
        b = bar(1:3, [mu_A, mu_B], 'grouped');
        b(1).FaceColor = [0.2 0.6 0.8]; % Color for A leads
        b(2).FaceColor = [0.8 0.4 0.2]; % Color for B leads
        
        % Error bars
        x_A = b(1).XEndPoints;
        x_B = b(2).XEndPoints;
        errorbar(x_A, mu_A, se_A, 'k', 'linestyle', 'none', 'LineWidth', 1.5);
        errorbar(x_B, mu_B, se_B, 'k', 'linestyle', 'none', 'LineWidth', 1.5);
        
        xticks(1:3);
        xticklabels({'Naive', 'Pre-LP', 'Post-LP'});
        if p == 1; ylabel('Net Shared Info Capacity (Bits)'); end
        title(sprintf('Information Dominance'), 'FontSize', 12);
        legend({sprintf('%s Leads', regA), sprintf('%s Leads', regB)}, 'Location', 'northwest');
        box on; grid on;
    end
    
    sgtitle(sprintf('Summary: Directionality of Shared Information (%s)', strrep(target, '_', ' ')), 'FontSize', 16);
end

%% --- Helper Function for Fast MI Calculation ---
function I = compute_mi_core(x, y, nx, ny)
    % Accumulate 2D histogram of joint occurrences
    joint_counts = accumarray([x(:), y(:)], 1, [nx, ny]);
    
    % Convert counts to probabilities
    P_xy = joint_counts / sum(joint_counts(:));
    
    % Marginal probabilities
    P_x = sum(P_xy, 2);
    P_y = sum(P_xy, 1);
    
    % Compute I(X;Y) avoiding log(0)
    I = 0;
    [r, c] = find(P_xy > 0);
    for k = 1:length(r)
        px = P_x(r(k));
        py = P_y(c(k));
        pxy = P_xy(r(k), c(k));
        I = I + pxy * log2(pxy / (px * py));
    end
end