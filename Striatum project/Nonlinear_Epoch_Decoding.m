%% 1. Configuration & Setup
clear; clc; close all;

cfg.data_file = 'preprocessed_data.mat';
cfg.regions = {'DMS', 'DLS', 'ACC'};
cfg.colors = {[0 0.4470 0.7410], [0.4660 0.6740 0.1880], [0.8500 0.3250 0.0980]};

% --- Decoding Toggle ---
cfg.decoder_type = 'Ridge'; % Options: 'Ridge' (Linear, Fast) or 'GPR' (Non-linear, Slow)

% Decoding parameters
cfg.ridge_lambda = 1.0; % L2 penalty (only used if decoder_type is 'Ridge')
cfg.max_bin = 30;       % Clip decoding to Reward Zone Start (25) + 5 bins
cfg.target_rz_bin = 25; % Reward zone start bin for the "Look-ahead" plot
cfg.min_units = 5;      
cfg.behav_targets = {'lick_rate', 'velocity'};

% Dynamic save file based on decoder choice
cfg.save_file = sprintf('%s_epoch_decoding_results.mat', lower(cfg.decoder_type));

%% 2. Data Loading, Preprocessing & Epoch Extraction
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
    
    z_err = preprocessed_data(ianimal).zscored_lick_errors(1:n_trials);
    lp = find(conv(double(z_err <= -2), ones(1,7), 'valid') == 7, 1, 'first');
    if isempty(lp); lp = NaN; end
    
    % Clip space to max_bin
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

%% 3. Cross-Spatial Decoding (Ridge or GPR)
if exist(cfg.save_file, 'file')
    fprintf('Loading existing %s decoding results...\n', cfg.decoder_type);
    load(cfg.save_file, 'decoding_results');
else
    fprintf('--- Running %s Cross-Spatial Decoding ---\n', cfg.decoder_type);
    decoding_results = struct();
    
    for ianimal = 1:n_animals
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        n_bins = clean_data(ianimal).n_bins;
        n_trials = clean_data(ianimal).n_trials;
        lp = clean_data(ianimal).lp;
        
        % Define Trial Epochs: Naive, Pre-LP, Post-LP
        epochs = cell(1, 3);
        epochs{1} = 1:min(10, n_trials); 
        if ~isnan(lp)
            ep2 = (lp-10):(lp-1); epochs{2} = ep2(ep2 >= 1); 
            ep3 = (lp+1):(lp+10); epochs{3} = ep3(ep3 <= n_trials); 
        else
            epochs{2} = []; epochs{3} = [];
        end
        
        for i_reg = 1:numel(cfg.regions)
            region = cfg.regions{i_reg};
            idx_reg = clean_data(ianimal).(['is_' lower(region)]);
            
            if sum(idx_reg) < cfg.min_units; continue; end
            reg_neural = clean_data(ianimal).neural(idx_reg, :, :); % [Units x Bins x Trials]
            
            for i_targ = 1:numel(cfg.behav_targets)
                target_name = cfg.behav_targets{i_targ};
                Y_all = clean_data(ianimal).(target_name); % [Bins x Trials]
                
                for i_ep = 1:3
                    tr_idx = epochs{i_ep};
                    if length(tr_idx) < 5 % Minimum trials for valid CV
                        decoding_results(ianimal).(region).(target_name).epoch(i_ep).r_mat = nan(n_bins, n_bins);
                        decoding_results(ianimal).(region).(target_name).epoch(i_ep).r_shuff = nan(n_bins, n_bins);
                        continue;
                    end
                    
                    r_mat = nan(n_bins, n_bins);
                    r_shuff_mat = nan(n_bins, n_bins);
                    n_ep_trials = length(tr_idx);
                    
                    for target_b = 1:n_bins
                        Y_ep = Y_all(target_b, tr_idx)';
                        Y_shuff = Y_ep(randperm(n_ep_trials));
                        
                        if std(Y_ep) == 0; continue; end % Skip flat behavior
                        
                        for source_b = 1:target_b
                            X_raw = squeeze(reg_neural(:, source_b, tr_idx)); 
                            if size(X_raw, 2) == 1; X_raw = X_raw'; end
                            X_ep = X_raw'; % [Trials x Units]
                            
                            % Add tiny noise to prevent GPR covariance matrix singularity
                            if strcmp(cfg.decoder_type, 'GPR')
                                X_ep = X_ep + randn(size(X_ep)) * 1e-6; 
                            end
                            
                            Y_pred = nan(n_ep_trials, 1);
                            Y_pred_shuff = nan(n_ep_trials, 1);
                            
                            % Leave-One-Trial-Out Decoding
                            for test_t = 1:n_ep_trials
                                train_mask = true(n_ep_trials, 1);
                                train_mask(test_t) = false;
                                
                                X_train = X_ep(train_mask, :);
                                X_test  = X_ep(test_t, :);
                                
                                Y_train = Y_ep(train_mask);
                                Y_train_shuff = Y_shuff(train_mask);
                                
                                if strcmp(cfg.decoder_type, 'Ridge')
                                    % Manual Standardization for Ridge
                                    mu_X = mean(X_train, 1);
                                    sig_X = std(X_train, 0, 1); sig_X(sig_X == 0) = 1;
                                    X_train_z = (X_train - mu_X) ./ sig_X;
                                    X_test_z  = (X_test - mu_X) ./ sig_X;
                                    
                                    I = eye(size(X_train_z, 2));
                                    
                                    % Real
                                    W = (X_train_z' * X_train_z + cfg.ridge_lambda * I) \ (X_train_z' * Y_train);
                                    Y_pred(test_t) = X_test_z * W;
                                    
                                    % Shuffled
                                    W_s = (X_train_z' * X_train_z + cfg.ridge_lambda * I) \ (X_train_z' * Y_train_shuff);
                                    Y_pred_shuff(test_t) = X_test_z * W_s;
                                    
                                elseif strcmp(cfg.decoder_type, 'GPR')
                                    % Gaussian Process Regression
                                    try
                                        mdl = fitrgp(X_train, Y_train, 'KernelFunction', 'squaredexponential', 'Standardize', true, 'PredictMethod', 'exact');
                                        Y_pred(test_t) = predict(mdl, X_test);
                                        
                                        mdl_s = fitrgp(X_train, Y_train_shuff, 'KernelFunction', 'squaredexponential', 'Standardize', true, 'PredictMethod', 'exact');
                                        Y_pred_shuff(test_t) = predict(mdl_s, X_test);
                                    catch
                                        Y_pred(test_t) = mean(Y_train);
                                        Y_pred_shuff(test_t) = mean(Y_train_shuff);
                                    end
                                end
                            end
                            
                            % Calculate correlation
                            r = corrcoef(Y_pred, Y_ep); 
                            r_mat(target_b, source_b) = r(1,2);
                            
                            r_s = corrcoef(Y_pred_shuff, Y_ep); 
                            r_shuff_mat(target_b, source_b) = r_s(1,2);
                        end
                    end
                    decoding_results(ianimal).(region).(target_name).epoch(i_ep).r_mat = r_mat;
                    decoding_results(ianimal).(region).(target_name).epoch(i_ep).r_shuff = r_shuff_mat;
                end
            end
        end
    end
    save(cfg.save_file, 'decoding_results', '-v7.3');
    fprintf('%s decoding complete and saved.\n', cfg.decoder_type);
end

%% 4. Plotting: Delta Decoding ($\Delta r$) Evolution Approaching Reward Zone
fprintf('--- Plotting Spatial Look-Ahead Improvement (Delta r) ---\n');
epoch_names = {'Naive (Trials 1-10)', 'Pre-LP (-10 to -1)', 'Post-LP (+1 to +10)'};
targ_b = cfg.target_rz_bin; 
source_bins = 1:targ_b;

figure('Position', [100 150 1500 700], 'Color', 'w', 'Name', sprintf('[%s] Delta r Evolution Approaching RZ', cfg.decoder_type));
t_approach = tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    
    for i_ep = 1:3
        ax = nexttile(t_approach); hold(ax, 'on');
        legend_handles = [];
        
        for i_reg = 1:numel(cfg.regions)
            region = cfg.regions{i_reg};
            
            delta_curves = [];
            for ianimal = 1:n_animals
                if isfield(decoding_results(ianimal), region) && isfield(decoding_results(ianimal).(region), target)
                    if length(decoding_results(ianimal).(region).(target).epoch) >= i_ep
                        % Extract Target Bin 25
                        mat = decoding_results(ianimal).(region).(target).epoch(i_ep).r_mat;
                        mat_shuff = decoding_results(ianimal).(region).(target).epoch(i_ep).r_shuff;
                        
                        if size(mat, 1) >= targ_b
                            % Calculate Delta (Real - Shuff)
                            r_real = mat(targ_b, source_bins);
                            r_null = mat_shuff(targ_b, source_bins);
                            delta_curves = [delta_curves; (r_real - r_null)];
                        end
                    end
                end
            end
            
            if ~isempty(delta_curves)
                mu = mean(delta_curves, 1, 'omitnan');
                se = std(delta_curves, 0, 1, 'omitnan') ./ sqrt(size(delta_curves, 1));
                
                % Smooth slightly for visualization of trends
                mu_smooth = smoothdata(mu, 'gaussian', 3);
                
                h = shadedErrorBar(source_bins, mu_smooth, se, 'lineprops', {'Color', cfg.colors{i_reg}, 'LineWidth', 2.5});
                legend_handles(end+1) = h.mainLine;
            end
        end
        
        % Formatting
        xline(targ_b, 'r-', 'Reward Zone', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
        yline(0, 'k--', 'Chance Level (\Delta r = 0)', 'LineWidth', 1.5);
        
        xlim([1 targ_b]);
        ylim([-0.1 0.6]); % Standardized Y-axis for Delta r
        
        if i_targ == 1; title(sprintf('%s', epoch_names{i_ep}), 'FontSize', 13); end
        if i_ep == 1; ylabel(sprintf('%s\\n\\Delta Decoding (r)', strrep(target, '_', ' ')), 'FontWeight', 'bold', 'FontSize', 12); end
        if i_targ == 2; xlabel('Source Bin (Predictive Neural Activity)', 'FontSize', 11); end
        
        if i_ep == 3 && i_targ == 1; legend(legend_handles, cfg.regions, 'Location', 'northwest'); end
        box on; grid on;
    end
end
title(t_approach, sprintf('Evolution of Predictive Horizon (\\Delta r) Approaching Reward Zone [%s Decoder]', cfg.decoder_type), 'FontSize', 16);

%% 5. Plotting: Full Cross-Spatial Delta Heatmaps
fprintf('--- Plotting Cross-Spatial Delta Heatmaps ---\n');

for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    
    figure('Position', [100+i_targ*50 100+i_targ*50 1400 800], 'Color', 'w', 'Name', sprintf('[%s] Delta Heatmap - %s', cfg.decoder_type, target));
    t_heat = tiledlayout(3, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    for i_ep = 1:3
        for i_reg = 1:numel(cfg.regions)
            region = cfg.regions{i_reg};
            nexttile(t_heat);
            
            delta_matrix_all = [];
            for ianimal = 1:n_animals
                if isfield(decoding_results(ianimal), region) && isfield(decoding_results(ianimal).(region), target)
                    if length(decoding_results(ianimal).(region).(target).epoch) >= i_ep
                        mat = decoding_results(ianimal).(region).(target).epoch(i_ep).r_mat;
                        mat_shuff = decoding_results(ianimal).(region).(target).epoch(i_ep).r_shuff;
                        delta_matrix_all = cat(3, delta_matrix_all, (mat - mat_shuff));
                    end
                end
            end
            
            if ~isempty(delta_matrix_all)
                mu_mat = mean(delta_matrix_all, 3, 'omitnan');
                
                imagesc(mu_mat, [-0.1, 0.5]); % Standardized Delta C-axis
                colormap(gca, 'parula');
                if i_reg == 3; cb = colorbar; cb.Label.String = '\Delta r (Real - Shuff)'; end
                
                hold on;
                plot([1 size(mu_mat,1)], [1 size(mu_mat,2)], 'w--', 'LineWidth', 1); % Instantaneous Diagonal
                xline(cfg.target_rz_bin, 'r:', 'LineWidth', 1.5); 
                yline(cfg.target_rz_bin, 'r:', 'LineWidth', 1.5);
                hold off;
                
                axis square;
                set(gca, 'YDir', 'normal'); 
                xlim([1 cfg.max_bin]); ylim([1 cfg.max_bin]);
                
                if i_ep == 1; title(region, 'FontSize', 14); end
                if i_reg == 1; ylabel(sprintf('%s\nTarget Bin', epoch_names{i_ep}), 'FontWeight', 'bold'); end
                if i_ep == 3; xlabel('Source Bin'); end
            end
        end
    end
    sgtitle(sprintf('Target: %s | %s \\Delta r Mapping', strrep(target, '_', ' '), cfg.decoder_type), 'FontSize', 16);
end

%% 5. Plotting: Smoothed Heatmaps & Spatial Lag Profiles
fprintf('--- Quantifying Cross-Spatial Matrices ---\n');

% Parameters for summaries
smooth_sigma = 0.5; % Mild smoothing for heatmaps
max_lag = 20; % Maximum spatial lag to plot (in bins)
epoch_names = {'Naive (Trials 1-10)', 'Intermediate (-10 to -1)', 'Expert (+1 to +10)'};

for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    
    % =====================================================================
    % 5A. Smoothed Delta Heatmaps
    % =====================================================================
    fig_heat = figure('Position', [50, 50, 1200, 900], 'Color', 'w', 'Name', sprintf('[%s] Smoothed Heatmaps - %s', cfg.decoder_type, target));
    t_heat = tiledlayout(3, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    for i_ep = 1:3
        for i_reg = 1:numel(cfg.regions)
            region = cfg.regions{i_reg};
            nexttile(t_heat);
            
            delta_matrix_all = [];
            for ianimal = 1:n_animals
                if isfield(decoding_results(ianimal), region) && isfield(decoding_results(ianimal).(region), target)
                    if length(decoding_results(ianimal).(region).(target).epoch) >= i_ep
                        mat = decoding_results(ianimal).(region).(target).epoch(i_ep).r_mat;
                        mat_shuff = decoding_results(ianimal).(region).(target).epoch(i_ep).r_shuff;
                        delta_matrix_all = cat(3, delta_matrix_all, (mat - mat_shuff));
                    end
                end
            end
            
            if ~isempty(delta_matrix_all)
                % Average across mice
                mu_mat = mean(delta_matrix_all, 3, 'omitnan');
                
                % Apply 2D Gaussian smoothing to reveal macroscopic structure
                % (Ignore NaNs / upper triangle during smoothing)
                mask = tril(true(size(mu_mat)));
                mu_mat(~mask) = nan;
                smoothed_mat = imgaussfilt(mu_mat, smooth_sigma, 'FilterDomain', 'spatial', 'padding', 'symmetric');
                smoothed_mat(~mask) = nan; % Re-mask upper triangle
                
                imagesc(smoothed_mat, [-0.1, 0.4]); % Fixed color axis
                colormap(gca, 'parula');
                if i_reg == 3; cb = colorbar; cb.Label.String = 'Smoothed \Delta r'; end
                
                hold on;
                plot([1 size(mu_mat,1)], [1 size(mu_mat,2)], 'w--', 'LineWidth', 1); % Diagonal
                hold off;
                
                axis square; set(gca, 'YDir', 'normal'); 
                xlim([1 cfg.max_bin]); ylim([1 cfg.max_bin]);
                
                if i_ep == 1; title(region, 'FontSize', 14); end
                if i_reg == 1; ylabel(sprintf('%s\nTarget Bin', epoch_names{i_ep}), 'FontWeight', 'bold'); end
                if i_ep == 3; xlabel('Source Bin'); end
            end
        end
    end
    sgtitle(sprintf('Target: %s | Smoothed %s \\Delta r', strrep(target, '_', ' '), cfg.decoder_type), 'FontSize', 16);
    
    % =====================================================================
    % 5B. Decoding Decay by Spatial Lag (1D Profile)
    % =====================================================================
    % Lag = Target Bin - Source Bin. 
    % Lag 0 = Diagonal (Instantaneous). Lag 5 = predicting 5 bins into the future.
    
    fig_lag = figure('Position', [100, 100, 1400, 400], 'Color', 'w', 'Name', sprintf('[%s] Spatial Lag Profile - %s', cfg.decoder_type, target));
    t_lag = tiledlayout(1, 3, 'Padding', 'compact');
    
    for i_reg = 1:numel(cfg.regions)
        region = cfg.regions{i_reg};
        nexttile(t_lag); hold on;
        
        legend_handles = [];
        
        for i_ep = 1:3
            lag_curves = []; % [Mice x Lags]
            
            for ianimal = 1:n_animals
                if isfield(decoding_results(ianimal), region) && isfield(decoding_results(ianimal).(region), target)
                    if length(decoding_results(ianimal).(region).(target).epoch) >= i_ep
                        mat = decoding_results(ianimal).(region).(target).epoch(i_ep).r_mat;
                        mat_shuff = decoding_results(ianimal).(region).(target).epoch(i_ep).r_shuff;
                        delta_mat = mat - mat_shuff;
                        
                        mouse_lags = nan(1, max_lag + 1);
                        for lag = 0:max_lag
                            % Extract the k-th sub-diagonal (negative k means lower triangle)
                            sub_diag = diag(delta_mat, -lag);
                            mouse_lags(lag + 1) = mean(sub_diag, 'omitnan');
                        end
                        lag_curves = [lag_curves; mouse_lags];
                    end
                end
            end
            
            if ~isempty(lag_curves)
                mu_lag = mean(lag_curves, 1, 'omitnan');
                se_lag = std(lag_curves, 0, 1, 'omitnan') ./ sqrt(size(lag_curves, 1));
                
                % We use epoch_colors if you defined them, otherwise use generic lines
                epoch_cols = {[0.298, 0.447, 0.690], [0.867, 0.518, 0.322], [0.333, 0.776, 0.333]}; 
                
                h = shadedErrorBar(0:max_lag, mu_lag, se_lag, 'lineprops', {'-', 'Color', epoch_cols{i_ep}, 'LineWidth', 2});
                legend_handles(end+1) = h.mainLine;
            end
        end
        
        yline(0, 'k--', 'LineWidth', 1);
        xlim([0 max_lag]);
        
        % Force consistent Y-axis limits across regions for easy visual comparison
        % ylim([-0.05 0.5]); 
        
        title(region, 'FontSize', 14);
        xlabel('Spatial Lag (Target Bin - Source Bin)');
        if i_reg == 1; ylabel('Mean \Delta r'); end
        if i_reg == 3; legend(legend_handles, epoch_names, 'Location', 'northeast'); end
        grid on; box on;
        
    end
    linkaxes
    title(t_lag, sprintf('Predictive Horizon: Decoding Decay vs. Spatial Lag (%s)', strrep(target, '_', ' ')), 'FontSize', 16);
    
    % =====================================================================
    % 5C. Summary Quantifications: Instantaneous vs. Predictive (Bar Chart)
    % =====================================================================
    fig_bar = figure('Position', [150, 150, 1000, 400], 'Color', 'w', 'Name', sprintf('[%s] Decoding Modes - %s', cfg.decoder_type, target));
    t_bar = tiledlayout(1, 2, 'Padding', 'compact');
    
    % Storage for means: [Region x Epoch x Mouse]
    inst_data = nan(3, 3, n_animals);
    pred_data = nan(3, 3, n_animals);
    
    for i_reg = 1:numel(cfg.regions)
        region = cfg.regions{i_reg};
        for i_ep = 1:3
            for ianimal = 1:n_animals
                if isfield(decoding_results(ianimal), region) && isfield(decoding_results(ianimal).(region), target)
                    if length(decoding_results(ianimal).(region).(target).epoch) >= i_ep
                        mat = decoding_results(ianimal).(region).(target).epoch(i_ep).r_mat;
                        mat_shuff = decoding_results(ianimal).(region).(target).epoch(i_ep).r_shuff;
                        delta_mat = mat - mat_shuff;
                        
                        % Instantaneous = Mean of the main diagonal
                        inst_data(i_reg, i_ep, ianimal) = mean(diag(delta_mat), 'omitnan');
                        
                        % Predictive = Mean of the strictly lower triangle (lags > 0)
                        mask_pred = tril(true(size(delta_mat)), -1);
                        pred_data(i_reg, i_ep, ianimal) = mean(delta_mat(mask_pred), 'omitnan');
                    end
                end
            end
        end
    end
    
    % Plot Instantaneous
    nexttile(t_bar); hold on;
    for i_reg = 1:3
        mu = mean(squeeze(inst_data(i_reg, :, :)), 2, 'omitnan');
        se = std(squeeze(inst_data(i_reg, :, :)), 0, 2, 'omitnan') ./ sqrt(n_animals);
        errorbar((1:3) + (i_reg-2)*0.1, mu, se, '-o', 'Color', cfg.colors{i_reg}, 'LineWidth', 2, 'MarkerFaceColor', cfg.colors{i_reg});
    end
    yline(0, 'k--');
    xticks(1:3); xticklabels({'Naive', 'Pre-LP', 'Post-LP'});
    xlim([0.5 3.5]); 
    % ylim([-0.05 0.5]);
    title('Instantaneous Decoding (Diagonal)'); ylabel('Mean \Delta r');
    grid on; box on;
    
    % Plot Predictive
    nexttile(t_bar); hold on;
    for i_reg = 1:3
        mu = mean(squeeze(pred_data(i_reg, :, :)), 2, 'omitnan');
        se = std(squeeze(pred_data(i_reg, :, :)), 0, 2, 'omitnan') ./ sqrt(n_animals);
        errorbar((1:3) + (i_reg-2)*0.1, mu, se, '-o', 'Color', cfg.colors{i_reg}, 'LineWidth', 2, 'MarkerFaceColor', cfg.colors{i_reg});
    end
    yline(0, 'k--');
    xticks(1:3); xticklabels({'Naive', 'Pre-LP', 'Post-LP'});
    xlim([0.5 3.5]); 
    % ylim([-0.05 0.2]); % Note tighter Y-axis for predictive
    title('Predictive Decoding (Lower Triangle)'); 
    legend(cfg.regions, 'Location', 'northwest');
    grid on; box on;
    
    title(t_bar, sprintf('Evolution of Decoding Modes across Learning (%s)', strrep(target, '_', ' ')), 'FontSize', 16);
    linkaxes
end