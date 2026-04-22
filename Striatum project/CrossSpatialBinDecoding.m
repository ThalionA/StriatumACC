%% 1. Configuration & Setup
clear; clc; close all;

cfg.data_file = 'preprocessed_data.mat';
cfg.save_file = 'cross_spatial_decoding_results.mat';
cfg.regions = {'DMS', 'DLS', 'ACC'};
cfg.colors = {[0 0.4470 0.7410], [0.4660 0.6740 0.1880], [0.8500 0.3250 0.0980]};

% Decoding parameters
cfg.ridge_lambda = 1.0; % L2 penalty
cfg.min_units = 5;      % Minimum units required to decode a region
cfg.behav_targets = {'lick_rate', 'velocity'};
cfg.n_shuffles = 100;   
cfg.corr_window = 9;    % Moving window (trials) for trial-resolved correlation

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
    
    z_err = preprocessed_data(ianimal).zscored_lick_errors(1:n_trials);
    lp = find(conv(double(z_err <= -2), ones(1,7), 'valid') == 7, 1, 'first');
    if isempty(lp); lp = NaN; end
    
    clean_data(ianimal).neural = preprocessed_data(ianimal).spatial_binned_fr_all(:, :, 1:n_trials);
    clean_data(ianimal).lick_rate = lick_rate'; 
    clean_data(ianimal).velocity = velocity';   
    clean_data(ianimal).lp = lp;
    clean_data(ianimal).n_trials = n_trials;
    clean_data(ianimal).is_dms = preprocessed_data(ianimal).is_dms;
    clean_data(ianimal).is_dls = preprocessed_data(ianimal).is_dls;
    clean_data(ianimal).is_acc = preprocessed_data(ianimal).is_acc;
end

%% 3. Cross-Spatial Bin Decoding (Target Bin vs. Previous Source Bins)
if exist(cfg.save_file, 'file')
    fprintf('Loading existing decoding results...\n');
    load(cfg.save_file, 'decoding_results');
else
    fprintf('--- Running Cross-Spatial Decoding ---\n');
    decoding_results = struct();
    
    for ianimal = 1:n_animals
        fprintf('Processing Animal %d/%d...\n', ianimal, n_animals);
        n_bins = size(clean_data(ianimal).neural, 2);
        n_trials = clean_data(ianimal).n_trials;
        
        for i_reg = 1:numel(cfg.regions)
            region = cfg.regions{i_reg};
            idx_reg = clean_data(ianimal).(['is_' lower(region)]);
            
            if sum(idx_reg) < cfg.min_units
                continue; 
            end
            
            reg_neural = clean_data(ianimal).neural(idx_reg, :, :); % [Units x Bins x Trials]
            
            for i_targ = 1:numel(cfg.behav_targets)
                target_name = cfg.behav_targets{i_targ};
                Y_all = clean_data(ianimal).(target_name); % [Bins x Trials]
                
                % Initialize 3D storage: [TargetBin x SourceBin x Trial]
                Y_pred = nan(n_bins, n_bins, n_trials);
                Y_pred_shuff = nan(n_bins, n_bins, n_trials);
                
                shuff_idx = randperm(n_trials);
                
                for target_b = 1:n_bins
                    Y = Y_all(target_b, :)';
                    Y_shuff = Y(shuff_idx);
                    
                    % Only decode using current and previous bins
                    for source_b = 1:target_b
                        X_raw = squeeze(reg_neural(:, source_b, :)); 
                        if size(X_raw, 2) == 1; X_raw = X_raw'; end
                        X = X_raw'; % [Trials x Units]
                        
                        mu_X = mean(X, 1);
                        sig_X = std(X, 0, 1);
                        sig_X(sig_X == 0) = 1; 
                        X_z = (X - mu_X) ./ sig_X;
                        
                        I = eye(size(X_z, 2));
                        
                        for test_trial = 1:n_trials
                            train_mask = true(n_trials, 1);
                            train_mask(test_trial) = false;
                            
                            X_train = X_z(train_mask, :);
                            X_test  = X_z(test_trial, :);
                            
                            % Real
                            Y_train = Y(train_mask);
                            W = (X_train' * X_train + cfg.ridge_lambda * I) \ (X_train' * Y_train);
                            Y_pred(target_b, source_b, test_trial) = X_test * W;
                            
                            % Shuffled
                            Y_train_shuff = Y_shuff(train_mask);
                            W_shuff = (X_train' * X_train + cfg.ridge_lambda * I) \ (X_train' * Y_train_shuff);
                            Y_pred_shuff(target_b, source_b, test_trial) = X_test * W_shuff;
                        end
                    end
                end
                
                % Calculate trial-resolved moving correlation
                half_win = floor(cfg.corr_window / 2);
                moving_r = nan(n_bins, n_bins, n_trials);
                moving_r_shuff = nan(n_bins, n_bins, n_trials);
                
                for target_b = 1:n_bins
                    for source_b = 1:target_b
                        for t = 1:n_trials
                            win = max(1, t - half_win) : min(n_trials, t + half_win);
                            if length(win) > 3
                                r = corrcoef(squeeze(Y_pred(target_b, source_b, win)), Y_all(target_b, win));
                                moving_r(target_b, source_b, t) = r(1,2);
                                
                                r_s = corrcoef(squeeze(Y_pred_shuff(target_b, source_b, win)), Y_all(target_b, win));
                                moving_r_shuff(target_b, source_b, t) = r_s(1,2);
                            end
                        end
                    end
                end
                
                decoding_results(ianimal).(region).(target_name).moving_r = moving_r;
                decoding_results(ianimal).(region).(target_name).moving_r_shuff = moving_r_shuff;
            end
        end
    end
    save(cfg.save_file, 'decoding_results', '-v7.3');
    fprintf('Decoding complete and saved.\n');
end

%% 4. Plotting: Cross-Spatial Heatmap (Averaged across trials)
fprintf('--- Plotting Cross-Spatial Profile ---\n');
figure('Position', [100 100 1400 600], 'Color', 'w', 'Name', 'Cross-Spatial Decoding Heatmaps');
tiledlayout(2, 3, 'Padding', 'compact');

for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    
    for i_reg = 1:numel(cfg.regions)
        region = cfg.regions{i_reg};
        nexttile;
        
        r_matrix_all = [];
        for ianimal = 1:n_animals
            if isfield(decoding_results(ianimal), region) && isfield(decoding_results(ianimal).(region), target)
                % Mean across all trials for this animal
                r_mat = mean(decoding_results(ianimal).(region).(target).moving_r, 3, 'omitnan');
                r_matrix_all = cat(3, r_matrix_all, r_mat);
            end
        end
        
        if ~isempty(r_matrix_all)
            mu_mat = mean(r_matrix_all, 3, 'omitnan');
            
            % Plot lower triangular heatmap
            imagesc(mu_mat);
            colormap(gca, 'parula');
            cb = colorbar;
            cb.Label.String = 'Pearson r';
            
            % Overlay the diagonal for reference
            hold on;
            plot([1 size(mu_mat,1)], [1 size(mu_mat,2)], 'w--', 'LineWidth', 1);
            hold off;
            
            axis square;
            set(gca, 'YDir', 'normal'); % Origin at bottom left
            title(sprintf('%s - %s', region, strrep(target, '_', ' ')));
            xlabel('Source Bin (Predictive Activity)');
            ylabel('Target Bin (Behavior)');
        end
    end
end

%% 5. Plotting: Evolution of Predictive vs Instantaneous Decoding Aligned to LP
fprintf('--- Plotting Learning Evolution ---\n');
align_win = -15:15; 
n_align = length(align_win);

figure('Position', [100 600 1200 500], 'Color', 'w', 'Name', 'LP-Aligned Predictive Evolution');
tiledlayout(1, 2, 'Padding', 'compact');

for i_targ = 1:numel(cfg.behav_targets)
    target = cfg.behav_targets{i_targ};
    nexttile; hold on;
    
    for i_reg = 1:numel(cfg.regions)
        region = cfg.regions{i_reg};
        
        r_predictive_all = nan(n_animals, n_align);
        for ianimal = 1:n_animals
            lp = clean_data(ianimal).lp;
            if isnan(lp); continue; end
            
            if isfield(decoding_results(ianimal), region) && isfield(decoding_results(ianimal).(region), target)
                mov_r = decoding_results(ianimal).(region).(target).moving_r; % [Target x Source x Trials]
                n_tr = size(mov_r, 3);
                n_b = size(mov_r, 1);
                
                % Calculate the average PREDICTIVE power per trial (Source < Target)
                pred_tr = nan(n_tr, 1);
                for t = 1:n_tr
                    mat_t = mov_r(:, :, t);
                    % Extract lower triangular part excluding the diagonal (strictly predictive)
                    pred_vals = mat_t(tril(true(n_b), -1)); 
                    pred_tr(t) = mean(pred_vals, 'omitnan');
                end
                
                for i_w = 1:n_align
                    tr_idx = lp + align_win(i_w);
                    if tr_idx >= 1 && tr_idx <= n_tr
                        r_predictive_all(ianimal, i_w) = pred_tr(tr_idx);
                    end
                end
            end
        end
        
        valid_mice = sum(~isnan(r_predictive_all), 1);
        mu = mean(r_predictive_all, 1, 'omitnan');
        se = std(r_predictive_all, 0, 1, 'omitnan') ./ sqrt(valid_mice);
        
        plot_idx = valid_mice >= 3;
        shadedErrorBar(align_win(plot_idx), mu(plot_idx), se(plot_idx), ...
            'lineprops', {'Color', cfg.colors{i_reg}, 'LineWidth', 2});
    end
    
    xline(0, 'k--', 'Learning Point', 'LineWidth', 1.5);
    title(sprintf('%s: Mean Predictive Power (Source < Target)', strrep(target, '_', ' ')));
    xlabel('Trials relative to LP');
    ylabel('Mean Predictive Accuracy (r)');
    if i_targ == 1; legend(cfg.regions, 'Location', 'best'); end
    grid on; box on;
end