%% 1. Configuration & Setup
clear; clc; close all;

% Project-wide constants (areas, colours, ridge, min_units, n_shuffles, ...)
cfg = project_cfg();

% Script-specific overrides
cfg.data_file   = cfg.task_data_file;
cfg.save_file   = 'processed_data/cross_spatial_decoding_results.mat';
cfg.regions     = cfg.areas;                                 % alias
cfg.colors      = mat2cell(cfg.area_colors, ones(size(cfg.area_colors,1),1), 3)';
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
    
    % LP via shared helper. Original convention here was "7 strictly
    % consecutive trials with z_err <= -2"; we map that to lp_window=7,
    % lp_min_consecutive=7. Truncate the trace to n_trials first so the
    % helper sees the same data the inline version did.
    pd_view = preprocessed_data(ianimal);
    pd_view.zscored_lick_errors = pd_view.zscored_lick_errors(1:n_trials);
    lp = find_learning_points(pd_view, struct( ...
        'lp_z_threshold', -2, 'lp_window', 7, 'lp_min_consecutive', 7));

    clean_data(ianimal).neural = preprocessed_data(ianimal).spatial_binned_fr_all(:, :, 1:n_trials);
    clean_data(ianimal).lick_rate = lick_rate'; 
    clean_data(ianimal).velocity = velocity';   
    clean_data(ianimal).lp = lp;
    clean_data(ianimal).n_trials = n_trials;
    clean_data(ianimal).is_dms = preprocessed_data(ianimal).is_dms;
    clean_data(ianimal).is_dls = preprocessed_data(ianimal).is_dls;
    clean_data(ianimal).is_acc = preprocessed_data(ianimal).is_acc;
    clean_data(ianimal).is_v1  = is_v1_safe(preprocessed_data(ianimal));
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

                % Cache features per source bin so we don't re-extract them
                % once per shuffle.
                X_by_source = cell(1, n_bins);
                for source_b = 1:n_bins
                    X_raw = squeeze(reg_neural(:, source_b, :));
                    if size(X_raw, 2) == 1; X_raw = X_raw'; end
                    X_by_source{source_b} = X_raw';   % [Trials x Units]
                end

                % --- Real predictions (closed-form PRESS LOO ridge) ---
                for target_b = 1:n_bins
                    Y = Y_all(target_b, :)';
                    for source_b = 1:target_b
                        Y_pred(target_b, source_b, :) = ...
                            loo_ridge_press(X_by_source{source_b}, Y, cfg.ridge_lambda);
                    end
                end

                % Calculate trial-resolved moving correlation
                half_win = floor(cfg.corr_window / 2);
                moving_r = nan(n_bins, n_bins, n_trials);

                % --- Shuffle distribution: cfg.n_shuffles trial-permutations ---
                % Online aggregates avoid storing the full 4D tensor.
                shuff_sum  = zeros(n_bins, n_bins, n_trials);
                shuff_sum2 = zeros(n_bins, n_bins, n_trials);
                shuff_cnt  = zeros(n_bins, n_bins, n_trials);
                shuff_above = zeros(n_bins, n_bins, n_trials); % cumulative count where shuff >= real (computed at end)
                moving_r_shuff_first = nan(n_bins, n_bins, n_trials);  % keep one shuffle for plotting compat
                
                % Real moving correlation
                for target_b = 1:n_bins
                    for source_b = 1:target_b
                        for t = 1:n_trials
                            win = max(1, t - half_win) : min(n_trials, t + half_win);
                            if length(win) > 3
                                r = corrcoef(squeeze(Y_pred(target_b, source_b, win)), Y_all(target_b, win));
                                moving_r(target_b, source_b, t) = r(1,2);
                            end
                        end
                    end
                end

                % Shuffle distribution: n_shuffles independent permutations.
                for s_iter = 1:cfg.n_shuffles
                    shuff_idx_s = randperm(n_trials);
                    Y_pred_shuff_s = nan(n_bins, n_bins, n_trials);
                    for target_b = 1:n_bins
                        Y_s = Y_all(target_b, shuff_idx_s)';
                        for source_b = 1:target_b
                            Y_pred_shuff_s(target_b, source_b, :) = ...
                                loo_ridge_press(X_by_source{source_b}, Y_s, cfg.ridge_lambda);
                        end
                    end
                    % Compute moving_r for this shuffle
                    moving_r_s = nan(n_bins, n_bins, n_trials);
                    for target_b = 1:n_bins
                        for source_b = 1:target_b
                            for t = 1:n_trials
                                win = max(1, t - half_win) : min(n_trials, t + half_win);
                                if length(win) > 3
                                    r_s = corrcoef(squeeze(Y_pred_shuff_s(target_b, source_b, win)), Y_all(target_b, win));
                                    moving_r_s(target_b, source_b, t) = r_s(1, 2);
                                end
                            end
                        end
                    end
                    % Online aggregation
                    valid = ~isnan(moving_r_s);
                    shuff_sum(valid)   = shuff_sum(valid)   + moving_r_s(valid);
                    shuff_sum2(valid)  = shuff_sum2(valid)  + moving_r_s(valid).^2;
                    shuff_cnt(valid)   = shuff_cnt(valid)   + 1;
                    shuff_above       = shuff_above + double(moving_r_s >= moving_r);
                    if s_iter == 1
                        moving_r_shuff_first = moving_r_s; % preserve a single sample for compat
                    end
                end
                shuff_mean = shuff_sum ./ max(shuff_cnt, 1);
                shuff_var  = max(0, shuff_sum2 ./ max(shuff_cnt, 1) - shuff_mean.^2);
                shuff_std  = sqrt(shuff_var);
                empirical_p = shuff_above ./ cfg.n_shuffles;     % one-sided

                decoding_results(ianimal).(region).(target_name).moving_r = moving_r;
                % Backwards-compat field (one realisation of the shuffle)
                decoding_results(ianimal).(region).(target_name).moving_r_shuff = moving_r_shuff_first;
                % New aggregates from the n_shuffles-bumped null
                decoding_results(ianimal).(region).(target_name).shuff_mean = shuff_mean;
                decoding_results(ianimal).(region).(target_name).shuff_std  = shuff_std;
                decoding_results(ianimal).(region).(target_name).empirical_p = empirical_p;
                decoding_results(ianimal).(region).(target_name).n_shuffles  = cfg.n_shuffles;
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