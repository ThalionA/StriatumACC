%% CCA_striatum_spatial_v3.m
% Spatial CCA driver for the Striatum project, v5 spec.
%
% Composes the v5_* primitives into the H&H-adapted CCA pipeline for the
% Striatum dataset, spatial alignment mode. One fit per
% (animal × area-pair × epoch) with epochs = {naive, pre-LP, post-LP} as
% 10-trial windows.
%
% Spec: /Users/theoamvr/Documents/ResearchVault/Methods/CCA_HH_Adapted.md
%       See §6.2 (project deviations) and §6.2.x (data IO contract).
%
% Mirrors the structure of
%   /Users/theoamvr/Desktop/Experiments/TomLearning/HC_V1_Code/CCA_HC_V1_spatial_v3.m
% with three differences:
%   1. Data layout: pre-binned (n_units × n_bins × n_trials) per animal in
%      `preprocessed_data.mat`, with per-area logical masks `is_<area>`.
%   2. Learning-point detection: in-pipeline via `processTaskData`, not
%      from a pre-computed `animal_behaviour.mat`. Non-learners are DROPPED.
%   3. Disengagement truncation on the trial axis is supported (legacy v2
%      convention).
%
% Project-wide constants come from `project_cfg.m`; override per-script
% knobs in §1 below.

clear; clc; close all;

%% 1. CONFIG

cfg = project_cfg();

% --- Paths ---
cfg.save_dir    = fullfile(pwd, 'CCA_Results');
cfg.current_date = datestr(now, 'yyyy_mm_dd');
cfg.save_path   = fullfile(cfg.save_dir, ...
    sprintf('Striatum_CCA_v3_%s.mat', cfg.current_date));
if ~exist(cfg.save_dir, 'dir'), mkdir(cfg.save_dir); end

% --- Analysis selection ---
cfg.analysis_mode      = 'task_only';
cfg.areas_to_include   = cfg.areas;             % all 6 from project_cfg
cfg.area_pairs_v5      = cfg.area_pairs;        % all 15 pairs from project_cfg
n_pairs                = size(cfg.area_pairs_v5, 1);

% --- processTaskData parameters (alias project_cfg) ---
cfg.task_lp_zscore_threshold = cfg.lp_z_threshold;
cfg.task_lp_window_length    = cfg.lp_window;
cfg.task_lp_min_consecutive  = cfg.lp_min_consecutive;
cfg.control_epoch_method     = 'fixed_trial';
cfg.control_fixed_ref_trial  = 40;
cfg.control_epoch_windows    = {1:10, [-10, -1], [1, 10]};

% --- Spatial geometry (override for spatial CCA specifically) ---
cfg.truncate_at_disengagement = true;
cfg.spatial_truncate_at_max_bin = false;   % use cfg.max_bin (= 30) not cfg.n_bins_full (= 50)
if cfg.spatial_truncate_at_max_bin
    cfg.n_position_bins = cfg.max_bin;     % 30 bins × 5 cm = 150 cm corridor pre-RZ
else
    cfg.n_position_bins = cfg.n_bins_full; % full 50 bins
end

% --- Epoch definition ---
cfg.n_trials_per_epoch = 10;

% --- Unit inclusion ---
cfg.min_units_per_region = cfg.min_units;     % from project_cfg (= 5)

% --- PCA / CCA k rule (spec §3.2) ---
cfg.k_cap = 30;
cfg.k_samples_rule = 50;
cfg.k_variance_target = 0.80;
cfg.k_mode = 'samples';    % 'samples' | 'variance'

% --- Shuffle null (spec §3.1) ---
cfg.n_shuffles = 50;
cfg.cv_splits = 5;

% --- Lagged-refit IFI (position-bin lags) ---
cfg.max_lag_bins = 3;
cfg.central_window_bins = 3;
cfg.smooth_size = 3;
cfg.smooth_sigma = 1.0;

% --- Projection IFI ---
cfg.proj_max_lag_bins = 3;
cfg.proj_min_paired_samples = 5;

% --- Reproducibility ---
cfg.cca_fold_seed = 0;
cfg.shuffle_seed = 0;

% Make sure v5 primitives are on path.
addpath(pwd);

fprintf('CCA_striatum_spatial_v3 — %s\n', cfg.current_date);
fprintf('  Output: %s\n', cfg.save_path);
fprintf('  Spatial bins: %d × %.1f cm = %.0f cm corridor\n', ...
    cfg.n_position_bins, cfg.bin_size_cm, cfg.n_position_bins * cfg.bin_size_cm);

%% 2. LOAD TASK DATA + LEARNING POINTS

if ~isfile(cfg.task_data_file)
    error('Task data file not found: %s', cfg.task_data_file);
end
fprintf('  Loading task data from %s\n', cfg.task_data_file);
loaded_data = load(cfg.task_data_file, 'preprocessed_data');
task_data_raw = filterDataByArea(loaded_data.preprocessed_data, ...
    cfg.areas_to_include, cfg.area_field_map);
fprintf('  Filtered to %d areas: %s\n', numel(cfg.areas_to_include), ...
    strjoin(cfg.areas_to_include, ', '));

[task_data, learning_points_cell, avg_learning_point] = ...
    processTaskData(task_data_raw, cfg);
n_animals = size(task_data, 2);
if n_animals == 0
    error('No animals with defined learning point after processTaskData.');
end

% Striatum convention: non-learners already dropped → all included animals
% are learners by definition. Keep the flag for plot-compatibility with HC.
is_learner = true(n_animals, 1);
learning_points = nan(n_animals, 1);
for i = 1:n_animals
    if ~isempty(learning_points_cell{i})
        learning_points(i) = learning_points_cell{i};
    end
end
analysis_lp = learning_points;
fprintf('  %d learner animals included. Avg LP = %d.\n', n_animals, avg_learning_point);

%% 3. INITIALISE RESULTS

group_results = init_results_struct(cfg, n_animals, n_pairs);

%% 4. PER-ANIMAL LOOP

for ianimal = 1:n_animals
    % mouse_id = task_data(ianimal).mouseid;
    % fprintf('\n[%d/%d] Mouse %s (LP=%d)\n', ianimal, n_animals, ...
    %     num2str(mouse_id), analysis_lp(ianimal));

    try
        animal = build_animal_tensor(task_data(ianimal), cfg);
    catch err
        warning('  Failed to build tensor for %s: %s', num2str(mouse_id), err.message);
        continue;
    end
    if isempty(fieldnames(animal.area_tensors))
        fprintf('  No valid areas after unit selection. Skipping.\n');
        continue;
    end

    % Epochs.
    n_trials = animal.n_trials;
    lp = analysis_lp(ianimal);
    epochs = epoch_indices_for_animal(n_trials, lp, cfg.n_trials_per_epoch);
    if isempty(epochs)
        fprintf('  No valid epochs (LP or trial count out of range). Skipping.\n');
        continue;
    end

    % Per area-pair loop.
    for ipair = 1:n_pairs
        a1 = cfg.area_pairs_v5{ipair, 1};
        a2 = cfg.area_pairs_v5{ipair, 2};
        if ~isfield(animal.area_tensors, a1) || ~isfield(animal.area_tensors, a2)
            continue;
        end
        S_x = animal.area_tensors.(a1);   % (n_trials × n_bins × n_units_x)
        S_y = animal.area_tensors.(a2);
        n_units_x = size(S_x, 3);
        n_units_y = size(S_y, 3);

        % k fixed within (animal × pair) across the three epochs.
        smallest_epoch_size = min(cellfun(@(e) numel(e), {epochs.trials_idx}));
        n_samples_smallest = smallest_epoch_size * cfg.n_position_bins;
        k_chosen = choose_k(cfg, n_units_x, n_units_y, n_samples_smallest, ...
                            S_x, S_y, epochs(1).trials_idx);

        prev_A = []; prev_B = [];
        for iep = 1:numel(epochs)
            tr_idx = epochs(iep).trials_idx;
            S_x_ep = S_x(tr_idx, :, :);
            S_y_ep = S_y(tr_idx, :, :);

            res = fit_one(S_x_ep, S_y_ep, k_chosen, cfg);

            if ~isempty(prev_A)
                res.principal_angles_A = v5_principal_angles(prev_A, res.A);
                res.principal_angles_B = v5_principal_angles(prev_B, res.B);
            else
                res.principal_angles_A = nan(min(size(res.A, 2), 1), 1);
                res.principal_angles_B = nan(min(size(res.B, 2), 1), 1);
            end
            prev_A = res.A;  prev_B = res.B;

            group_results(ipair).per_epoch(ianimal, iep) = res;
        end
        fprintf('  pair %s-%s: k=%d, samples/var=%.1f, cc_cv=', ...
            a1, a2, k_chosen, n_samples_smallest / (2 * k_chosen));
        cc_vec = zeros(1, numel(epochs));
        for iep = 1:numel(epochs)
            cc_vec(iep) = group_results(ipair).per_epoch(ianimal, iep).cc_cv(1);
        end
        fprintf('%s\n', mat2str(cc_vec, 3));
    end
end

%% 5. SAVE

save(cfg.save_path, 'group_results', 'cfg', 'learning_points', ...
    'analysis_lp', 'is_learner', '-v7.3');
fprintf('\nSaved to %s\n', cfg.save_path);


%% Helpers

function animal = build_animal_tensor(animal_data_raw, cfg)
% Build per-area (n_trials × n_pos_bins × n_units) tensors from one animal's
% preprocessed_data row. Applies disengagement truncation and spatial-bin
% truncation per cfg flags.
    if ~isfield(animal_data_raw, 'spatial_binned_fr_all')
        error('build_animal_tensor:missingField', ...
            'animal struct missing spatial_binned_fr_all');
    end
    fr_all = animal_data_raw.spatial_binned_fr_all;   % (n_units × n_bins × n_trials)
    [n_units, n_bins_data, n_trials_raw] = size(fr_all);

    % Disengagement truncation on the trial axis.
    if cfg.truncate_at_disengagement && isfield(animal_data_raw, 'change_point_mean')
        cp = animal_data_raw.change_point_mean;
        if ~isempty(cp) && ~isnan(cp)
            diseng_idx = min([cp, n_trials_raw]);
            fr_all = fr_all(:, :, 1:diseng_idx);
            n_trials_raw = size(fr_all, 3);
        end
    end

    % Spatial-bin truncation.
    if n_bins_data < cfg.n_position_bins
        error('build_animal_tensor:badShape', ...
            'animal has %d bins; cfg expects %d', n_bins_data, cfg.n_position_bins);
    end
    fr_trunc = fr_all(:, 1:cfg.n_position_bins, :);

    % Permute to (n_trials × n_bins × n_units) for v5 primitives.
    fr_tbu = permute(fr_trunc, [3 2 1]);

    % Build per-area tensors via is_<area> masks.
    area_tensors = struct();
    area_kept_counts = struct();
    for ia = 1:numel(cfg.areas_to_include)
        area_name = cfg.areas_to_include{ia};
        field = cfg.area_field_map(area_name);
        if ~isfield(animal_data_raw, field), continue; end
        u_logical = logical(animal_data_raw.(field));
        u_logical = u_logical(:);
        if numel(u_logical) ~= n_units
            warning('build_animal_tensor:maskMismatch', ...
                '%s mask length %d != n_units %d. Skipping area.', ...
                area_name, numel(u_logical), n_units);
            continue;
        end
        if sum(u_logical) < cfg.min_units_per_region, continue; end
        area_tensors.(area_name) = fr_tbu(:, :, u_logical);
        area_kept_counts.(area_name) = sum(u_logical);
    end

    animal = struct( ...
        'n_trials', n_trials_raw, ...
        'n_pos_bins', cfg.n_position_bins, ...
        'area_tensors', area_tensors, ...
        'area_kept_counts', area_kept_counts);
end


function epochs = epoch_indices_for_animal(n_trials, lp, n_per_epoch)
    epochs = struct('name', {}, 'trials_idx', {}, 'iep', {});
    if n_trials < 3 * n_per_epoch || lp < n_per_epoch || lp + n_per_epoch > n_trials
        return;
    end
    epochs(1) = struct('name', 'naive',   'trials_idx', 1:n_per_epoch, 'iep', 1);
    epochs(2) = struct('name', 'pre_lp',  'trials_idx', (lp - n_per_epoch + 1):lp, 'iep', 2);
    epochs(3) = struct('name', 'post_lp', 'trials_idx', (lp + 1):(lp + n_per_epoch), 'iep', 3);
end


function k = choose_k(cfg, n_units_x, n_units_y, n_samples, S_x_full, S_y_full, ref_trials)
    k_units = min(n_units_x, n_units_y);
    k_samples = floor(n_samples / cfg.k_samples_rule);
    k_default = min([k_units, k_samples, cfg.k_cap]);
    switch lower(cfg.k_mode)
        case 'samples'
            k = max(1, k_default);
        case 'variance'
            S_x_ref = S_x_full(ref_trials, :, :);
            S_y_ref = S_y_full(ref_trials, :, :);
            S_x_res = v5_residualise(S_x_ref, ones(size(S_x_ref, 1), 1));
            S_y_res = v5_residualise(S_y_ref, ones(size(S_y_ref, 1), 1));
            [~, state_x] = v5_pca_reduce(S_x_res, min(k_default, size(S_x_ref, 3)));
            [~, state_y] = v5_pca_reduce(S_y_res, min(k_default, size(S_y_ref, 3)));
            k_x = find(cumsum(state_x.explained_variance_ratio) >= cfg.k_variance_target, 1, 'first');
            k_y = find(cumsum(state_y.explained_variance_ratio) >= cfg.k_variance_target, 1, 'first');
            if isempty(k_x), k_x = numel(state_x.explained_variance_ratio); end
            if isempty(k_y), k_y = numel(state_y.explained_variance_ratio); end
            k = max(1, min([k_x, k_y, k_default]));
        otherwise
            error('Unknown k_mode: %s', cfg.k_mode);
    end
end


function res = fit_one(S_x_3d, S_y_3d, k, cfg)
% One CCA fit for one (animal × area-pair × epoch). Identical body to
% CCA_HC_V1_spatial_v3's fit_one — kept in lock-step.
    [n_trials, n_pos_bins, ~] = size(S_x_3d);
    cl = ones(n_trials, 1);

    S_x_res = v5_residualise(S_x_3d, cl);
    S_y_res = v5_residualise(S_y_3d, cl);

    [P_x_3d, state_x] = v5_pca_reduce(S_x_res, k);
    [P_y_3d, state_y] = v5_pca_reduce(S_y_res, k);

    P_x_flat = reshape(permute(P_x_3d, [2 1 3]), n_trials * n_pos_bins, k);
    P_y_flat = reshape(permute(P_y_3d, [2 1 3]), n_trials * n_pos_bins, k);

    fit_res = v5_cca_fit_cv(P_x_flat, P_y_flat, cfg.cv_splits, true, cfg.cca_fold_seed);

    null = v5_shuffle_null(P_x_flat, P_y_flat, cfg.n_shuffles, n_pos_bins, ...
                            cfg.cv_splits, cfg.shuffle_seed, true, cfg.cca_fold_seed);

    lagged = v5_lagged_refit_ifi(P_x_3d, P_y_3d, ...
        cfg.max_lag_bins, cfg.central_window_bins, ...
        cfg.smooth_size, cfg.smooth_sigma);

    u_3d = v5_project(P_x_3d, fit_res.x_mean, fit_res.A(:, 1));
    v_3d = v5_project(P_y_3d, fit_res.y_mean, fit_res.B(:, 1));
    u = squeeze(u_3d);
    v = squeeze(v_3d);
    if isvector(u) && n_trials > 1
        u = reshape(u, n_trials, n_pos_bins);
        v = reshape(v, n_trials, n_pos_bins);
    elseif isvector(u) && n_trials == 1
        u = u(:).'; v = v(:).';
    end
    [ifi_proj_pt, lag_corr_pt] = v5_projection_ifi(u, v, ...
        cfg.proj_max_lag_bins, cfg.proj_min_paired_samples);

    cc_excess = fit_res.cc_cv(1) - null.mean;
    res = struct( ...
        'A', fit_res.A, 'B', fit_res.B, ...
        'x_mean', fit_res.x_mean, 'y_mean', fit_res.y_mean, ...
        'cc_full', fit_res.cc, 'cc_cv', fit_res.cc_cv, ...
        'cc_cv_per_fold', fit_res.cc_cv_per_fold, ...
        'cc_excess', cc_excess, ...
        'null_threshold', null.threshold, 'null_mean', null.mean, ...
        'null_cc1', null.cc1_null, ...
        'ifi_lagged', lagged.ifi, ...
        'ifi_lagged_curve_xy', lagged.cc_xy, ...
        'ifi_lagged_curve_yx', lagged.cc_yx, ...
        'ifi_lagged_curve_xy_smooth', lagged.cc_xy_smooth, ...
        'ifi_lagged_curve_yx_smooth', lagged.cc_yx_smooth, ...
        'ifi_proj_per_trial', ifi_proj_pt, ...
        'ifi_proj_lag_corr', lag_corr_pt, ...
        'k_used', k, ...
        'var_x_at_k', sum(state_x.explained_variance_ratio), ...
        'var_y_at_k', sum(state_y.explained_variance_ratio), ...
        'n_samples', n_trials * n_pos_bins, ...
        'samples_per_var', (n_trials * n_pos_bins) / (2 * k), ...
        'principal_angles_A', [], 'principal_angles_B', []);
end


function group_results = init_results_struct(cfg, n_animals, n_pairs)
    empty_res = struct( ...
        'A', [], 'B', [], 'x_mean', [], 'y_mean', [], ...
        'cc_full', [], 'cc_cv', [], 'cc_cv_per_fold', [], 'cc_excess', NaN, ...
        'null_threshold', NaN, 'null_mean', NaN, 'null_cc1', [], ...
        'ifi_lagged', NaN, 'ifi_lagged_curve_xy', [], 'ifi_lagged_curve_yx', [], ...
        'ifi_lagged_curve_xy_smooth', [], 'ifi_lagged_curve_yx_smooth', [], ...
        'ifi_proj_per_trial', [], 'ifi_proj_lag_corr', [], ...
        'k_used', NaN, 'var_x_at_k', NaN, 'var_y_at_k', NaN, ...
        'n_samples', NaN, 'samples_per_var', NaN, ...
        'principal_angles_A', [], 'principal_angles_B', []);
    group_results = struct();
    for ipair = 1:n_pairs
        group_results(ipair).pair_name = sprintf('%s-%s', ...
            cfg.area_pairs_v5{ipair, 1}, cfg.area_pairs_v5{ipair, 2});
        group_results(ipair).area_x = cfg.area_pairs_v5{ipair, 1};
        group_results(ipair).area_y = cfg.area_pairs_v5{ipair, 2};
        group_results(ipair).per_epoch = repmat(empty_res, n_animals, 3);
    end
end
