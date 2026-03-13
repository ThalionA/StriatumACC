%% 1. Configuration and Calculation
window_size = 10; 
corr_window_size = 10; % Window for correlating the areas
n_animals = length(task_data);

% Storage
stability_DMS_all = cell(1, n_animals);
stability_ACC_all = cell(1, n_animals);
stability_DLS_all = cell(1, n_animals); % New DLS storage

coupling_DMS_ACC = cell(1, n_animals);
coupling_DMS_DLS = cell(1, n_animals); % New coupling pair
coupling_ACC_DLS = cell(1, n_animals); % New coupling pair

mean_stability_DMS = nan(n_animals, 1);
mean_stability_ACC = nan(n_animals, 1);
mean_stability_DLS = nan(n_animals, 1);

% Colors
c_dms = [0, 0.4470, 0.7410]; % Blue
c_acc = [0.8500, 0.3250, 0.0980]; % Red/Orange
c_dls = [0.4660, 0.6740, 0.1880]; % Green (using your previous green for DLS)

fprintf('--- Calculating Neural Stability and Inter-Area Coupling ---\n');

for ianimal = 1:n_animals
    n_trials = size(task_data(ianimal).spatial_binned_fr_all, 3);
    
    % --- 1. DMS Calculation ---
    if isfield(task_data(ianimal), 'is_dms') && sum(task_data(ianimal).is_dms) > 1 
        dms_spikes = task_data(ianimal).spatial_binned_fr_all(task_data(ianimal).is_dms, :, :);
        stability_DMS_all{ianimal} = calc_neural_cosine_stability(dms_spikes, window_size);
        mean_stability_DMS(ianimal) = mean(stability_DMS_all{ianimal}, 'omitnan');
    else
        stability_DMS_all{ianimal} = nan(n_trials, 1);
    end
    
    % --- 2. ACC Calculation ---
    if isfield(task_data(ianimal), 'is_acc') && sum(task_data(ianimal).is_acc) > 1
        acc_spikes = task_data(ianimal).spatial_binned_fr_all(task_data(ianimal).is_acc, :, :);
        stability_ACC_all{ianimal} = calc_neural_cosine_stability(acc_spikes, window_size);
        mean_stability_ACC(ianimal) = mean(stability_ACC_all{ianimal}, 'omitnan');
    else
        stability_ACC_all{ianimal} = nan(n_trials, 1);
    end

    % --- 3. DLS Calculation ---
    if isfield(task_data(ianimal), 'is_dls') && sum(task_data(ianimal).is_dls) > 1
        dls_spikes = task_data(ianimal).spatial_binned_fr_all(task_data(ianimal).is_dls, :, :);
        stability_DLS_all{ianimal} = calc_neural_cosine_stability(dls_spikes, window_size);
        mean_stability_DLS(ianimal) = mean(stability_DLS_all{ianimal}, 'omitnan');
    else
        stability_DLS_all{ianimal} = nan(n_trials, 1);
    end

    % --- 4. Sliding Correlation (Coupling) Calculation ---
    half_corr = floor(corr_window_size/2);
    
    % Helper function for coupling
    calc_coupling = @(trace1, trace2) compute_sliding_corr(trace1, trace2, n_trials, half_corr);

    coupling_DMS_ACC{ianimal} = calc_coupling(stability_DMS_all{ianimal}, stability_ACC_all{ianimal});
    coupling_DMS_DLS{ianimal} = calc_coupling(stability_DMS_all{ianimal}, stability_DLS_all{ianimal});
    coupling_ACC_DLS{ianimal} = calc_coupling(stability_ACC_all{ianimal}, stability_DLS_all{ianimal});
end

%% 2. Visualise Relationship: Global Correlations (3 Pairs)
figure('Name', 'Global Stability Interaction', 'Position', [100, 100, 1200, 400]);
t = tiledlayout(1, 3);

% DMS vs ACC
nexttile;
scatter(mean_stability_DMS, mean_stability_ACC, 60, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Mean DMS Stability'); ylabel('Mean ACC Stability');
title('DMS vs ACC'); lsline; axis square; grid on;

% DMS vs DLS
nexttile;
scatter(mean_stability_DMS, mean_stability_DLS, 60, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Mean DMS Stability'); ylabel('Mean DLS Stability');
title('DMS vs DLS'); lsline; axis square; grid on;

% ACC vs DLS
nexttile;
scatter(mean_stability_ACC, mean_stability_DLS, 60, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Mean ACC Stability'); ylabel('Mean DLS Stability');
title('ACC vs DLS'); lsline; axis square; grid on;

%% 3. Individual Animal Dashboards
for ianimal = 1:n_animals
    
    if isempty(task_data(ianimal).spatial_binned_data), continue; end
    
    % Data Prep
    licks = task_data(ianimal).spatial_binned_data.licks ./ task_data(ianimal).spatial_binned_data.durations; 
    [n_trials, n_bins] = size(licks);
    trials_vec = 1:n_trials;
    clim_max = quantile(licks(:), 0.99);
    lp = learning_points_task{ianimal};
    dp = task_data(ianimal).change_point_mean; 
    
    figure('Name', sprintf('Animal %d Analysis', ianimal), 'Position', [100, 100, 1400, 600]);
    % 1x4 Layout: Heatmap (2), Stability (1), Coupling (1)
    t = tiledlayout(1, 4, 'TileSpacing', 'compact');
    
    % --- Panel 1: Lick Heatmap ---
    ax1 = nexttile([1, 2]); 
    imagesc(licks_plot);
    colormap(ax1, 'parula');
    c = colorbar; c.Label.String = 'Lick Rate (Hz)';
    clim([0 clim_max]);
    hold on;
    xline([25, 34], 'w--', 'LineWidth', 1.5); 
    yline(lp, 'r-', 'LineWidth', 2);
    if ~isnan(dp), yline(dp, 'c--', 'LineWidth', 2); end
    hold off;
    ylabel('Trial Number'); xlabel('Spatial Bin');
    title(sprintf('Animal %d: Behavior', ianimal));
    axis tight
    
    % --- Panel 2: Neural Stability Traces ---
    ax2 = nexttile;
    hold on;
    if sum(task_data(ianimal).is_dms) > 1
        plot(stability_DMS_all{ianimal}, trials_vec, 'Color', c_dms, 'LineWidth', 2, 'DisplayName', 'DMS');
    end
    if sum(task_data(ianimal).is_acc) > 1
        plot(stability_ACC_all{ianimal}, trials_vec, 'Color', c_acc, 'LineWidth', 2, 'DisplayName', 'ACC');
    end
    if sum(task_data(ianimal).is_dls) > 1
        plot(stability_DLS_all{ianimal}, trials_vec, 'Color', c_dls, 'LineWidth', 2, 'DisplayName', 'DLS');
    end
    
    set(gca, 'YDir', 'reverse'); ylim([0.5, n_trials + 0.5]);
    yline(lp, 'r-', 'LineWidth', 2);
    if ~isnan(dp), yline(dp, 'c--', 'LineWidth', 2); end
    xlabel('Cosine Stability'); title('Neural Stability');
    grid on; legend('Location', 'best');
    axis tight
    
    % --- Panel 3: Area Coupling ---
    ax3 = nexttile;
    hold on; xline(0, 'k:');
    
    % Plot couplings
    if ~all(isnan(coupling_DMS_ACC{ianimal}))
        plot(coupling_DMS_ACC{ianimal}, trials_vec, 'Color', 'k', 'LineWidth', 2, 'DisplayName', 'DMS-ACC');
    end
    if ~all(isnan(coupling_DMS_DLS{ianimal}))
        plot(coupling_DMS_DLS{ianimal}, trials_vec, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'DMS-DLS');
    end
    
    set(gca, 'YDir', 'reverse'); ylim([0.5, n_trials + 0.5]); xlim([-1 1]);
    yline(lp, 'r-', 'LineWidth', 2);
    if ~isnan(dp), yline(dp, 'c--', 'LineWidth', 2); end
    xlabel('Correlation (r)'); title('Inter-Area Coupling');
    grid on; legend('Location', 'best');
    axis tight
    % linkaxes([ax1, ax2, ax3], 'y');
    sgtitle(sprintf('Animal %d: Behavior vs Neural Dynamics', ianimal));
end

%% 4. Epoch-based Summary Analysis (Pre vs Engaged vs Disengaged)
fprintf('--- Calculating Epoch Averages ---\n');

% Initialize Matrices [Animals x 3 Epochs]
epoch_DMS = nan(n_animals, 3);
epoch_ACC = nan(n_animals, 3);
epoch_DLS = nan(n_animals, 3);

epoch_CPL_DMSACC = nan(n_animals, 3);
epoch_CPL_DMSDLS = nan(n_animals, 3);
epoch_CPL_ACCDLS = nan(n_animals, 3);

for ianimal = 1:n_animals
    lp = learning_points_task{ianimal};
    dp = task_data(ianimal).change_point_mean; 
    
    % Determine max trials from available stability traces
    traces = {stability_DMS_all{ianimal}, stability_ACC_all{ianimal}, stability_DLS_all{ianimal}};
    lens = cellfun(@length, traces);
    if all(lens == 0), continue; end
    n_trials = max(lens);
    
    % --- Define Indices ---
    if lp > 1, idx_pre = 1:(lp-1); else, idx_pre = []; end
    
    if isnan(dp) || dp > n_trials
        idx_eng = lp:n_trials; idx_dis = [];
    else
        idx_eng = lp:(dp-1); idx_dis = dp:n_trials;
    end
    
    extract_mean = @(data, idx) mean(data(idx), 'omitnan');
    
    % --- Extract Means: Neural Stability ---
    if ~all(isnan(stability_DMS_all{ianimal}))
        if ~isempty(idx_pre), epoch_DMS(ianimal, 1) = extract_mean(stability_DMS_all{ianimal}, idx_pre); end
        if ~isempty(idx_eng), epoch_DMS(ianimal, 2) = extract_mean(stability_DMS_all{ianimal}, idx_eng); end
        if ~isempty(idx_dis), epoch_DMS(ianimal, 3) = extract_mean(stability_DMS_all{ianimal}, idx_dis); end
    end
    if ~all(isnan(stability_ACC_all{ianimal}))
        if ~isempty(idx_pre), epoch_ACC(ianimal, 1) = extract_mean(stability_ACC_all{ianimal}, idx_pre); end
        if ~isempty(idx_eng), epoch_ACC(ianimal, 2) = extract_mean(stability_ACC_all{ianimal}, idx_eng); end
        if ~isempty(idx_dis), epoch_ACC(ianimal, 3) = extract_mean(stability_ACC_all{ianimal}, idx_dis); end
    end
    if ~all(isnan(stability_DLS_all{ianimal}))
        if ~isempty(idx_pre), epoch_DLS(ianimal, 1) = extract_mean(stability_DLS_all{ianimal}, idx_pre); end
        if ~isempty(idx_eng), epoch_DLS(ianimal, 2) = extract_mean(stability_DLS_all{ianimal}, idx_eng); end
        if ~isempty(idx_dis), epoch_DLS(ianimal, 3) = extract_mean(stability_DLS_all{ianimal}, idx_dis); end
    end
    
    % --- Extract Means: Coupling ---
    % 1. DMS-ACC
    if ~all(isnan(coupling_DMS_ACC{ianimal}))
        if ~isempty(idx_pre), epoch_CPL_DMSACC(ianimal, 1) = extract_mean(coupling_DMS_ACC{ianimal}, idx_pre); end
        if ~isempty(idx_eng), epoch_CPL_DMSACC(ianimal, 2) = extract_mean(coupling_DMS_ACC{ianimal}, idx_eng); end
        if ~isempty(idx_dis), epoch_CPL_DMSACC(ianimal, 3) = extract_mean(coupling_DMS_ACC{ianimal}, idx_dis); end
    end
    % 2. DMS-DLS
    if ~all(isnan(coupling_DMS_DLS{ianimal}))
        if ~isempty(idx_pre), epoch_CPL_DMSDLS(ianimal, 1) = extract_mean(coupling_DMS_DLS{ianimal}, idx_pre); end
        if ~isempty(idx_eng), epoch_CPL_DMSDLS(ianimal, 2) = extract_mean(coupling_DMS_DLS{ianimal}, idx_eng); end
        if ~isempty(idx_dis), epoch_CPL_DMSDLS(ianimal, 3) = extract_mean(coupling_DMS_DLS{ianimal}, idx_dis); end
    end
    % 3. ACC-DLS
    if ~all(isnan(coupling_ACC_DLS{ianimal}))
        if ~isempty(idx_pre), epoch_CPL_ACCDLS(ianimal, 1) = extract_mean(coupling_ACC_DLS{ianimal}, idx_pre); end
        if ~isempty(idx_eng), epoch_CPL_ACCDLS(ianimal, 2) = extract_mean(coupling_ACC_DLS{ianimal}, idx_eng); end
        if ~isempty(idx_dis), epoch_CPL_ACCDLS(ianimal, 3) = extract_mean(coupling_ACC_DLS{ianimal}, idx_dis); end
    end
end

% --- Visualization ---
figure('Name', 'Epoch Summary: Stability & All Couplings', 'Position', [100, 100, 1200, 800]);
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Row 1: Stability
nexttile; plot_epoch_data(gca, epoch_DMS, 'DMS Stability', 'Cosine Sim', c_dms);
nexttile; plot_epoch_data(gca, epoch_ACC, 'ACC Stability', 'Cosine Sim', c_acc);
nexttile; plot_epoch_data(gca, epoch_DLS, 'DLS Stability', 'Cosine Sim', c_dls);

% Row 2: Coupling
c_gray = [0.4 0.4 0.4];
nexttile; plot_epoch_data(gca, epoch_CPL_DMSACC, 'DMS-ACC Coupling', 'Corr (r)', c_gray);
nexttile; plot_epoch_data(gca, epoch_CPL_DMSDLS, 'DMS-DLS Coupling', 'Corr (r)', c_gray);
nexttile; plot_epoch_data(gca, epoch_CPL_ACCDLS, 'ACC-DLS Coupling', 'Corr (r)', c_gray);

sgtitle('Neural Dynamics across Learning Epochs');

%% 5. Behavioral Stability & Neuro-Behavioral Coupling
fprintf('--- Calculating Behavioral Stability and Neuro-Behavioral Coupling ---\n');

% Storage
stability_Lick_all = cell(1, n_animals);
stability_Vel_all  = cell(1, n_animals);

% Couplings (Neural Stability vs Behavioral Stability)
coupling_DMS_Lick = cell(1, n_animals); coupling_DMS_Vel = cell(1, n_animals);
coupling_ACC_Lick = cell(1, n_animals); coupling_ACC_Vel = cell(1, n_animals);
coupling_DLS_Lick = cell(1, n_animals); coupling_DLS_Vel = cell(1, n_animals);

for ianimal = 1:n_animals
    if isempty(task_data(ianimal).spatial_binned_data), continue; end

    % --- 1. Behavioral Stability Calculation ---
    % Lick Rate (Licks / Duration)
    licks = task_data(ianimal).spatial_binned_data.licks;
    durs  = task_data(ianimal).spatial_binned_data.durations;
    lick_rate = licks ./ durs;
    lick_rate(isnan(lick_rate) | isinf(lick_rate)) = 0; % Sanitize
    
    % Velocity (1 / Duration)
    % Note: If duration is 0 (rare), vel is Inf. Handle this.
    vel = 1 ./ durs;
    vel(isinf(vel)) = nan; 

    % Calculate Stability (Sliding Window Pairwise Correlation)
    stability_Lick_all{ianimal} = calc_sliding_reliability(lick_rate, window_size);
    stability_Vel_all{ianimal}  = calc_sliding_reliability(vel, window_size);
    
    % --- 2. Neuro-Behavioral Coupling ---
    % Correlate Neural Stability Trace with Behavioral Stability Trace
    n_trials = length(stability_Lick_all{ianimal});
    half_corr = floor(corr_window_size/2);
    calc_coup = @(n_trace, b_trace) compute_sliding_corr(n_trace, b_trace, n_trials, half_corr);
    
    % DMS vs Behavior
    if ~all(isnan(stability_DMS_all{ianimal}))
        coupling_DMS_Lick{ianimal} = calc_coup(stability_DMS_all{ianimal}, stability_Lick_all{ianimal});
        coupling_DMS_Vel{ianimal}  = calc_coup(stability_DMS_all{ianimal}, stability_Vel_all{ianimal});
    end
    
    % ACC vs Behavior
    if ~all(isnan(stability_ACC_all{ianimal}))
        coupling_ACC_Lick{ianimal} = calc_coup(stability_ACC_all{ianimal}, stability_Lick_all{ianimal});
        coupling_ACC_Vel{ianimal}  = calc_coup(stability_ACC_all{ianimal}, stability_Vel_all{ianimal});
    end
    
    % DLS vs Behavior
    if ~all(isnan(stability_DLS_all{ianimal}))
        coupling_DLS_Lick{ianimal} = calc_coup(stability_DLS_all{ianimal}, stability_Lick_all{ianimal});
        coupling_DLS_Vel{ianimal}  = calc_coup(stability_DLS_all{ianimal}, stability_Vel_all{ianimal});
    end
end

% --- Visualization A: Individual Animal Neuro-Behavioral Dynamics ---
for ianimal = 1:n_animals
    if isempty(stability_Lick_all{ianimal}), continue; end
    
    n_trials = length(stability_Lick_all{ianimal});
    trials_vec = 1:n_trials;
    lp = learning_points_task{ianimal};
    
    figure('Name', sprintf('Animal %d: Neuro-Behavioral Coupling', ianimal), 'Position', [100, 100, 1200, 500]);
    t = tiledlayout(1, 3, 'TileSpacing', 'compact');
    
    % Panel 1: Behavioral Stability Traces
    nexttile; hold on;
    plot(stability_Lick_all{ianimal}, trials_vec, 'k', 'LineWidth', 2, 'DisplayName', 'Lick Stability');
    plot(stability_Vel_all{ianimal}, trials_vec, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'DisplayName', 'Vel Stability');
    set(gca, 'YDir', 'reverse'); ylim([0.5, n_trials+0.5]);
    yline(lp, 'r-', 'LineWidth', 2);
    title('Behavioral Stability'); xlabel('Consistency (r)'); legend('Location', 'best'); grid on;
    
    % Panel 2: Neural-Lick Coupling
    nexttile; hold on; xline(0, 'k:');
    if ~isempty(coupling_DMS_Lick{ianimal}), plot(coupling_DMS_Lick{ianimal}, trials_vec, 'Color', c_dms, 'LineWidth', 2, 'DisplayName', 'DMS-Lick'); end
    if ~isempty(coupling_ACC_Lick{ianimal}), plot(coupling_ACC_Lick{ianimal}, trials_vec, 'Color', c_acc, 'LineWidth', 2, 'DisplayName', 'ACC-Lick'); end
    if ~isempty(coupling_DLS_Lick{ianimal}), plot(coupling_DLS_Lick{ianimal}, trials_vec, 'Color', c_dls, 'LineWidth', 2, 'DisplayName', 'DLS-Lick'); end
    set(gca, 'YDir', 'reverse'); ylim([0.5, n_trials+0.5]); xlim([-1 1]);
    yline(lp, 'r-', 'LineWidth', 2);
    title('Neural-Lick Coupling'); xlabel('Correlation (r)'); legend('Location', 'best'); grid on;
    
    % Panel 3: Neural-Velocity Coupling
    nexttile; hold on; xline(0, 'k:');
    if ~isempty(coupling_DMS_Vel{ianimal}), plot(coupling_DMS_Vel{ianimal}, trials_vec, 'Color', c_dms, 'LineWidth', 2, 'DisplayName', 'DMS-Vel'); end
    if ~isempty(coupling_ACC_Vel{ianimal}), plot(coupling_ACC_Vel{ianimal}, trials_vec, 'Color', c_acc, 'LineWidth', 2, 'DisplayName', 'ACC-Vel'); end
    if ~isempty(coupling_DLS_Vel{ianimal}), plot(coupling_DLS_Vel{ianimal}, trials_vec, 'Color', c_dls, 'LineWidth', 2, 'DisplayName', 'DLS-Vel'); end
    set(gca, 'YDir', 'reverse'); ylim([0.5, n_trials+0.5]); xlim([-1 1]);
    yline(lp, 'r-', 'LineWidth', 2);
    title('Neural-Vel Coupling'); xlabel('Correlation (r)'); legend('Location', 'best'); grid on;
    
    sgtitle(sprintf('Animal %d: Stability Coupling Analysis', ianimal));
end

% --- Visualization B: Population Summary of Couplings ---
% Calculate mean coupling per animal (ignoring NaNs)
mean_CPL_DMS_Lick = cellfun(@(x) mean(x, 'omitnan'), coupling_DMS_Lick);
mean_CPL_ACC_Lick = cellfun(@(x) mean(x, 'omitnan'), coupling_ACC_Lick);
mean_CPL_DLS_Lick = cellfun(@(x) mean(x, 'omitnan'), coupling_DLS_Lick);

figure('Name', 'Population Neuro-Behavioral Coupling', 'Position', [100, 100, 600, 450]);
bar_data = [mean(mean_CPL_DMS_Lick, 'omitnan'), mean(mean_CPL_ACC_Lick, 'omitnan'), mean(mean_CPL_DLS_Lick, 'omitnan')];
bar_sem = [std(mean_CPL_DMS_Lick, 'omitnan'), std(mean_CPL_ACC_Lick, 'omitnan'), std(mean_CPL_DLS_Lick, 'omitnan')] ./ sqrt(n_animals);

b = bar(1:3, bar_data);
b.FaceColor = 'flat';
b.CData(1,:) = c_dms; b.CData(2,:) = c_acc; b.CData(3,:) = c_dls;
hold on;
errorbar(1:3, bar_data, bar_sem, 'k.', 'LineWidth', 2);
xticks(1:3); xticklabels({'DMS-Lick', 'ACC-Lick', 'DLS-Lick'});
ylabel('Mean Coupling (Correlation)');
title('Which Area Tracks Behavioral Stability Best?');
grid on; box off;

%% 6. Metric Comparison: Cosine vs Pearson Correlation Stability
fprintf('--- Calculating Correlation-based Stability & Comparing Metrics ---\n');

% Storage for the new metric
stability_DMS_corr = cell(1, n_animals);
stability_ACC_corr = cell(1, n_animals);
stability_DLS_corr = cell(1, n_animals);

for ianimal = 1:n_animals
    n_trials = size(task_data(ianimal).spatial_binned_fr_all, 3);
    
    % --- 1. DMS ---
    if isfield(task_data(ianimal), 'is_dms') && sum(task_data(ianimal).is_dms) > 1 
        dms_spikes = task_data(ianimal).spatial_binned_fr_all(task_data(ianimal).is_dms, :, :);
        % Calculate NEW correlation metric
        stability_DMS_corr{ianimal} = calc_neural_corr_stability(dms_spikes, window_size);
    else
        stability_DMS_corr{ianimal} = nan(n_trials, 1);
    end
    
    % --- 2. ACC ---
    if isfield(task_data(ianimal), 'is_acc') && sum(task_data(ianimal).is_acc) > 1
        acc_spikes = task_data(ianimal).spatial_binned_fr_all(task_data(ianimal).is_acc, :, :);
        stability_ACC_corr{ianimal} = calc_neural_corr_stability(acc_spikes, window_size);
    else
        stability_ACC_corr{ianimal} = nan(n_trials, 1);
    end

    % --- 3. DLS ---
    if isfield(task_data(ianimal), 'is_dls') && sum(task_data(ianimal).is_dls) > 1
        dls_spikes = task_data(ianimal).spatial_binned_fr_all(task_data(ianimal).is_dls, :, :);
        stability_DLS_corr{ianimal} = calc_neural_corr_stability(dls_spikes, window_size);
    else
        stability_DLS_corr{ianimal} = nan(n_trials, 1);
    end
end

% --- Visualization: Metric Relationship ---
figure('Name', 'Stability Metrics Comparison', 'Position', [100, 100, 1200, 400]);
t = tiledlayout(1, 3, 'TileSpacing', 'compact');

% Prepare data for scatter plots (concatenate all animals/trials)
plot_metric_scatter(nexttile, stability_DMS_all, stability_DMS_corr, 'DMS', c_dms);
plot_metric_scatter(nexttile, stability_ACC_all, stability_ACC_corr, 'ACC', c_acc);
plot_metric_scatter(nexttile, stability_DLS_all, stability_DLS_corr, 'DLS', c_dls);

sgtitle('Cosine Similarity (Direction) vs Pearson Correlation (Pattern)');

% --- Visualization: Representative Trace Comparison ---
% Pick a representative animal (e.g., first one with data)
rep_animal = 10;
if ~isempty(rep_animal)
    figure('Name', sprintf('Metric Traces: Animal %d', rep_animal), 'Position', [100, 100, 1000, 400]);
    t2 = tiledlayout(1, 3, 'TileSpacing', 'compact');
    
    plot_metric_traces(nexttile, stability_DMS_all{rep_animal}, stability_DMS_corr{rep_animal}, 'DMS', c_dms);
    plot_metric_traces(nexttile, stability_ACC_all{rep_animal}, stability_ACC_corr{rep_animal}, 'ACC', c_acc);
    plot_metric_traces(nexttile, stability_DLS_all{rep_animal}, stability_DLS_corr{rep_animal}, 'DLS', c_dls);
    
    sgtitle(sprintf('Trial-by-Trial Metric Comparison (Animal %d)', rep_animal));
end

%% Helper Functions for this Section

function plot_metric_scatter(ax, cos_data, corr_data, area_name, col)
    hold(ax, 'on');
    all_cos = vertcat(cos_data{:});
    all_corr = vertcat(corr_data{:});
    
    % Remove NaNs
    valid = ~isnan(all_cos) & ~isnan(all_corr);
    
    scatter(ax, all_cos(valid), all_corr(valid), 15, col, 'filled', 'MarkerFaceAlpha', 0.3);
    
    % Identity Line
    plot(ax, [0 1], [0 1], 'k--', 'LineWidth', 1.5);
    
    xlabel(ax, 'Cosine Similarity');
    ylabel(ax, 'Pearson Correlation');
    title(ax, area_name);
    xlim(ax, [-0.2 1]); ylim(ax, [-0.2 1]);
    axis(ax, 'square'); grid(ax, 'on');
    
    [r, p] = corr(all_cos(valid), all_corr(valid));
    text(ax, 0.05, 0.9, sprintf('r = %.2f', r), 'Units', 'normalized', 'FontWeight', 'bold');
end

function plot_metric_traces(ax, cos_trace, corr_trace, area_name, col)
    hold(ax, 'on');
    trials = 1:length(cos_trace);
    
    plot(ax, trials, cos_trace, '-', 'Color', col, 'LineWidth', 2, 'DisplayName', 'Cosine');
    plot(ax, trials, corr_trace, '--', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5, 'DisplayName', 'Correlation');
    
    title(ax, area_name);
    xlabel(ax, 'Trial'); ylabel(ax, 'Stability');
    ylim(ax, [-0.1 1]);
    legend(ax, 'Location', 'best');
    grid(ax, 'on');
end

%% Helper Functions

function plot_epoch_data(ax, data, title_str, y_label, col)
    hold(ax, 'on');
    plot(ax, 1:3, data', '-o', 'Color', [0.7 0.7 0.7, 0.4], 'MarkerSize', 4, 'MarkerFaceColor', 'none', 'LineWidth', 1);
    
    grp_mean = mean(data, 1, 'omitnan');
    grp_sem = std(data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(data)));
    
    errorbar(ax, 1:3, grp_mean, grp_sem, '-o', 'Color', col, 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', col, 'CapSize', 0);
    
    set(ax, 'XTick', 1:3, 'XTickLabel', {'Naive', 'Engaged', 'Disengaged'});
    xlim(ax, [0.5 3.5]); ylabel(ax, y_label); title(ax, title_str);
    grid(ax, 'on'); box(ax, 'off');
    
    % T-test Naive vs Engaged
    [~, p_ne] = ttest(data(:,1), data(:,2));
    if ~isnan(p_ne) && p_ne < 0.05
        yl = ylim(ax); y_star = yl(2) * 0.95;
        line(ax, [1, 2], [y_star, y_star], 'Color', 'k', 'LineWidth', 1.5);
        text(ax, 1.5, y_star*1.01, '*', 'HorizontalAlignment', 'center', 'FontSize', 14);
    end
    
    % T-test Engaged vs Disengaged
    [~, p_ed] = ttest(data(:,2), data(:,3));
    if ~isnan(p_ed) && p_ed < 0.05
        yl = ylim(ax); y_star = yl(2) * 0.95; 
        if ~isnan(p_ne) && p_ne < 0.05, y_star = y_star * 0.9; end
        line(ax, [2, 3], [y_star, y_star], 'Color', 'k', 'LineWidth', 1.5);
        text(ax, 2.5, y_star*1.01, '*', 'HorizontalAlignment', 'center', 'FontSize', 14);
    end
end

function trace = compute_sliding_corr(trace1, trace2, n_requested, half_corr)
    % Robustly calculates sliding correlation between two traces
    % Handles mismatched lengths by truncating to the shorter trace.
    
    % 1. Determine common length
    L1 = length(trace1);
    L2 = length(trace2);
    n_common = min(L1, L2);
    
    % 2. If requested trials exceed available data, warn and truncate
    if n_requested > n_common
        % Optional: fprintf('Warning: Mismatch in trace lengths (%d vs %d). Truncating to %d.\n', L1, L2, n_common);
        n_calc = n_common;
    else
        n_calc = n_requested;
    end
    
    % 3. Initialize output
    trace = nan(n_calc, 1);
    
    % 4. Run sliding correlation
    if ~all(isnan(trace1)) && ~all(isnan(trace2))
        for t = 1:n_calc
            % Define window indices
            t_start = max(1, t - half_corr);
            t_end = min(n_calc, t + half_corr); % Ensure window doesn't go beyond n_calc
            idx = t_start:t_end;
            
            % Compute correlation if we have enough points
            if length(idx) > 3
                % Because we capped n_calc at min(L1, L2), idx is safe for both
                trace(t) = corr(trace1(idx), trace2(idx), 'Rows', 'complete');
            end
        end
    end
end

function rel = calc_sliding_reliability(data, win)
    % Calculates average pairwise correlation between trials in a sliding window
    % data: [n_trials x n_bins]
    
    [n_trials, ~] = size(data);
    rel = nan(n_trials, 1);
    half = floor(win/2);
    
    for t = 1:n_trials
        ts = max(1, t-half); 
        te = min(n_trials, t+half);
        
        if te - ts < 1, continue; end % Need at least 2 trials
        
        chunk = data(ts:te, :);
        
        % Check for empty rows (all NaNs or zeros) to avoid correlation errors
        valid_rows = any(chunk ~= 0 & ~isnan(chunk), 2);
        if sum(valid_rows) < 2, continue; end
        chunk = chunk(valid_rows, :);
        
        % Pairwise correlation of rows (Transpose so trials are columns)
        R = corr(chunk', 'rows', 'pairwise'); 
        
        % Extract upper triangle unique pairs
        vals = R(triu(true(size(R)), 1));
        rel(t) = mean(vals, 'omitnan');
    end
end

function [neural_stability] = calc_neural_corr_stability(spike_tensor, window_size)
    % Calculates stability using Pearson correlation of flattened population vectors
    % spike_tensor: [n_neurons x n_bins x n_trials]
    
    [n_neurons, n_bins, n_trials] = size(spike_tensor);
    neural_stability = nan(n_trials, 1);
    half_win = floor(window_size/2);

    % Flatten: [features x trials]
    % Features = neurons * spatial_bins
    flat_data = reshape(permute(spike_tensor, [2 1 3]), [n_bins*n_neurons, n_trials]);
    
    for t = 1:n_trials
        t_start = max(1, t - half_win);
        t_end = min(n_trials, t + half_win);
        idx = t_start:t_end;
        
        if length(idx) < 2, continue; end
        
        current_vectors = flat_data(:, idx);
        
        % Remove features that are zero/NaN across the whole window to avoid NaN corr
        valid_features = any(current_vectors ~= 0 & ~isnan(current_vectors), 2);
        if sum(valid_features) < 2
            continue; 
        end
        current_vectors = current_vectors(valid_features, :);
        
        % Pearson Correlation Matrix
        R = corr(current_vectors, 'Rows', 'complete');
        
        % Extract off-diagonal unique pairs
        mask = triu(true(size(R)), 1);
        unique_vals = R(mask);
        
        neural_stability(t) = mean(unique_vals, 'omitnan');
    end
end