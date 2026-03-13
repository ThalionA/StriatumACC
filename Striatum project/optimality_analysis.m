%% ================= Behavioral Optimality Analysis =================
% This section calculates the key behavioral parameters needed to model and
% test the optimality of the animals' licking strategy.

fprintf('--- Analyzing Behavioral Strategy for Optimality ---\n');

% --- Get Key Parameters directly from your config file ---
bin_size_au = cfg.plot.zone_params.bin_size; % Size of spatial bins in arbitrary units (AU)
reward_zone_au = cfg.plot.zone_params.reward_zone_au; % Start and end of RZ

% Convert reward zone from AU to bin indices
rz_bins = floor(reward_zone_au(1)/bin_size_au) : floor(reward_zone_au(2)/bin_size_au);

n_animals = length(task_data);
optimality_results = struct(); % To store results

for ianimal = 1:n_animals
    fprintf('Processing Animal %d for optimality metrics...\n', ianimal);

    % --- Aggregate trial data to get overall behavioral statistics ---

    total_licks_per_bin = task_data(ianimal).spatial_binned_data.licks;
    total_duration_per_bin = task_data(ianimal).spatial_binned_data.durations;

    % Quantify the Time Cost of Licking (Velocity Analysis)

    valid_bins = total_duration_per_bin > 0;

    % Speed in AU per second for each bin
    speed_au_per_s = nan(size(total_duration_per_bin));
    speed_au_per_s(valid_bins) = (1 ./ total_duration_per_bin(valid_bins)) * bin_size_au; % speed = distance/time

    % Lick rate in licks per second for each bin
    lick_rate_per_s = nan(size(total_duration_per_bin));
    lick_rate_per_s(valid_bins) = total_licks_per_bin(valid_bins) ./ total_duration_per_bin(valid_bins);

    % Separate bins by licking activity
    non_licking_bins = (lick_rate_per_s == 0) & valid_bins;
    licking_bins = (lick_rate_per_s > 0) & valid_bins;

    % Calculate average speeds
    baseline_speed = mean(speed_au_per_s(non_licking_bins), 'omitnan');
    licking_speed = mean(speed_au_per_s(licking_bins), 'omitnan');
    slowdown_factor = licking_speed / baseline_speed;

    fprintf('  Baseline Speed: %.2f AU/s\n', baseline_speed);
    fprintf('  Licking Speed:  %.2f AU/s\n', licking_speed);
    fprintf('  Slowdown Factor: %.2f\n', slowdown_factor);

    optimality_results(ianimal).baseline_speed_au_s = baseline_speed;
    optimality_results(ianimal).slowdown_factor = slowdown_factor;

    % --- Calculate Mean Lick Rate for Expert Trials ---
    % This normalizes for the time spent in each bin, which is more rigorous
    % than total counts. This assumes you have a 'durations_by_trial' field.
    expert_trials = learning_points_task{ianimal}+1:learning_points_task{ianimal}+10;
    expert_licks = sum(task_data(ianimal).spatial_binned_data.licks(expert_trials, :), 1, 'omitnan');
    expert_durations = sum(task_data(ianimal).spatial_binned_data.durations(expert_trials, :), 1, 'omitnan');

    expert_lick_rate = expert_licks ./ expert_durations;
    expert_lick_rate(isinf(expert_lick_rate) | isnan(expert_lick_rate)) = 0; % Clean up NaNs/Infs

    % Estimate Positional Uncertainty (sigma_p)

    num_bins = length(total_licks_per_bin);

    % Define an analysis window around the reward zone based on cfg
    analysis_window = (min(rz_bins)-50/bin_size_au) : (max(rz_bins)+20/bin_size_au);
    analysis_window = round(analysis_window); % Ensure integer bins
    analysis_window = analysis_window(analysis_window > 0 & analysis_window <= num_bins);

    licks_in_window = sum(total_licks_per_bin(learning_points_task{ianimal}+1:learning_points_task{ianimal}+10, analysis_window), 'omitmissing');

    % Normalize to a probability distribution
    lick_distribution = licks_in_window / sum(licks_in_window);

    % Calculate weighted mean and standard deviation
    window_bin_indices = 1:length(analysis_window);
    mean_lick_bin_in_window = sum(lick_distribution .* window_bin_indices);
    variance = sum(((window_bin_indices - mean_lick_bin_in_window).^2) .* lick_distribution);
    sigma_p_bins = sqrt(variance);
    sigma_p_au = sigma_p_bins * bin_size_au; % Convert to same units as track

    % Alternative calculation of sigma
    % Define a "Commitment Threshold" ---
    % A simple threshold could be a fraction of the max rate in the expert phase,
    % or a more robust measure like mean+std of non-zero lick rates.
    licks_expert = task_data(ianimal).spatial_binned_data.licks(expert_trials, :);
    durations_expert = task_data(ianimal).spatial_binned_data.durations(expert_trials, :);
    lick_rate_expert = licks_expert ./ durations_expert;

    all_expert_rates = lick_rate_expert(:);
    lick_rate_threshold = quantile(all_expert_rates(all_expert_rates>0.01), 0.5); % e.g., 50% of the max lick rate


    commitment_bins = nan(1, length(expert_trials));
    for i_trial = 1:length(expert_trials)
        % Find the first bin in the analysis window where lick rate crosses threshold
        first_lick_bin = find(lick_rate_expert(i_trial, analysis_window) > lick_rate_threshold, 1, 'first');
        
        if ~isempty(first_lick_bin)
            % Convert window index back to absolute track bin index
            commitment_bins(i_trial) = analysis_window(first_lick_bin);
        end
    end

    % Calculate Sigma as the Standard Deviation of commitment bins ---
    if sum(~isnan(commitment_bins)) < 2 % Need at least 2 data points to calculate std
        warning('Animal %d has fewer than 2 valid commitment trials. Cannot calculate sigma_p.', ianimal);
        sigma_p_bins = nan;
    else
        sigma_p_bins = std(commitment_bins, 'omitnan');
    end
    sigma_p_au = sigma_p_bins * bin_size_au; % Convert to same units as track


    fprintf('  Positional Uncertainty (sigma_p): %.2f bins (%.2f AU)\n', sigma_p_bins, sigma_p_au);

    optimality_results(ianimal).sigma_p_bins = sigma_p_bins;
    optimality_results(ianimal).sigma_p_au = sigma_p_au;

    % Revised Plot 1: Box Plot of Velocities

    % This plot provides a statistically robust comparison of running speeds in
    % spatial bins with and without licking activity.

    figure('Name', sprintf('Animal %d: Velocity Comparison', ianimal), 'Position', [100, 100, 500, 600]);

    % Extract the speed data for the two groups
    speeds_no_licks = speed_au_per_s(non_licking_bins);
    speeds_with_licks = speed_au_per_s(licking_bins);

    % To use the boxplot function, we create a single data vector and a
    % corresponding grouping variable.
    combined_speeds = [speeds_no_licks(:); speeds_with_licks(:)];
    group_labels = [repmat({'No Licks'}, numel(speeds_no_licks), 1); ...
        repmat({'Licks'}, numel(speeds_with_licks), 1)];

    % Create the box plot
    boxplot(combined_speeds, group_labels, 'Notch', 'on', 'Colors', [0 0.4470 0.7410; 0.8500 0.3250 0.0980]);

    % --- Add a statistical test ---
    % A Wilcoxon rank-sum test is appropriate here as we can't assume normality.
    % This tests if the two groups have equal medians.
    p_val = ranksum(speeds_no_licks, speeds_with_licks);

    % Formatting and Labels
    ylabel('Running Speed (AU/s)');
    title(sprintf('Animal %d: Velocity Comparison (p = %.2e)', ianimal, p_val));
    grid on;
    box on;

    % Add the p-value text to the plot
    ax = gca;
    text(1.5, max(ax.YLim)*0.9, sprintf('p = %.2e', p_val), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');


    % Revised Plot 2: Line Plot of Lick Rate Distribution

    % This plot shows the mean lick rate across the track as a clean line plot,
    % overlaid with the Gaussian fit representing positional uncertainty.

    figure('Name', sprintf('Animal %d: Lick Rate Distribution', ianimal), 'Position', [650, 100, 900, 500]);
    hold on;

    % Get track geometry from cfg
    vz_au = cfg.plot.zone_params.visual_zones_au;
    rz_au = cfg.plot.zone_params.reward_zone_au;
    track_end_au = cfg.plot.zone_params.corridor_end_au;
    track_bins = 1:(track_end_au / bin_size_au);
    track_au_positions = track_bins * bin_size_au;

    % Set Y-axis limit BEFORE drawing patches
    ylim_top = max(expert_lick_rate) * 1.15;
    if ylim_top == 0, ylim_top = 1; end
    ylim([0 ylim_top]);

    % --- Shade the visual and reward zones for context ---
    patch([vz_au(1) vz_au(2) vz_au(2) vz_au(1)], [0 0 ylim_top ylim_top], ...
        [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Visual Zone');
    patch([rz_au(1) rz_au(2) rz_au(2) rz_au(1)], [0 0 ylim_top ylim_top], ...
        cfg.plot.colors.dls, 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'Reward Zone');

    % --- Plot the mean lick rate as a line plot ---
    p_lickrate = plot(track_au_positions, expert_lick_rate(track_bins), ...
        'Color', cfg.plot.colors.dms, 'LineWidth', 2);
    p_lickrate.DisplayName = 'Mean Lick Rate (Expert)';

    % --- Overlay the calculated Gaussian fit for positional uncertainty ---
    % The sigma_p calculation itself is based on the spatial probability of
    % licks, so it remains valid. We just scale its visualization.
    if isfield(optimality_results(ianimal), 'sigma_p_au') && ~isnan(optimality_results(ianimal).sigma_p_au)
        mean_lick_pos_au = (mean_lick_bin_in_window-1) * bin_size_au + analysis_window(1) * bin_size_au;
        gaussian_fit = normpdf(track_au_positions, mean_lick_pos_au, optimality_results(ianimal).sigma_p_au);

        % Scale the PDF to match the height of the lick rate plot
        scaled_gaussian_fit = gaussian_fit * (max(expert_lick_rate) / max(gaussian_fit));

        p_gauss = plot(track_au_positions, scaled_gaussian_fit, 'Color', cfg.plot.colors.acc, 'LineWidth', 2.5, 'LineStyle', '--');
        p_gauss.DisplayName = sprintf('Gaussian Fit (\\sigma_p = %.1f AU)', optimality_results(ianimal).sigma_p_au);
    end

    % Formatting and Labels
    xlabel('Position on Track (AU)');
    ylabel('Mean Lick Rate (Hz)');
    title(sprintf('Animal %d: Expert Lick Rate Distribution', ianimal));
    legend('show', 'Location', 'northeast');
    xlim([0 track_end_au]);
    grid on;
    box on;
    hold off;

end

%% ================= Optimality Modeling and Visualization (with Learning Trajectory) =================
fprintf('--- Generating Optimality Landscape with Learning Trajectories ---\n');

fprintf('--- Phase 1: Simulating strategy space for all animals ---\n');

% We will first compute the reward map for every animal and store it.
for ianimal = 1:n_animals
    % --- Setup parameters for the simulation ---
    animal_params.baseline_speed = optimality_results(ianimal).baseline_speed_au_s;
    animal_params.slowdown_factor = optimality_results(ianimal).slowdown_factor;
    animal_params.sigma_p_bins = optimality_results(ianimal).sigma_p_bins;
    
    if isnan(animal_params.sigma_p_bins)
        warning('Skipping simulation for Animal %d due to invalid sigma_p.', ianimal);
        optimality_results(ianimal).reward_rate_map = []; % Mark as empty
        continue;
    end
    
    num_bins = floor(cfg.plot.zone_params.corridor_end_au / cfg.plot.zone_params.bin_size);
    track_params.num_bins = num_bins;
    track_params.bin_size_au = cfg.plot.zone_params.bin_size;
    track_params.rz_bins = rz_bins;
    
    start_bins_to_test = 1:num_bins;
    stop_bins_to_test = 1:num_bins;
    reward_rate_map = nan(length(stop_bins_to_test), length(start_bins_to_test));
    
    % --- Run the simulation for every possible strategy ---
    for s_idx = 1:length(start_bins_to_test)
        for e_idx = 1:length(stop_bins_to_test)
            start_bin = start_bins_to_test(s_idx);
            stop_bin = stop_bins_to_test(e_idx);
            if stop_bin < start_bin, continue; end
            
            rate = calculate_reward_rate(start_bin, stop_bin, animal_params, track_params, animal_params.sigma_p_bins);
            reward_rate_map(stop_bin, start_bin) = rate;
        end
    end
    
    % Store the computed map
    optimality_results(ianimal).reward_rate_map = reward_rate_map;
    
    % --- Determine animal's actual median policy and store it ---
    expert_trials = learning_points_task{ianimal} + 1: (learning_points_task{ianimal} + 10);
    expert_trials = expert_trials(expert_trials <= size(task_data(ianimal).spatial_binned_data.licks, 1));
    expert_licks_by_trial = task_data(ianimal).spatial_binned_data.licks(expert_trials, :);
    expert_durations_by_trial = task_data(ianimal).spatial_binned_data.durations(expert_trials, :);
    expert_lick_rate_by_trial = expert_licks_by_trial ./ expert_durations_by_trial;
    expert_lick_rate_by_trial(isnan(expert_lick_rate_by_trial)|isinf(expert_lick_rate_by_trial)) = 0;
    
    all_rates = expert_lick_rate_by_trial(:);
    lick_rate_threshold = prctile(all_rates(all_rates > 0), 95) * 0.33;
    actual_lick_bins = expert_lick_rate_by_trial >= lick_rate_threshold;
    
    actual_start_bin = nan(1, size(expert_lick_rate_by_trial,1));
    actual_stop_bin = nan(1, size(expert_lick_rate_by_trial,1));
    
    for itrial = 1:size(actual_lick_bins, 1)
        bins_this_trial = find(actual_lick_bins(itrial, :));
        if ~isempty(bins_this_trial)
            actual_start_bin(itrial) = min(bins_this_trial);
            actual_stop_bin(itrial) = max(bins_this_trial);
        end
    end
    
    optimality_results(ianimal).actual_start_bin = round(median(actual_start_bin, 'omitnan'));
    optimality_results(ianimal).actual_stop_bin = round(median(actual_stop_bin, 'omitnan'));
end

% --- Create the figure and tiled layout ---
figure('Position', [100, 100, 1400, 900]); % A larger figure window
t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Determine the global color scale across all valid maps ---
% (This logic is moved from your previous script version to before the plotting loop)
all_maps_stacked = cat(3, optimality_results.reward_rate_map);
global_min = min(all_maps_stacked(:));
global_max = max(all_maps_stacked(:));

% --- Loop through each animal to create their subplot ---
for ianimal = 1:n_animals
    
    reward_rate_map = optimality_results(ianimal).reward_rate_map;
    if isempty(reward_rate_map)
        continue;
    end
    
    % --- Go to the next tile and plot the landscape background ---
    nexttile;
    imagesc(reward_rate_map);
    set(gca, 'YDir', 'normal');
    clim([global_min, global_max]);
    hold on;
    
    % =====================================================================
    % --- NEW: Calculate and Plot Learning Trajectory for Each Epoch ---
    % =====================================================================
    
    % --- 1. Define Trial Epochs ---
    lp = learning_points_task{ianimal};

    % Naive: First 10 trials
    naive_trials = 1:10;

    % Intermediate: 10 trials before the learning point
    intermediate_trials = (lp - 10):(lp - 1);
    
    % Expert: 10 trials after the learning point
    expert_trials = lp+1:(lp + 10);
    
    epoch_trials = {naive_trials, intermediate_trials, expert_trials};
    epoch_names = {'Naive', 'Intermediate', 'Expert'};
    epoch_policies = nan(3, 2); % To store [start, stop] for each epoch
    
    % --- 2. Calculate Median Policy for Each Epoch ---
    for i_epoch = 1:3
        current_trials = epoch_trials{i_epoch};
        
        if isempty(current_trials)
            continue; % Skip if epoch has no trials (e.g., LP is very early)
        end
        
        % Calculate lick rate for the current epoch's trials
        licks_epoch = task_data(ianimal).spatial_binned_data.licks(current_trials, :);
        durations_epoch = task_data(ianimal).spatial_binned_data.durations(current_trials, :);
        lick_rate_epoch = licks_epoch ./ durations_epoch;
        lick_rate_epoch(isnan(lick_rate_epoch) | isinf(lick_rate_epoch)) = 0;
        
        % Determine policy start/stop for each trial in the epoch
        all_rates = lick_rate_epoch(:);
        lick_rate_threshold = prctile(all_rates(all_rates > 0), 95) * 0.5; % Use a robust threshold
        
        start_bins = nan(1, length(current_trials));
        stop_bins = nan(1, length(current_trials));

        for itrial = 1:size(lick_rate_epoch, 1)
            bins_this_trial = find(lick_rate_epoch(itrial, :) >= lick_rate_threshold);
            if ~isempty(bins_this_trial)
                start_bins(itrial) = min(bins_this_trial);
                stop_bins(itrial) = max(bins_this_trial);
            end
        end
        
        % Store the median policy for the epoch
        epoch_policies(i_epoch, 1) = round(median(start_bins, 'omitnan'));
        epoch_policies(i_epoch, 2) = round(median(stop_bins, 'omitnan'));
    end
    
    % --- 3. Plot the Policies as Colored Crosses ---
    % Use the epoch colors defined in your original cfg file for consistency
    epoch_colors = {cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};
    plot_handles = gobjects(3,1); % Graphic object handles for the legend

    for i_epoch = 1:3
        start_bin = epoch_policies(i_epoch, 1);
        stop_bin = epoch_policies(i_epoch, 2);
        if ~isnan(start_bin) && ~isnan(stop_bin)
            plot_handles(i_epoch) = plot(start_bin, stop_bin, '+', ...
                'Color', epoch_colors{i_epoch}, 'MarkerSize', 14, 'LineWidth', 3);
        end
    end    
    
    % --- Formatting for each subplot ---
    title(sprintf('Animal %d', ianimal));
    axis square;
    grid on;
    if ianimal == 1 % Add a legend to the first plot only
        legend(plot_handles(isgraphics(plot_handles)), epoch_names(isgraphics(plot_handles)), ...
               'Location', 'southeast', 'TextColor', 'white', 'FontSize', 8);
    end
    hold off;
end

% --- Finalize the entire figure with shared labels and colorbar ---
title(t, 'Evolution of Licking Strategy on Modeled Reward Landscape', 'FontSize', 16, 'FontWeight', 'bold');
xlabel(t, 'Lick Start Bin', 'FontSize', 12);
ylabel(t, 'Lick Stop Bin', 'FontSize', 12);

cb = colorbar;
colormap('plasma')
cb.Layout.Tile = 'east';
ylabel(cb, 'Reward Rate (rewards/sec)', 'FontSize', 12);

fprintf('Tiled figure with learning trajectories generated successfully.\n');

%% ================= MLE Policy Inference (start/stop) =================
% This section estimates the licking‐policy window (start, stop) that
% maximises the likelihood of the *expert* lick raster, given σ, and
% stores the result (plus Wald CIs) inside optimality_results.

fprintf('--- Maximum-Likelihood Policy Inference ---\n');

alphaCI = 0.05;                     % 95 % confidence intervals

for ianimal = 1:n_animals
    fprintf('  Animal %d …\n', ianimal);

    % ---------------- prepare expert raster ----------------
    lp = learning_points_task{ianimal};
    expert_trials = lp+1 : lp+10;                 % same “expert” window you used before
    if any(expert_trials > size(task_data(ianimal).spatial_binned_data.licks,1))
        warning('    • not enough trials after LP ⇒ skipping.');  %#ok<WNTAG>
        continue
    end

    lick_counts = task_data(ianimal).spatial_binned_data.licks(expert_trials, :);
    raster      = lick_counts > 0;                % binary raster (T×B)

    % ---------------- positional uncertainty ----------------
    sigmaBins = optimality_results(ianimal).sigma_p_bins;
    if isnan(sigmaBins)
        warning('    • σ is NaN ⇒ skipping.');    %#ok<WNTAG>
        continue
    end

    % ---------------- run MLE inference ----------------
    mleOut = infer_policy_mle(raster, sigmaBins, alphaCI);

    % ---------------- store ----------------
    optimality_results(ianimal).mle_start_bin   = mleOut.startHat;
    optimality_results(ianimal).mle_stop_bin    = mleOut.stopHat;
    optimality_results(ianimal).mle_logL        = mleOut.logL;
    optimality_results(ianimal).mle_ci_start    = mleOut.ciStart;
    optimality_results(ianimal).mle_ci_stop     = mleOut.ciStop;

    fprintf('    → MLE window  [%d → %d]  (95%% CI start %4.1f–%4.1f,  stop %4.1f–%4.1f)\n', ...
        mleOut.startHat, mleOut.stopHat, ...
        mleOut.ciStart(1), mleOut.ciStart(2), ...
        mleOut.ciStop (1), mleOut.ciStop (2));
end


%% ================= Strategy Comparison Analysis =================
% This section formally compares the reward rate of the animal's actual
% learned policy against several alternative, simpler strategies to
% demonstrate that the learned behavior is superior.

fprintf('--- Comparing Performance of Alternative Strategies ---\n');

% --- Define the strategies we will test ---
strategy_names = {'Always Lick', 'Animal''s Actual', 'Reactive Licker', 'Perfect RZ'};
num_strategies = length(strategy_names);
strategy_results = nan(n_animals, num_strategies); % Store results [animal x strategy]

% --- Get track parameters once ---
num_bins = floor(cfg.plot.zone_params.corridor_end_au / cfg.plot.zone_params.bin_size);
track_params.num_bins = num_bins;
track_params.bin_size_au = cfg.plot.zone_params.bin_size;
track_params.rz_bins = rz_bins;

% --- Loop through each animal to calculate rates for each strategy ---
for ianimal = 1:n_animals
    
    % Get this animal's specific parameters
    animal_params.baseline_speed = optimality_results(ianimal).baseline_speed_au_s;
    animal_params.slowdown_factor = optimality_results(ianimal).slowdown_factor;
    animal_params.sigma_p_bins = optimality_results(ianimal).sigma_p_bins;
    
    if isnan(animal_params.sigma_p_bins)
        continue; % Skip animal if params are invalid
    end
    
    % --- Strategy 1: Animal's Actual Policy ---
    % We get this from the results of the previous section
    start_bin_actual = optimality_results(ianimal).mle_start_bin;
    stop_bin_actual = optimality_results(ianimal).mle_stop_bin;
    if ~isnan(start_bin_actual) && ~isnan(stop_bin_actual)
        rate_actual = calculate_reward_rate(start_bin_actual, stop_bin_actual, animal_params, track_params, animal_params.sigma_p_bins);
    else
        rate_actual = NaN;
    end
    
    % --- Strategy 2: "Always Lick" ---
    % Licks from the very start to the very end of the track.
    start_bin_always = 1;
    stop_bin_always = num_bins;
    rate_always = calculate_reward_rate(start_bin_always, stop_bin_always, animal_params, track_params, animal_params.sigma_p_bins);

    % --- Strategy 3: "Reactive Licker" ---
    % Starts licking only when the visual zone appears.
    start_bin_reactive = floor(cfg.plot.zone_params.visual_zones_au(1) / track_params.bin_size_au);
    stop_bin_reactive = max(track_params.rz_bins); % Licks until the end of RZ
    rate_reactive = calculate_reward_rate(start_bin_reactive, stop_bin_reactive, animal_params, track_params, animal_params.sigma_p_bins);

    % --- Strategy 4: "Perfect RZ" ---
    % Intends to lick only within the exact reward zone boundaries.
    % The animal's positional uncertainty (sigma_p) is still applied.
    start_bin_perfect = min(track_params.rz_bins);
    stop_bin_perfect = start_bin_perfect+1;
    rate_perfect = calculate_reward_rate(start_bin_perfect, stop_bin_perfect, animal_params, track_params, 0.001);
    
    % Store all calculated rates for this animal
    strategy_results(ianimal, :) = [rate_always, rate_actual, rate_reactive, rate_perfect];
end

% --- Visualize the Strategy Comparison ---

figure
my_simple_errorbar_plot(strategy_results)

% Formatting and Labels
set(gca, 'xtick', 1:num_strategies, 'xticklabel', strategy_names);
xtickangle(30); % Angle labels to prevent overlap
ylabel('Mean Reward Rate (rewards/sec)');
title('Performance Comparison of Learned vs. Alternative Strategies');
hold off;

fprintf('Strategy comparison analysis complete.\n');



%% Actual reward rate 
figure
t = tiledlayout('flow');
for ianimal = 1:n_animals
    lp = learning_points_task{ianimal};
    trial_durations = sum(task_data(ianimal).spatial_binned_data.durations, 2, 'omitmissing');
    nexttile
    shadedErrorBar(1:lp+10, movmean(1./trial_durations(1:lp+10), 5), movstd(1./trial_durations(1:lp+10), 5))
end

%% ================= Deriving Optimal Policy as a Function of Sigma and Cost =================
fprintf('--- Deriving optimal policy via numerical optimization sweep ---\n');

% --- Define the parameter space to explore ---
sigma_sweep = linspace(0.25, 10, 50); % Test 20 values of sigma (in bins)
cost_sweep = linspace(1.0, 0.1, 50);  % Test 21 values of slowdown_factor (1.0 = no cost)

% --- Use a representative "average animal" for other parameters ---
% We can take the mean of the baseline speeds from your results
avg_baseline_speed = mean([optimality_results.baseline_speed_au_s], 'omitnan');
animal_params_sweep = struct('baseline_speed', avg_baseline_speed);

% Get track parameters once
num_bins = floor(cfg.plot.zone_params.corridor_end_au / cfg.plot.zone_params.bin_size);
track_params.num_bins = num_bins;
track_params.bin_size_au = cfg.plot.zone_params.bin_size;
track_params.rz_bins = rz_bins;

% --- Initialize matrices to store the results ---
optimal_starts = nan(length(sigma_sweep), length(cost_sweep));
optimal_stops = nan(length(sigma_sweep), length(cost_sweep));

fprintf('Running optimization across %d parameter combinations...\n', numel(optimal_starts));
h_wait = waitbar(0, 'Running optimization sweep...');

% --- Loop through every combination of sigma and cost ---
total_iterations = numel(optimal_starts);
current_iteration = 0;
for i_sigma = 1:length(sigma_sweep)
    for j_cost = 1:length(cost_sweep)
        % Update parameters for this iteration
        animal_params_sweep.sigma_p_bins = sigma_sweep(i_sigma);
        animal_params_sweep.slowdown_factor = cost_sweep(j_cost);

        % Define the objective function with the current parameters
        obj_fun = @(x) objective_for_optim(x, animal_params_sweep, track_params);

        % Define bounds and initial guess for the optimizer
        lower_bounds = [1, 1];
        upper_bounds = [num_bins, num_bins];
        initial_guess = [min(rz_bins), max(rz_bins)]; % A sensible starting point

        % Use a robust optimizer like patternsearch
        options = optimoptions('patternsearch', 'Display', 'none');
        [opt_policy, ~] = patternsearch(obj_fun, initial_guess, [], [], [], [], lower_bounds, upper_bounds, [], options);

        % Store the optimal start and stop bins
        optimal_starts(i_sigma, j_cost) = round(opt_policy(1));
        optimal_stops(i_sigma, j_cost) = round(opt_policy(2));
        
        current_iteration = current_iteration + 1;
        waitbar(current_iteration / total_iterations, h_wait);
    end
end
close(h_wait);

% --- Visualize the Derived Functional Relationship ---
figure('Position', [200, 200, 1200, 600]);
t = tiledlayout(1, 2, 'TileSpacing', 'compact');

% Plot 1: Optimal Start Bin
nexttile;
imagesc(cost_sweep, sigma_sweep, optimal_starts);
set(gca, 'YDir', 'normal');
title('Optimal Start Bin');
xlabel('Lick Cost (Slowdown Factor)');
ylabel('Positional Uncertainty (\sigma_{bins})');
colormap(gca, 'parula'); % Parula is fine for this type of parameter map
cb1 = colorbar;
ylabel(cb1, 'Bin Index');
hold on;
% Draw a line indicating the reward zone start
yline(min(rz_bins), '--w', 'RZ Start', 'LineWidth', 1.5);

% Plot 2: Optimal Lick Duration
nexttile;
optimal_durations = optimal_stops - optimal_starts; % Calculate duration in bins
imagesc(cost_sweep, sigma_sweep, optimal_durations);
set(gca, 'YDir', 'normal');
title('Optimal Lick Duration');
xlabel('Lick Cost (Slowdown Factor)');
% ylabel('Positional Uncertainty (\sigma_{bins})'); % Not needed for second plot
colormap(gca, 'plasma');
cb2 = colorbar;
ylabel(cb2, 'Duration (bins)');

sgtitle('Derived Optimal Policy as a Function of Cost and Uncertainty', 'FontSize', 16, 'FontWeight', 'bold');
fprintf('Analysis complete.\n');

%% Local functions

function reward_rate = calculate_reward_rate(start_bin, stop_bin, animal_params, track_params, positional_unc)
% Calculates the theoretical reward rate (rewards/sec) for a given licking policy.
%
% INPUTS:
%   start_bin     - The spatial bin where the licking policy begins.
%   stop_bin      - The spatial bin where the licking policy ends.
%   animal_params - A struct with animal-specific measured parameters:
%                   .baseline_speed (in AU/sec)
%                   .slowdown_factor (ratio, e.g., 0.7)
%                   .sigma_p_bins (positional uncertainty in bins)
%   track_params  - A struct with track-specific parameters:
%                   .num_bins (total number of bins on the track)
%                   .bin_size_au (size of each bin in AU)
%                   .rz_bins (a vector of bin indices for the reward zone)
%
% OUTPUT:
%   reward_rate   - The calculated rewards per second for this policy.

% --- 1. Calculate Total Lap Duration ---
bin_size = track_params.bin_size_au;
licking_speed = animal_params.baseline_speed * animal_params.slowdown_factor;

% Calculate distance and time for the licking portion of the policy
num_licking_bins = stop_bin - start_bin + 1;
dist_licking = num_licking_bins * bin_size;
time_licking = dist_licking / licking_speed;

% Calculate distance and time for the non-licking portion
total_track_dist = track_params.num_bins * bin_size;
dist_non_licking = total_track_dist - dist_licking;
time_non_licking = dist_non_licking / animal_params.baseline_speed;

lap_duration_sec = time_licking + time_non_licking;


% --- 2. Calculate Probability of Reward ---
% This accounts for positional uncertainty (sigma_p). A reward is received if
% the animal's *perceived* location is within the reward zone while it is licking.

% Create the animal's intended lick policy (1 where it intends to lick, 0 otherwise)
lick_policy_vector = zeros(1, track_params.num_bins);
lick_policy_vector(start_bin:stop_bin) = 1;

% Create a Gaussian kernel representing positional uncertainty
kernel_size = track_params.num_bins * 2; % Ensure kernel is long enough
gauss_kernel = normpdf(-kernel_size:kernel_size, 0, positional_unc);

% Convolve the policy with the uncertainty kernel to get a "probabilistic"
% lick policy. This vector gives the probability of licking in any given bin.
prob_lick_vector = conv(lick_policy_vector, gauss_kernel, 'same');

% The total probability of getting a reward is the sum of the probabilities
% of licking within the actual reward zone.
prob_reward = sum(prob_lick_vector(track_params.rz_bins));

% Ensure probability does not exceed 1 due to edge effects of convolution
prob_reward = min(prob_reward, 1);


% --- 3. Calculate Final Reward Rate ---
reward_rate = prob_reward / lap_duration_sec;

end

function out = infer_policy_mle(raster, sigmaBins, alpha)
%INFERS_POLICY_MLE  Maximum-likelihood estimate of (start, stop) lick window.
%
%   out = infer_policy_mle(raster, sigmaBins, alpha)
%
%   INPUTS
%   -------               (T = trials,  B = spatial bins)
%   raster      [T×B]   : binary lick matrix (1 = lick, 0 = no-lick)
%   sigmaBins           : positional SD expressed in *bins*
%   alpha               : 1-confidence-level (e.g. 0.05 for 95 % CI)
%
%   OUTPUT (struct)
%   ----------------
%   startHat            : MLE start bin  (1-based)
%   stopHat             : MLE stop  bin  (1-based, ≥ startHat)
%   logL                : maximised log-likelihood
%   ciStart             : [low  high] Wald CI for startHat
%   ciStop              : [low  high] Wald CI for stopHat
%
%   Method:
%     • exhaustive grid-search over all permissible (start,stop) pairs  
%     • Gaussian perceptual blur with s.d. = sigmaBins  
%     • Wald confidence intervals from observed Fisher information
%
%   (C) 2024  – feel free to use / modify.
%

% ---------------- basic checks ----------------
if nargin < 3,  alpha = 0.05;               end
if isempty(raster) || ndims(raster) ~= 2
    error('Input "raster" must be a non-empty 2-D matrix (trials × bins).');
end
raster = logical(raster);                     % force 0/1

[T, B] = size(raster);

% ---------------- perceptual kernel ----------------
span = ceil(5*sigmaBins);                     % ±5 σ covers > 99 %
x    = -span:span;
ker  = exp(-0.5*(x./sigmaBins).^2);
ker  = ker / sum(ker);

% ---------------- exhaustive search ----------------
bestLL = -inf;   bestS = NaN;   bestE = NaN;

for s = 1:B
    for e = s:B
        ll = window_ll(raster, s, e, ker);    % log-likelihood
        if ll > bestLL
            bestLL = ll;   bestS = s;   bestE = e;
        end
    end
end

% ---------------- Wald confidence intervals ----------------
H = zeros(2);                      % 2×2 Hessian via finite differences
h = 1;                             
p0 = [bestS, bestE];
f0 = window_ll(raster, p0(1), p0(2), ker);

for i = 1:2
    for j = 1:2
        shift_i = zeros(1,2);  shift_i(i) = h;
        shift_j = zeros(1,2);  shift_j(j) = h;

        f_pp = window_ll(raster, p0(1)+shift_i(1)+shift_j(1), ...
                                  p0(2)+shift_i(2)+shift_j(2), ker);
        f_pm = window_ll(raster, p0(1)+shift_i(1)-shift_j(1), ...
                                  p0(2)+shift_i(2)-shift_j(2), ker);
        f_mp = window_ll(raster, p0(1)-shift_i(1)+shift_j(1), ...
                                  p0(2)-shift_i(2)+shift_j(2), ker);
        f_mm = window_ll(raster, p0(1)-shift_i(1)-shift_j(1), ...
                                  p0(2)-shift_i(2)-shift_j(2), ker);

        H(i,j) = (f_pp - f_pm - f_mp + f_mm) / (4*h^2);
    end
end

covMat = inv(-H);                         % observed information
se      = sqrt(diag(covMat))';
z       = norminv(1-alpha/2);

ciStart = bestS + z*[-se(1), se(1)];
ciStop  = bestE + z*[-se(2), se(2)];

% ---------------- pack output ----------------
out = struct( ...
    'startHat', bestS, ...
    'stopHat' , bestE, ...
    'logL'    , bestLL, ...
    'ciStart' , ciStart, ...
    'ciStop'  , ciStop ...
    );

end  % -------- end main function --------


% ==============================================================
function ll = window_ll(raster, startBin, stopBin, ker)
%LOG-LIKELIHOOD  Bernoulli log-likelihood for given start/stop window.

B = size(raster,2);

pol = zeros(1,B);
pol(startBin:stopBin) = 1;

p = conv(pol, ker, 'same');               % blurred policy
p = max(min(p, 1-1e-6), 1e-6);            % clamp for log safety

ll = sum( raster .* log(p) + (1-raster) .* log(1-p), 'all' );

end

function neg_reward_rate = objective_for_optim(policy_bins, animal_params, track_params)
% Objective function for optimization. Returns the negative reward rate.
%
% INPUTS:
%   policy_bins   - A 2-element vector [start_bin, stop_bin] to be optimized.
%   animal_params - Struct with .baseline_speed, .slowdown_factor
%   track_params  - Struct with .num_bins, .bin_size_au, .rz_bins
%
% OUTPUT:
%   neg_reward_rate - The negative of the calculated reward rate.

start_bin = round(policy_bins(1));
stop_bin = round(policy_bins(2));

% The policy is invalid if start > stop
if start_bin > stop_bin
    neg_reward_rate = inf; % Return a very high cost for invalid policies
    return;
end

% We can use the original calculate_reward_rate function, just passing the
% animal's own sigma for this objective.
positional_unc = animal_params.sigma_p_bins;

reward_rate = calculate_reward_rate(start_bin, stop_bin, animal_params, track_params, positional_unc);

% The optimizer minimizes, so we return the negative of the reward rate.
neg_reward_rate = -reward_rate;

end