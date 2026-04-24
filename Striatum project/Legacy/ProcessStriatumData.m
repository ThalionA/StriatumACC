%% Load

if ~exist("all_data", 'var')
    load all_data.mat
end

%% Run analysis
clearvars -except all_data
clc

n_animals = size(all_data, 2);
plot_summary_fig = false;

if plot_summary_fig
    figure
    t = tiledlayout(7, n_animals, "TileIndexing", "columnmajor");
end

reward_zone_start_cm = 125; % in cm
reward_zone_start_au = 100;


bin_size = 4; % x1.25 = cm
bin_edges = 0:bin_size:200;
bin_edges(end) = 202;
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres);

reward_zone_start_bins = reward_zone_start_au/bin_size;


for ianimal = 1:n_animals

    % cut data per trial
    n_vr_datapoints = length(all_data(ianimal).corrected_vr_time);
    n_npx_datapoints = length(all_data(ianimal).npx_time);

    changeIdx_vr = [find(diff(all_data(ianimal).vr_trial) ~= 0), n_vr_datapoints];

    n_trials = numel(changeIdx_vr);

    trialStartIdx_vr = [1, changeIdx_vr(1:end-1) + 1];
    trialEndIdx_vr = changeIdx_vr;
    trialLengths_vr = changeIdx_vr - trialStartIdx_vr + 1;

    trialTimes_vr = mat2cell(all_data(ianimal).corrected_vr_time, 1, trialLengths_vr);
    trial_times_zeroed = cellfun(@(x) x - x(1), trialTimes_vr, UniformOutput=false);
    trialStartTimes_vr = all_data(ianimal).corrected_vr_time(trialStartIdx_vr);
    trialEndTimes_vr = all_data(ianimal).corrected_vr_time(trialEndIdx_vr);

    trialDurations_vr = (trialEndTimes_vr - trialStartTimes_vr)/1000;
    trial_licks = mat2cell(all_data(ianimal).corrected_licks, 1, trialLengths_vr);
    trial_position = mat2cell(all_data(ianimal).vr_position, 1, trialLengths_vr);
    trial_reward = mat2cell(all_data(ianimal).vr_reward, 1, trialLengths_vr);
    trial_world = mat2cell(all_data(ianimal).vr_world, 1, trialLengths_vr);

    npxStartIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialStartTimes_vr, 'nearest', 'extrap');
    npxEndIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialEndTimes_vr, 'nearest', 'extrap');

    is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
    is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

    temp_final_spikes = all_data(ianimal).final_spikes;
    temp_final_spikes(1:sum(is_dms), :) = all_data(ianimal).final_spikes(is_dms, :);
    temp_final_spikes(sum(is_dms)+1:end, :) = all_data(ianimal).final_spikes(is_acc, :);

    all_data(ianimal).final_spikes = temp_final_spikes;
    all_data(ianimal).final_areas(1:sum(is_dms)) = deal({'DMS'});
    all_data(ianimal).final_areas(sum(is_dms)+1:end) = deal({'ACC'});

    is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
    is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

    binned_spikes_trials = arrayfun(@(s,e) all_data(ianimal).final_spikes(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
    npx_times_trials = cellfun(@(x) 0:size(x, 2)-1, binned_spikes_trials, 'UniformOutput', false);

    final_spikes_dms = all_data(ianimal).final_spikes(is_dms, :);
    final_spikes_acc = all_data(ianimal).final_spikes(is_acc, :);

    binned_spikes_trials_dms = arrayfun(@(s,e) final_spikes_dms(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
    binned_spikes_trials_acc = arrayfun(@(s,e) final_spikes_acc(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);

    trial_lick_no = cellfun(@sum, trial_licks);

    trial_success = cellfun(@max, trial_reward);

    trial_average_fr_dms = cellfun(@(x) mean(sum(x, 2)), binned_spikes_trials_dms)./trialDurations_vr;
    trial_sem_fr_dms = cellfun(@(x) sem(sum(x, 2)), binned_spikes_trials_dms)./trialDurations_vr;

    trial_average_fr_acc = cellfun(@(x) mean(sum(x, 2)), binned_spikes_trials_acc)./trialDurations_vr;
    trial_sem_fr_acc = cellfun(@(x) sem(sum(x, 2)), binned_spikes_trials_acc)./trialDurations_vr;

    mov_window_size = 5;
    [~, duration_peaks] = findpeaks(movmean(trialDurations_vr, mov_window_size), 'MinPeakProminence', 5, 'Annotate', 'peaks');
    trial_licks_change = find(movmean(trial_lick_no, 10) < 20, 1);
    trial_success_change = find(movmean(trial_success, 10) < 0.5, 1);

    [~, loc1] = min(abs(duration_peaks - trial_licks_change));
    [~, loc2] = min(abs(duration_peaks - trial_success_change));
    most_likely_change_duration = mean([duration_peaks(loc1), duration_peaks(loc2)]);

    change_point_mean = floor(mean([trial_licks_change, trial_success_change, most_likely_change_duration]));

    corridor_start_idx_vr = cellfun(@(x) find(x > 6, 1), trial_world);
    corridor_start_time_vr = cellfun(@(x, y) x(find(y > 6, 1) + 1), trial_times_zeroed, trial_world);

    corridor_start_idx_npx = nan(size(corridor_start_idx_vr));
    binned_spikes_dark = cell(1, n_trials-1);
    binned_spikes_corridor = cell(1, n_trials-1);

    trial_position_corridor = cell(1, n_trials-1);
    trial_position_dark = cell(1, n_trials-1);

    trial_licks_corridor = cell(1, n_trials-1);
    trial_licks_dark = cell(1, n_trials-1);

    trial_reward_corridor = cell(1, n_trials-1);
    trial_reward_dark = cell(1, n_trials-1);

    trial_times_corridor = cell(1, n_trials-1);
    trial_times_dark = cell(1, n_trials-1);

    spatial_binned_licks = nan(n_trials-1, num_bins);
    spatial_binned_durations = nan(n_trials-1, num_bins);

    spatial_binned_spikes = cell(1, n_trials-1);

    n_units = size(binned_spikes_trials{1}, 1);

    for itrial = 1:n_trials-1
        [~, corridor_start_idx_npx(itrial)] = min(abs(npx_times_trials{itrial} - corridor_start_time_vr(itrial)));
        binned_spikes_dark{itrial} = binned_spikes_trials{itrial}(:, 1:corridor_start_idx_npx(itrial)-1);
        binned_spikes_corridor{itrial} = binned_spikes_trials{itrial}(:, corridor_start_idx_npx(itrial):end);

        trial_position_corridor{itrial} = trial_position{itrial}(corridor_start_idx_vr(itrial):end);
        trial_position_dark{itrial} = trial_position{itrial}(1:corridor_start_idx_vr(itrial)-1);

        trial_licks_corridor{itrial} = trial_licks{itrial}(corridor_start_idx_vr(itrial):end);
        trial_licks_dark{itrial} = trial_licks{itrial}(1:corridor_start_idx_vr(itrial)-1);

        trial_reward_corridor{itrial} = trial_reward{itrial}(corridor_start_idx_vr(itrial):end);
        trial_reward_dark{itrial} = trial_reward{itrial}(1:corridor_start_idx_vr(itrial)-1);

        trial_times_corridor{itrial} = trial_times_zeroed{itrial}(corridor_start_idx_vr(itrial):end);
        trial_times_dark{itrial} = trial_times_zeroed{itrial}(1:corridor_start_idx_vr(itrial)-1);

        % Bin positions
        [~, ~, bin_idx] = histcounts(trial_position_corridor{itrial}, bin_edges);


        spatial_binned_spikes{itrial} = nan(n_units, num_bins);

        trial_times_corridor_zeroed = trial_times_corridor{itrial} - trial_times_corridor{itrial}(1);
        % Bin data in space
        for ibin = 1:num_bins
            idx_in_bin = (bin_idx == ibin);
            if any(idx_in_bin)


                % Compute total licks
                spatial_binned_licks(itrial, ibin) = sum(trial_licks_corridor{itrial}(idx_in_bin));
                bin_times = trial_times_corridor_zeroed(idx_in_bin);
                spatial_binned_durations(itrial, ibin) = (bin_times(end) - bin_times(1))/1000;

                [~, npx_bin_start_idx] = min(abs(bin_times(1) - (0:size(binned_spikes_corridor{itrial}, 2)-1)));
                [~, npx_bin_end_idx] = min(abs(bin_times(end) - (0:size(binned_spikes_corridor{itrial}, 2)-1)));
                spatial_binned_spikes{itrial}(:, ibin) = sum(binned_spikes_corridor{itrial}(:, npx_bin_start_idx:npx_bin_end_idx), 2);

            end
        end

        spatial_binned_firing_rates{itrial} = spatial_binned_spikes{itrial}./spatial_binned_durations(itrial, :);

    end

    fr_dark_trials = cellfun(@(x) sum(x, 2)/(size(x, 2)/1000), binned_spikes_dark, 'UniformOutput', false);
    fr_dark_trials = cat(1, [fr_dark_trials{:}]);

    fr_corridor_trials = cellfun(@(x) sum(x, 2)/(size(x, 2)/1000), binned_spikes_corridor, 'UniformOutput', false);
    fr_corridor_trials = cat(1, [fr_corridor_trials{:}]);

    spatial_binned_velocities = (bin_size * 1.25)./spatial_binned_durations;

    if plot_summary_fig

        nexttile
        plot(movmean(trialDurations_vr, mov_window_size))
        xline(duration_peaks)
        title(num2str(all_data(ianimal).mouseid))
        axis tight
        ylim([5, 40])
        ylabel('trial duration (s)')

        nexttile
        plot(movmean(trial_lick_no, mov_window_size))
        ylabel('lick #')
        axis tight
        xline(trial_licks_change)

        nexttile
        plot(movmean(trial_success, mov_window_size))
        ylabel('reward')
        axis tight
        ylim([-0.2, 1.2])
        xline(trial_success_change)

        nexttile
        shadedErrorBar(1:n_trials, movmean(trial_average_fr_dms, mov_window_size), movmean(trial_sem_fr_dms, mov_window_size))
        ylabel('DMS fr')
        axis tight

        nexttile
        shadedErrorBar(1:n_trials, movmean(trial_average_fr_acc, mov_window_size), movmean(trial_sem_fr_acc, mov_window_size))
        ylabel('ACC fr')
        axis tight

        nexttile
        shadedErrorBar(1:n_trials, mean(fr_corridor_trials), sem(fr_corridor_trials))
        axis tight
        xlabel('trial no')
        ylabel('firing rate')
        title('corridor')

        nexttile
        shadedErrorBar(1:n_trials, mean(fr_dark_trials), sem(fr_dark_trials))
        axis tight
        xlabel('trial no')
        ylabel('firing rate')
        title('dark')
    end


    nana = [spatial_binned_firing_rates{:}];
    spatial_binned_fr_all = reshape(nana, [size(spatial_binned_firing_rates{1}, 1), num_bins, n_trials-1]);
    mean_spatial_binned_fr = mean(spatial_binned_fr_all, 3);
    sem_spatial_binned_fr = sem(spatial_binned_fr_all, 3);

    % TCA with cross validation
    tca_with_cv(spatial_binned_fr_all, 'cp_nmu', 'none', 5, 15, 200, [sum(is_dms), reward_zone_start_bins, change_point_mean])

    fprintf('Done with animal %d\n', ianimal)
end

if plot_summary_fig
    xlabel(t, 'trial #')
end

%%


figure
imagesc(mean_spatial_binned_fr)
yline(sum(is_dms)+0.5, 'r', 'LineWidth', 2)

mean_spatial_binned_fr_engaged = mean(spatial_binned_fr_all(:, :, 1:change_point_mean), 3);
sem_spatial_binned_fr_engaged = sem(spatial_binned_fr_all(:, :, 1:change_point_mean), 3);

mean_spatial_binned_fr_disengaged = mean(spatial_binned_fr_all(:, :, change_point_mean+1:end), 3);
sem_spatial_binned_fr_disengaged = sem(spatial_binned_fr_all(:, :, change_point_mean+1:end), 3);

figure
for iunit = 1:size(mean_spatial_binned_fr, 1)
    shadedErrorBar(1:num_bins, mean_spatial_binned_fr_engaged(iunit, :), sem_spatial_binned_fr_engaged(iunit, :), 'lineProps', '-b')
    hold on
    shadedErrorBar(1:num_bins, mean_spatial_binned_fr_disengaged(iunit, :), sem_spatial_binned_fr_disengaged(iunit, :), 'lineProps', '-r')
    shadedErrorBar(1:num_bins, mean_spatial_binned_fr(iunit, :), sem_spatial_binned_fr(iunit, :))
    title(num2str(iunit))
    hold off
    legend({'engaged', 'disengaged', 'all'})
    pause
end

%% PCA

spatial_binned_fr_reshaped = spatial_binned_fr_all(:, :);
num_components = 3;

[coeff,score,latent,tsquared,explained,mu] = pca(spatial_binned_fr_reshaped', "NumComponents", num_components);

% figure
% plot(cumsum(explained))
% ylabel('explained variance (%)')
% xlabel('component #')

score_reshaped = reshape(score, [num_bins, n_trials-1, num_components]);
mean_score_early = squeeze(mean(score_reshaped(:, 1:3, :), 2));
mean_score_engaged = squeeze(mean(score_reshaped(:, 4:change_point_mean, :), 2));
mean_score_disengaged = squeeze(mean(score_reshaped(:, change_point_mean+1:end, :), 2));

saturation = linspace(0.1, 1, num_bins)';  % Saturation from 0.1 to 1

% Define base hues for each condition (H values in HSV)
H_early = 0.6667;     % Green
H_engaged = 0.3333;   % Blue
H_disengaged = 0;     % Red

% Create HSV color arrays for each condition
HSV_early = [H_early * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_early = hsv2rgb(HSV_early);

HSV_engaged = [H_engaged * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_engaged = hsv2rgb(HSV_engaged);

HSV_disengaged = [H_disengaged * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_disengaged = hsv2rgb(HSV_disengaged);


figure
hold on

if num_components == 3

    % Plot early condition
    x = mean_score_early(:,1);
    y = mean_score_early(:,2);
    z = mean_score_early(:,3);
    for i = 1:length(x)-1
        line([x(i), x(i+1)], [y(i), y(i+1)], [z(i), z(i+1)], ...
            'Color', colors_early(i,:), 'LineWidth', 2);
    end

    % Plot engaged condition
    x = mean_score_engaged(:,1);
    y = mean_score_engaged(:,2);
    z = mean_score_engaged(:,3);
    for i = 1:length(x)-1
        line([x(i), x(i+1)], [y(i), y(i+1)], [z(i), z(i+1)], ...
            'Color', colors_engaged(i,:), 'LineWidth', 2);
    end

    % Plot disengaged condition
    x = mean_score_disengaged(:,1);
    y = mean_score_disengaged(:,2);
    z = mean_score_disengaged(:,3);
    for i = 1:length(x)-1
        line([x(i), x(i+1)], [y(i), y(i+1)], [z(i), z(i+1)], ...
            'Color', colors_disengaged(i,:), 'LineWidth', 2);
    end
    hold off
elseif num_components == 2
    scatter(mean_score_early(:,1), mean_score_early(:,2), ...
        75, colors_early, 'filled', 'MarkerEdgeColor', 'k')
    scatter(mean_score_engaged(:,1), mean_score_engaged(:,2), ...
        75, colors_engaged, 'filled', 'MarkerEdgeColor', 'k')
    scatter(mean_score_disengaged(:,1), mean_score_disengaged(:,2), ...
        75, colors_disengaged, 'filled', 'MarkerEdgeColor', 'k')
    hold off
end

%% Tensor Component Analysis

data = tensor(spatial_binned_fr_all);
maxnFactors = 10;
nfits = 5;
err = nan(maxnFactors, nfits);
opts.maxiters = 200;
opts.printitn = 0;

all_mdls = cell(maxnFactors, nfits);

for inFactor = 1:maxnFactors
    nFactors = inFactor;
    for iFit = 1:nfits
        all_mdls{inFactor, iFit} = cp_nmu(data, nFactors, opts);
        err(inFactor, iFit) = norm(full(all_mdls{inFactor, iFit}) - data)/norm(data);

        fprintf('factor %d/%d, fit %d/%d \n', inFactor, maxnFactors, iFit, nfits)
    end
end

figure
plot(mean(err, 2), '-o')

minfactor = 10;
[~, min_mdl_idx] = min(err(minfactor, :));
best_mdl = all_mdls{minfactor, min_mdl_idx};


figure
t = tiledlayout(minfactor, 1);
for iFactor = 1:minfactor
    nexttile
    bar(best_mdl.u{1}(:, iFactor))
    xline(sum(is_dms))
    axis tight
    linkaxes
end
xlabel(t, 'unit #')

figure
t = tiledlayout(minfactor, 1);
for iFactor = 1:minfactor
    nexttile
    plot(best_mdl.u{2}(:, iFactor))
    xline(reward_zone_start_bins)
    axis tight
    linkaxes
end
xlabel(t, 'spatial bin')

figure
t = tiledlayout(minfactor, 1);
for iFactor = 1:minfactor
    nexttile
    shadedErrorBar(1:n_trials-1, movmean(best_mdl.u{3}(:, iFactor), 10), movstd(best_mdl.u{3}(:, iFactor), 10)/sqrt(10))
    % scatter(1:n_trials-1, best_mdl.u{3}(:, iFactor))
    xline(change_point_mean)
    linkaxes
    axis tight
end
xlabel(t, 'trial #')


%% Tucker decomposition

num_neuron_factors = 10;
num_spatial_factors = 5;
num_trial_factors = 2;

[num_neurons, num_bins, num_trials] = size(spatial_binned_fr_all);

data_reshaped = reshape(spatial_binned_fr_all, num_neurons, []);
data_zscored_reshaped = zscore(data_reshaped, 0, 2);
data_in = reshape(data_zscored_reshaped, num_neurons, num_bins, num_trials);

% Perform Tucker decomposition
tucker_model = tucker_als(tensor(data_in), [num_neuron_factors, num_spatial_factors, num_trial_factors]);

% Extract factor matrices
U1 = tucker_model.U{1};  % Neurons factors (size: num_neurons x 10)
U2 = tucker_model.U{2};  % Spatial bins factors (size: num_spatial_bins x 5)
U3 = tucker_model.U{3};  % Trials factors (size: num_trials x 4)

% Extract core tensor
G = tucker_model.core;   % Core tensor (size: 10 x 5 x 4)

% Plot neuron factors
figure('Name', 'Neuron Factors')
t = tiledlayout(num_neuron_factors, 1);
for ii = 1:num_neuron_factors
    nexttile
    bar(U1(:, ii))
    axis tight
end
xlabel(t, 'unit #')

% Plot spatial factors
figure('Name', 'Spatial Factors')
tiledlayout(num_spatial_factors, 1)
for ii = 1:num_spatial_factors
    nexttile
    plot(U2(:, ii))
    axis tight
end
xlabel(t, 'spatial bin')

% Plot trial factors
figure('Name', 'Trial Factors')
t = tiledlayout(num_trial_factors, 1);
for ii = 1:num_trial_factors
    nexttile
    plot(U3(:, ii))
    xlabel('Trial #')
    axis tight
end
xlabel(t, 'trial #')

% Visualize core tensor slices
for k = 1:num_trial_factors
    G_slice = double(G(:, :, k));
    figure('Name', sprintf('Core Tensor Slice for Trial Component %d', k))
    imagesc(G_slice)
    colorbar
    xlabel('Spatial Components')
    ylabel('Neuron Components')
    title(sprintf('G(:, :, %d)', k))
end


%% TCA with cross validation

tca_with_cv(spatial_binned_fr_all, 'cp_nmu', 'none', 5, 10, 200, [sum(is_dms), reward_zone_start_bins, change_point_mean])

%% Higher Order SVD

% Data Preparation
data_full = spatial_binned_fr_all;

% Assuming data_full is your original data tensor
[num_neurons, num_bins, num_trials] = size(data_full);

% Z-score the data
data_reshaped = reshape(data_full, num_neurons, []);
data_zscored_reshaped = zscore(data_reshaped, 0, 2);
data_zscored = reshape(data_zscored_reshaped, num_neurons, num_bins, num_trials);
data_tensor = tensor(data_zscored);

% Cross-Validation with HOSVD

K = 5;  % Number of folds
num_trials = size(data_zscored, 3);
c = cvpartition(num_trials, 'KFold', K);

candidate_ranks_neurons = 1:10;
candidate_ranks_bins = 1:3;
candidate_ranks_trials = 1:3;

num_ranks_neurons = length(candidate_ranks_neurons);
num_ranks_bins = length(candidate_ranks_bins);
num_ranks_trials = length(candidate_ranks_trials);
cv_errors_hosvd = zeros(num_ranks_neurons, num_ranks_bins, num_ranks_trials, K);

for idx_r1 = 1:num_ranks_neurons
    r1 = candidate_ranks_neurons(idx_r1);
    for idx_r2 = 1:num_ranks_bins
        r2 = candidate_ranks_bins(idx_r2);
        for idx_r3 = 1:num_ranks_trials
            r3 = candidate_ranks_trials(idx_r3);
            fprintf('Testing ranks [%d %d %d]...\n', r1, r2, r3);
            for ifold = 1:K
                fprintf('  Fold %d/%d\n', ifold, K);

                % Split data into training and validation sets
                test_idx = c.test(ifold);
                train_idx = c.training(ifold);

                % data_train = data_tensor(:, :, train_idx);
                data_train = tensor(data_zscored(:, :, train_idx));

                % data_test = data_tensor(:, :, test_idx);
                data_test = tensor(data_zscored(:, :, test_idx));

                % Perform HOSVD on training data
                [U1_full, ~, ~] = svd(double(tenmat(data_train, 1)), 'econ');
                [U2_full, ~, ~] = svd(double(tenmat(data_train, 2)), 'econ');
                [U3_full, ~, ~] = svd(double(tenmat(data_train, 3)), 'econ');

                % Truncate factor matrices
                U1_r = U1_full(:, 1:r1);
                U2_r = U2_full(:, 1:r2);
                U3_r = U3_full(:, 1:r3);

                % Compute core tensor for training data
                core_train = ttm(data_train, {U1_r', U2_r', U3_r'}, [1, 2, 3]);

                % Initialize reconstruction error for this fold
                num_test_trials = sum(test_idx);
                reconstruction_error = zeros(num_test_trials, 1);

                % Process each validation trial individually
                for t = 1:num_test_trials
                    % Extract the t-th validation trial
                    X_t = data_test(:, :, t);

                    % Project X_t onto U1_r and U2_r
                    Z_t = ttm(tensor(X_t), {U1_r', U2_r'}, [1, 2]);

                    % Reconstruct X_t
                    X_t_hat = ttm(Z_t, {U1_r, U2_r}, [1, 2]);

                    % Compute reconstruction error for trial t
                    error_t = norm(X_t - X_t_hat)^2 / norm(X_t)^2;
                    reconstruction_error(t) = error_t;
                end

                % Average reconstruction error over validation trials
                error = mean(reconstruction_error);
                cv_errors_hosvd(idx_r1, idx_r2, idx_r3, ifold) = error;
            end
        end
    end
end

% Aggregate and select optimal ranks
mean_cv_errors_hosvd = mean(cv_errors_hosvd, 4);
[min_error, min_idx] = min(mean_cv_errors_hosvd(:));
[idx_r1_best, idx_r2_best, idx_r3_best] = ind2sub(size(mean_cv_errors_hosvd), min_idx);
optimalRank = [candidate_ranks_neurons(idx_r1_best), candidate_ranks_bins(idx_r2_best), candidate_ranks_trials(idx_r3_best)];
fprintf('Optimal ranks: [%d %d %d]\n', optimalRank);

% Plotting mean CV errors
fixed_r3_idx = idx_r3_best;
errors_fixed_r3 = squeeze(mean_cv_errors_hosvd(:, :, fixed_r3_idx));

figure;
imagesc(candidate_ranks_bins, candidate_ranks_neurons, errors_fixed_r3);
colorbar;
xlabel('Rank r2 (Spatial Bins)');
ylabel('Rank r1 (Neurons)');
title(['Mean CV Error (Fixed r3 = ', num2str(candidate_ranks_trials(fixed_r3_idx)), ')']);

% Fit Final Model and Plot Factors

% Perform HOSVD on full data
[U1_full, ~, ~] = svd(double(tenmat(data_tensor, 1)), 'econ');
[U2_full, ~, ~] = svd(double(tenmat(data_tensor, 2)), 'econ');
[U3_full, ~, ~] = svd(double(tenmat(data_tensor, 3)), 'econ');

% Truncate factor matrices
r1 = optimalRank(1);
r2 = optimalRank(2);
r3 = optimalRank(3);

U1_opt = U1_full(:, 1:r1);
U2_opt = U2_full(:, 1:r2);
U3_opt = U3_full(:, 1:r3);

% Compute core tensor
core_opt = ttm(data_tensor, {U1_opt', U2_opt', U3_opt'}, [1, 2, 3]);

% Plot Neuron Factors
figure;
tiledlayout(r1, 1);
for i = 1:r1
    nexttile;
    bar(U1_opt(:, i));
    title(['Neuron Factor ', num2str(i)]);
    xlabel('Neuron #');
    ylabel('Loading');
    axis tight;
end

% Plot Spatial Bin Factors
figure;
tiledlayout(r2, 1);
for i = 1:r2
    nexttile;
    plot(U2_opt(:, i));
    title(['Spatial Bin Factor ', num2str(i)]);
    xlabel('Spatial Bin #');
    ylabel('Loading');
    axis tight;
end

% Plot Trial Factors
figure;
tiledlayout(r3, 1);
for i = 1:r3
    nexttile;
    plot(U3_opt(:, i));
    title(['Trial Factor ', num2str(i)]);
    xlabel('Trial #');
    ylabel('Loading');
    axis tight;
end

%% NMF on licks and velocity


% Ensure data is non-negative
data_matrix_nonneg = max(spatial_binned_velocities, 0)';

% Choose a range of components to test
maxNumComponents = 5;
reconstruction_errors = zeros(maxNumComponents, 1);

for ifold = 1:maxNumComponents
    fprintf('Testing %d components...\n', ifold);
    [W, H] = nnmf(data_matrix_nonneg, ifold, 'algorithm', 'mult');

    % Reconstruct data
    data_reconstructed = W * H;

    % Compute reconstruction error
    error = norm(data_matrix_nonneg - data_reconstructed, 'fro')^2 / norm(data_matrix_nonneg, 'fro')^2;
    reconstruction_errors(ifold) = error;
end

% Plot reconstruction error vs. number of components
figure;
plot(1:maxNumComponents, reconstruction_errors, 'o-');
xlabel('Number of Components');
ylabel('Normalized Reconstruction Error');
title('NMF Reconstruction Error');
grid on;

figure
t = tiledlayout(maxNumComponents, 1);
for iFactor = 1:maxNumComponents
    nexttile
    plot(W(:, iFactor))
    xline(reward_zone_start_bins)
    axis tight
    linkaxes
end
xlabel(t, 'spatial bin')
title(t, 'lick factors - positions')

figure
t = tiledlayout(maxNumComponents, 1);
for iFactor = 1:maxNumComponents
    nexttile
    shadedErrorBar(1:n_trials-1, movmean(H(iFactor, :), 10), movstd(H(iFactor, :), 10)/sqrt(10))
    % plot(1:n_trials-1, movmean(H(iFactor, :), 10))
    xline(change_point_mean)
    linkaxes
    axis tight
end
xlabel(t, 'trial #')
title(t, 'lick factors - trials')