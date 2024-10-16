%% Run preprocessing analysis
clearvars -except all_data
clc

reward_zone_start_cm = 125; % in cm
reward_zone_start_au = 100;

bin_size = 4; % x1.25 = cm
bin_edges = 0:bin_size:200;
bin_edges(end) = 202;
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres);

reward_zone_start_bins = reward_zone_start_au / bin_size;

% Check if preprocessed data exists
if exist('preprocessed_data.mat', 'file')
    fprintf('Loading preprocessed data...\n');
    load('preprocessed_data.mat', 'preprocessed_data');

    n_animals = size(preprocessed_data, 2);
else
    if ~exist("all_data", 'var')
        load all_data.mat
    end

    n_animals = size(all_data, 2);

    fprintf('Processing data for all animals...\n');
    preprocessed_data = struct();

    for ianimal = 1:n_animals
        fprintf('Processing data for animal %d...\n', ianimal);
        % Cut data per trial
        trialData = cut_data_per_trial(all_data, ianimal);

        % Reorganize spikes by area
        all_data = reorganize_spikes_by_area(all_data, ianimal);

        % Align neural data
        n_npx_datapoints = length(all_data(ianimal).npx_time);
        npxStartIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialData.trialStartTimes_vr, 'nearest', 'extrap');
        npxEndIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialData.trialEndTimes_vr, 'nearest', 'extrap');

        % Extract binned spikes per trial
        [binned_spikes_trials, npx_times_trials] = extract_binned_spikes(all_data, ianimal, npxStartIdx, npxEndIdx);

        % Compute trial metrics
        trial_metrics = compute_trial_metrics(trialData);

        % Compute firing rates for DMS, DLS, and ACC
        is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
        is_dls = strcmp(all_data(ianimal).final_areas, 'DLS');
        is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

        final_spikes_dms = all_data(ianimal).final_spikes(is_dms, :);
        final_spikes_dls = all_data(ianimal).final_spikes(is_dls, :);
        final_spikes_acc = all_data(ianimal).final_spikes(is_acc, :);

        binned_spikes_trials_dms = arrayfun(@(s,e) final_spikes_dms(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
        binned_spikes_trials_dls = arrayfun(@(s,e) final_spikes_dls(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
        binned_spikes_trials_acc = arrayfun(@(s,e) final_spikes_acc(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);

        [trial_average_fr_dms, trial_sem_fr_dms] = compute_firing_rates(binned_spikes_trials_dms, trialData.trialDurations_vr);
        [trial_average_fr_dls, trial_sem_fr_dls] = compute_firing_rates(binned_spikes_trials_dls, trialData.trialDurations_vr);
        [trial_average_fr_acc, trial_sem_fr_acc] = compute_firing_rates(binned_spikes_trials_acc, trialData.trialDurations_vr);

        % Find change points
        mov_window_size = 5;
        change_point_mean = find_change_points(trialData.trialDurations_vr, trial_metrics, mov_window_size);

        % Separate dark and corridor periods
        [darkData, corridorData] = separate_dark_and_corridor_periods(trialData, binned_spikes_trials, npx_times_trials);
        trial_lick_positions = cellfun(@(x, y) x(logical(y)), corridorData.trial_position, corridorData.trial_licks, 'UniformOutput', false);

        % Calculate lick performance
        [trial_lick_errors, shuffled_lick_error_means, shuffled_lick_error_stds, trial_lick_error_chance] = cellfun(@(x) calculate_lick_precision(x, reward_zone_start_au), trial_lick_positions);
        trial_lick_fractions = cellfun(@(x) (sum((x > reward_zone_start_au - 20) & x < reward_zone_start_au) + 1)/(sum(x > 0 & x < reward_zone_start_au)+1), trial_lick_positions);

        % Perform spatial binning
        spatial_binned_data = spatial_binning(corridorData, bin_edges, num_bins);

        n_units = size(darkData.binned_spikes{1}, 1);

        % Bin dark data in bins
        temporal_bin_duration = 100; % in ms
        temp_bin_edges = 1:temporal_bin_duration:5001;
        num_temp_bins = numel(temp_bin_edges) - 1;

        temp_binned_dark_spikes = nan(n_units, num_temp_bins, trialData.n_trials-1);

        for itrial = 1:trialData.n_trials
            [~, ~, bin_idx] = histcounts(1:length(darkData.binned_spikes{itrial}), temp_bin_edges);
            for ibin = 1:num_temp_bins
                idx_in_bin = (bin_idx == ibin);
                if any(idx_in_bin)
                    % Compute total spikes
                    temp_binned_dark_spikes(:, ibin, itrial) = sum(darkData.binned_spikes{itrial}(:, idx_in_bin), 2);
                end
            end
        end
        temp_binned_dark_fr = temp_binned_dark_spikes/(temporal_bin_duration/1000);
        temp_binned_dark_fr = temp_binned_dark_fr(:, :, 1:end-1);

        % Prepare data for TCA
        spatial_binned_fr_all = cat(3, spatial_binned_data.firing_rates{:});
        spatial_binned_fr_all = spatial_binned_fr_all(:, :, 1:trialData.n_trials-1);

        z_spatial_binned_fr_all = zscore(spatial_binned_fr_all, [], [2, 3]);

        % Cross area pairwise correlations
        DMS_data = spatial_binned_fr_all(is_dms, :, :);
        DLS_data = spatial_binned_fr_all(is_dls, :, :);
        ACC_data = spatial_binned_fr_all(is_acc, :, :);
        n_neurons_DMS = sum(is_dms);
        n_neurons_DLS = sum(is_dls);
        n_neurons_ACC = sum(is_acc);

        all_cross_area_correlations_DMSACC = nan(trialData.n_trials-1, n_neurons_DMS, n_neurons_ACC);
        all_cross_area_correlations_DMSDLS = nan(trialData.n_trials-1, n_neurons_DMS, n_neurons_DLS);

        for itrial = 1:trialData.n_trials-1
            % Get trial data
            DMS_trial = squeeze(DMS_data(:, :, itrial)); % [n_neurons_DMS x n_spatial_bins]
            DLS_trial = squeeze(DLS_data(:, :, itrial)); % [n_neurons_DLS x n_spatial_bins]
            ACC_trial = squeeze(ACC_data(:, :, itrial)); % [n_neurons_ACC x n_spatial_bins]

            % Compute correlations
            for iNeuron_DMS = 1:n_neurons_DMS
                for iNeuron_ACC = 1:n_neurons_ACC
                    all_cross_area_correlations_DMSACC(itrial, iNeuron_DMS, iNeuron_ACC) = corr(DMS_trial(iNeuron_DMS, :)', ACC_trial(iNeuron_ACC, :)');
                end

                for iNeuron_DLS = 1:n_neurons_DLS
                    all_cross_area_correlations_DMSDLS(itrial, iNeuron_DMS, iNeuron_DLS) = corr(DMS_trial(iNeuron_DMS, :)', DLS_trial(iNeuron_DLS, :)');
                end
            end
        end
        mean_cross_area_corr_DMSACC = squeeze(mean(all_cross_area_correlations_DMSACC, [2, 3], 'omitnan'));
        mean_cross_area_corr_DMSDLS = squeeze(mean(all_cross_area_correlations_DMSDLS, [2, 3], 'omitnan'));

        mean_abs_cross_area_corr_DMSACC = squeeze(mean(abs(all_cross_area_correlations_DMSACC), [2, 3], 'omitnan'));
        mean_abs_cross_area_corr_DMSDLS = squeeze(mean(abs(all_cross_area_correlations_DMSDLS), [2, 3], 'omitnan'));

        % Store all relevant variables into preprocessed_data struct
        preprocessed_data(ianimal).trialData = trialData;
        preprocessed_data(ianimal).is_dms = is_dms;
        preprocessed_data(ianimal).is_dls = is_dls;
        preprocessed_data(ianimal).is_acc = is_acc;
        preprocessed_data(ianimal).binned_spikes_trials = binned_spikes_trials;
        preprocessed_data(ianimal).npx_times_trials = npx_times_trials;
        preprocessed_data(ianimal).trial_metrics = trial_metrics;
        preprocessed_data(ianimal).trial_average_fr_dms = trial_average_fr_dms;
        preprocessed_data(ianimal).trial_sem_fr_dms = trial_sem_fr_dms;
        preprocessed_data(ianimal).trial_average_fr_dls = trial_average_fr_dls;
        preprocessed_data(ianimal).trial_sem_fr_dls = trial_sem_fr_dls;
        preprocessed_data(ianimal).trial_average_fr_acc = trial_average_fr_acc;
        preprocessed_data(ianimal).trial_sem_fr_acc = trial_sem_fr_acc;
        preprocessed_data(ianimal).change_point_mean = change_point_mean;
        preprocessed_data(ianimal).darkData = darkData;
        preprocessed_data(ianimal).corridorData = corridorData;
        preprocessed_data(ianimal).trial_lick_positions = trial_lick_positions;
        preprocessed_data(ianimal).trial_lick_errors = trial_lick_errors;
        preprocessed_data(ianimal).shuffled_lick_error_means = shuffled_lick_error_means;
        preprocessed_data(ianimal).shuffled_lick_error_stds = shuffled_lick_error_stds;
        preprocessed_data(ianimal).trial_lick_fractions = trial_lick_fractions;
        preprocessed_data(ianimal).spatial_binned_data = spatial_binned_data;
        preprocessed_data(ianimal).temp_binned_dark_fr = temp_binned_dark_fr;
        preprocessed_data(ianimal).spatial_binned_fr_all = spatial_binned_fr_all;
        preprocessed_data(ianimal).z_spatial_binned_fr_all = z_spatial_binned_fr_all;
        preprocessed_data(ianimal).mean_cross_area_corr_DMSACC = mean_cross_area_corr_DMSACC;
        preprocessed_data(ianimal).mean_cross_area_corr_DMSDLS = mean_cross_area_corr_DMSDLS;
        preprocessed_data(ianimal).mean_abs_cross_area_corr_DMSACC = mean_abs_cross_area_corr_DMSACC;
        preprocessed_data(ianimal).mean_abs_cross_area_corr_DMSDLS = mean_abs_cross_area_corr_DMSDLS;

        % Add any other variables you might need for plotting or analysis

        fprintf('Done with animal %d\n', ianimal);
    end

    % Save the preprocessed data struct
    save('preprocessed_data.mat', 'preprocessed_data', '-v7.3');
end


%% TCA plotting
for ianimal = 1

    % Run TCA with cross-validation
    xlines_to_plot = [sum(preprocessed_data(ianimal).is_dms), reward_zone_start_bins, preprocessed_data(ianimal).change_point_mean];
    try
        tca_with_cv(preprocessed_data(ianimal).spatial_binned_fr_all(:, :, 1:preprocessed_data(ianimal).change_point_mean+50), 'cp_nmu', 'none', 5, 10, 100, xlines_to_plot);
    catch
        tca_with_cv(preprocessed_data(ianimal).spatial_binned_fr_all(:, :, 1:end), 'cp_nmu', 'none', 5, 10, 100, xlines_to_plot);
    end
end

%% PCA plotting

for ianimal = 1:n_animals
    % Run pca plotting ALL
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all, 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr)
    sgtitle(sprintf('all areas - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_all', ianimal))

    % Run pca plotting DMS
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dms, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dms, :, :))
    sgtitle(sprintf('DMS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_dms', ianimal))

    % Run pca plotting DLS
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_dls, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_dls, :, :))
    sgtitle(sprintf('DLS only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_dls', ianimal))

    % Run pca plotting ACC
    plot_striatum_pca(preprocessed_data(ianimal).spatial_binned_fr_all(preprocessed_data(ianimal).is_acc, :, :), 3, preprocessed_data(ianimal).change_point_mean, preprocessed_data(ianimal).temp_binned_dark_fr(preprocessed_data(ianimal).is_acc, :, :))
    sgtitle(sprintf('ACC only - animal %d', ianimal))
    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    save_to_svg(sprintf('pca_3d_animal%d_acc', ianimal))
end


%% Behaviour plotting

for ianimal = 1:n_animals
    trial_lick_numbers = preprocessed_data(ianimal).trial_metrics.trial_lick_no(1:end-1);
    trial_lick_errors = preprocessed_data(ianimal).trial_lick_errors(1:end-1); % Adjust if necessary
    shuffled_lick_error_means = preprocessed_data(ianimal).shuffled_lick_error_means(1:end-1);
    shuffled_lick_error_stds = preprocessed_data(ianimal).shuffled_lick_error_stds(1:end-1);
    outlier_trials = isoutlier(trial_lick_errors, "percentiles", [0, 99]);
    trial_lick_errors(outlier_trials) = nan;
    trial_lick_errors(1) = nan;
    shuffled_lick_error_means(outlier_trials) = nan;
    shuffled_lick_error_means(1) = nan;

    zscored_errors = (trial_lick_errors - shuffled_lick_error_means)./shuffled_lick_error_stds;

    trial_lick_fractions = preprocessed_data(ianimal).trial_lick_fractions(1:end-1); % Adjust if necessary


    % Plot the correlations between areas
    figure
    mean_corr_DMSDLS = preprocessed_data(ianimal).mean_cross_area_corr_DMSDLS;
    mean_corr_DMSACC = preprocessed_data(ianimal).mean_cross_area_corr_DMSACC;

    subplot(1, 2, 1)
    hold on
    scatter(mean_corr_DMSDLS, zscored_errors')
    xlabel('Cross-area correlation')
    ylabel('Lick error')
    title('DMS-DLS')
    axis tight
    lsline
    [rho, pval] = corr(mean_corr_DMSDLS, zscored_errors', "Rows", "complete");
    legend(sprintf('\\rho = %.3f, pval = %.4f', rho, pval))
    

    subplot(1, 2, 2)
    hold on
    scatter(mean_corr_DMSACC, zscored_errors)
    xlabel('Cross-area correlation')
    ylabel('Lick error')
    title('DMS-ACC')
    axis tight
    lsline
    [rho, pval] = corr(mean_corr_DMSACC, zscored_errors', "Rows", "complete");
    legend(sprintf('\\rho = %.3f, pval = %.4f', rho, pval))

    fig = gcf();
    fig.Position = [100, 100, 1020, 420];
    sgtitle(sprintf('animal %d', ianimal))

    save_to_svg(sprintf('cross_area_lickerror_animal%d', ianimal))


    % Plot licking performance
    mov_window_size = 5;

    figure
    subplot(5, 1, 1)
    shadedErrorBar(1:length(trial_lick_fractions), movmean(trial_lick_fractions, mov_window_size, 'omitmissing'), movstd(trial_lick_fractions, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    xline(preprocessed_data(ianimal).change_point_mean)
    ylabel('precise lick fraction')
    axis tight
    subplot(5, 1, 2)
    trials = 1:length(trial_lick_errors);
    shadedErrorBar(1:length(trial_lick_errors), movmean(zscored_errors, mov_window_size, 'omitmissing'), movstd(zscored_errors, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    % Find indices where z-scores are greater than 2 or less than -2
    outlier_idx = find(zscored_errors <= -2);

    % Get the maximum y-value of the data
    y_max = max(zscored_errors);

    % Set y-level for asterisks slightly above the maximum value
    asterisk_y = y_max + 0.01 * (y_max - min(zscored_errors));  % Adjust 0.1 as necessary
    hold on;
    plot(outlier_idx, repmat(asterisk_y, size(outlier_idx)), 'r*', 'MarkerSize', 2);  % Red asterisks
    % hold on
    % shadedErrorBar(1:length(shuffled_lick_error_means), movmean(shuffled_lick_error_means, mov_window_size, 'omitmissing'), movstd(shuffled_lick_error_means, mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size), 'lineprops', {'Color', 'r'})
    xline(preprocessed_data(ianimal).change_point_mean)
    ylabel('lick error')
    axis tight
    subplot(5, 1, 3)
    shadedErrorBar(1:length(trial_lick_numbers), movmean(trial_lick_numbers, mov_window_size), movstd(trial_lick_numbers, mov_window_size)/sqrt(mov_window_size))
    ylabel('lick no')
    xline(preprocessed_data(ianimal).change_point_mean)
    axis tight
    % xlabel('trial #')

    subplot(5, 1, 4)
    shadedErrorBar(1:length(mean_corr_DMSDLS), movmean(mean_corr_DMSDLS, mov_window_size, 'omitmissing'), movstd(mean_corr_DMSDLS, mov_window_size, [], 1, 'omitmissing')/sqrt(mov_window_size))
    axis tight
    title('DMS-DLS correlation')
    ylabel(sprintf('\\rho'))
    subplot(5, 1, 5)
    shadedErrorBar(1:length(mean_corr_DMSACC), movmean(mean_corr_DMSACC, mov_window_size, 'omitmissing'), movstd(mean_corr_DMSACC, mov_window_size, [], 1, 'omitmissing')/sqrt(mov_window_size))
    axis tight
    title('DMS-ACC correlation')
    ylabel(sprintf('\\rho'))
    xlabel('trial #')

    sgtitle(sprintf('animal %d', ianimal))
    fig = gcf();
    fig.Position = [1145, 15, 560, 980];
    save_to_svg(sprintf('lickquant_animal%d', ianimal))
end


%% Trial-to-trial correlation

% Define colors for each area
color_dms = [0, 0.4470, 0.7410];       % Deep Blue for DMS
color_dls =  [0.4660, 0.6740, 0.1880];  % Forest Green for DLS
color_acc = [0.8500, 0.3250, 0.0980];  % Crimson Red for ACC

% Logical indices for each area
areas = {'DMS', 'DLS', 'ACC'};
area_colors = {color_dms, color_dls, color_acc};


for ianimal = 1:n_animals
    all_activity = preprocessed_data(ianimal).spatial_binned_fr_all;
    is_dms = preprocessed_data(ianimal).is_dms;
    is_dls = preprocessed_data(ianimal).is_dls;
    is_acc = preprocessed_data(ianimal).is_acc;

    area_indices = {is_dms, is_dls, is_acc};


    % Assuming 'data' is your 3D matrix of size [neurons x spatial_bins x trials]
    [neurons, spatial_bins, trials] = size(all_activity);
    window_size = 5;
    half_window = floor(window_size / 2);

    % Preallocate the output matrix with NaNs to handle cases with insufficient data
    avg_corrs = NaN(neurons, trials);

    % Loop over each neuron
    for n = 1:neurons
        % Loop over each trial
        for t = 1:trials
            % Define the window of trials centered on trial t
            trial_start = max(1, t - half_window);
            trial_end = min(trials, t + half_window);
            trials_in_window = trial_start:trial_end;
            num_trials_in_window = length(trials_in_window);

            % Proceed only if we have at least two trials to correlate
            if num_trials_in_window > 1
                % Extract the data for the current neuron and window
                data_block = squeeze(all_activity(n, :, trials_in_window)); % Size: [spatial_bins x num_trials_in_window]

                % Compute the correlation matrix for the trials in the current window
                R = corrcoef(data_block);

                % Extract the upper triangle of the correlation matrix (excluding the diagonal)
                upper_triangle = triu(R, 1);
                upper_vals = upper_triangle(upper_triangle ~= 0);

                % Compute the average correlation
                avg_corrs(n, t) = mean(upper_vals);
            else
                % If only one trial is available, set the average correlation to NaN
                avg_corrs(n, t) = NaN;
            end
        end
    end

    % figure
    % % Compute cosine similarity between trials
    % for a = 1:length(areas)
    %     idx = area_indices{a};
    %     if sum(idx) > 0 % Proceed if there are neurons in this area
    %         % Extract data for the area
    %         data_area = all_activity(idx, :, :);
    %         [neurons_area, ~, ~] = size(data_area);
    %
    %         % Flatten the population activity for each trial
    %         population_activity_area = reshape(data_area, [neurons_area * spatial_bins, trials]);
    %
    %         % Normalize the population activity vectors
    %         population_activity_norms_area = sqrt(sum(population_activity_area.^2, 1));
    %         population_activity_norms_area(population_activity_norms_area == 0) = 1;
    %         population_activity_normalized_area = bsxfun(@rdivide, population_activity_area, population_activity_norms_area);
    %
    %         % Preallocate the output vector
    %         avg_cosine_similarity_area = NaN(1, trials);
    %
    %         % Loop over each trial
    %         for t = 1:trials
    %             % Define the window of trials centered on trial t
    %             trial_start = max(1, t - half_window);
    %             trial_end = min(trials, t + half_window);
    %             trials_in_window = trial_start:trial_end;
    %
    %             if length(trials_in_window) > 1
    %                 % Extract data block
    %                 data_block = population_activity_normalized_area(:, trials_in_window);
    %
    %                 % Compute cosine similarity matrix
    %                 cosine_similarity_matrix = data_block' * data_block;
    %
    %                 % Extract unique pairwise similarities
    %                 upper_triangle = triu(cosine_similarity_matrix, 1);
    %                 upper_vals = upper_triangle(upper_triangle ~= 0);
    %
    %                 % Average cosine similarity
    %                 avg_cosine_similarity_area(t) = mean(upper_vals);
    %             else
    %                 avg_cosine_similarity_area(t) = NaN;
    %             end
    %         end
    %
    %         % Plot the average cosine similarity over trials for the area
    %         subplot(3, 1, a)
    %         plot(avg_cosine_similarity_area, 'LineWidth', 2, 'Color', area_colors{a})
    %         ylabel('average cosine similarity')
    %         title(areas{a})
    %         xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k')
    %         axis tight
    %         box off
    %     end
    % end
    % xlabel('trials')
    % sgtitle(sprintf('cosine similarity - animal %d', ianimal))
    % fig = gcf();
    % fig.Position = [1145, 15, 560, 980];
    % save_to_svg(sprintf('cosine similarity - animal %d', ianimal))




    % figure
    % subplot(3, 2, [1, 3, 5])
    % imagesc(avg_corrs)
    % ylabel('neurons')
    % xlabel('trials')
    % yline(sum(is_dms), 'Color', 'w', 'LineWidth', 1)
    % yline(sum(is_dms) + sum(is_dls), 'Color', 'w', 'LineWidth', 1)
    % colorbar
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1)
    %
    % subplot(3, 2, 2)
    % shadedErrorBar(1:trials, mean(avg_corrs(is_dms, :), 'omitmissing'), sem(avg_corrs(is_dms, :)), 'lineprops', {'Color', color_dms})
    % ylabel('corr')
    % xlabel('trials')
    % title('DMS only')
    % xline(preprocessed_data(ianimal).change_point_mean)
    % axis tight
    % ylim([-0.05, 0.5])
    %
    % subplot(3, 2, 4)
    % shadedErrorBar(1:trials, mean(avg_corrs(is_dls, :), 'omitmissing'), sem(avg_corrs(is_dls, :)), 'lineprops', {'Color', color_dls})
    % ylabel('corr')
    % xlabel('trials')
    % title('DLS only')
    % xline(preprocessed_data(ianimal).change_point_mean)
    % axis tight
    % ylim([-0.05, 0.5])
    %
    % subplot(3, 2, 6)
    % shadedErrorBar(1:trials, mean(avg_corrs(is_acc, :), 'omitmissing'), sem(avg_corrs(is_acc, :)), 'lineprops', {'Color', color_acc})
    % ylabel('corr')
    % xlabel('trials')
    % title('ACC only')
    % xline(preprocessed_data(ianimal).change_point_mean)
    % axis tight
    % ylim([-0.05, 0.5])
    %
    % sgtitle(sprintf('average trial-to-trial correlation - animal %d', ianimal))
    % fig = gcf();
    % fig.Position = [933, 11, 750, 950];
    %
    % save_to_svg(sprintf('average trial-to-trial correlation - animal %d', ianimal))



    % % Calculate the generalised variance for each trial, for each area separately
    % % Define regularization parameter
    % epsilon = 1e-4; % Small positive constant
    % 
    % % Initialize arrays to store log-determinant values
    % logdet_dms = NaN(1, trials);
    % logdet_dls = NaN(1, trials);
    % logdet_acc = NaN(1, trials);
    % 
    % logdet_values = {logdet_dms, logdet_dls, logdet_acc};
    % 
    % for a = 1:length(areas)
    %     idx = area_indices{a};
    %     if sum(idx) > 0 % Proceed if there are neurons in this area
    %         neurons_in_area = sum(idx);
    %         logdet_area = NaN(1, trials); % Initialize array for this area
    % 
    %         for t = 1:trials
    %             % Extract data for neurons in this area at trial t
    %             data_area = all_activity(idx, :, t); % Size: [neurons_in_area x spatial_bins]
    % 
    %             % Transpose data so that rows are observations (spatial bins)
    %             % and columns are variables (neurons)
    %             data_area_t = data_area'; % Size: [spatial_bins x neurons_in_area]
    % 
    %             % Check if there are enough observations to compute covariance
    %             if size(data_area_t, 1) >= size(data_area_t, 2)
    %                 % Compute covariance matrix
    %                 cov_matrix = cov(data_area_t);
    %             else
    %                 % If not enough observations, add zeros to match dimensions
    %                 padding = zeros(size(data_area_t, 2) - size(data_area_t, 1), size(data_area_t, 2));
    %                 cov_matrix = cov([data_area_t; padding]);
    %             end
    % 
    %             % Regularize covariance matrix
    %             cov_matrix_regularized = cov_matrix + epsilon * eye(size(cov_matrix));
    % 
    %             % Ensure the covariance matrix is positive definite
    %             % Compute eigenvalues
    %             eigenvalues = eig(cov_matrix_regularized);
    % 
    %             % Handle non-positive eigenvalues (should not occur with regularization)
    %             eigenvalues(eigenvalues <= 0) = epsilon;
    % 
    %             % Compute log-determinant
    %             logdet = sum(log(eigenvalues));
    % 
    %             % Store the log-determinant value
    %             logdet_area(t) = logdet;
    %         end
    % 
    %         % Save the array of log-determinant values
    %         logdet_values{a} = logdet_area;
    %     end
    % end

    % % Extract the computed log-determinant arrays
    % logdet_dms = logdet_values{1};
    % logdet_dls = logdet_values{2};
    % logdet_acc = logdet_values{3};
    % 
    % % Plotting the results for each area
    % figure;
    % 
    % % Plot for DMS
    % subplot(3, 1, 1);
    % plot(logdet_dms, 'LineWidth', 1, 'Color', color_dms);
    % ylabel('Log-Determinant');
    % title('DMS');
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k');
    % axis tight
    % box off
    % 
    % % Plot for DLS
    % subplot(3, 1, 2);
    % plot(logdet_dls, 'LineWidth', 1, 'Color', color_dls);
    % ylabel('Log-Determinant');
    % title('DLS');
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k');
    % axis tight
    % box off
    % 
    % % Plot for ACC
    % subplot(3, 1, 3);
    % plot(logdet_acc, 'LineWidth', 1, 'Color', color_acc);
    % xlabel('Trials');
    % ylabel('Log-Determinant');
    % title('ACC');
    % xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k');
    % axis tight
    % box off
    % 
    % sgtitle(sprintf('Log-Determinant of Covariance Matrix - Animal %d', ianimal));
    % fig = gcf();
    % fig.Position = [1145, 15, 560, 980];

end

%% Factorisation of licks

num_patterns = 4;
% Set options for reproducibility and convergence criteria
options = statset('MaxIter', 1000, 'Display', 'off');

for ianimal = 1:n_animals
    trial_lick_positions = preprocessed_data(ianimal).spatial_binned_data.licks(1:end-1, :);
    
    % Perform NMF on the entire dataset with the chosen number of factors
    [W, H] = nnmf(trial_lick_positions, num_patterns, 'algorithm', 'mult', 'options', options, 'replicates', 10);
    
    % Compute contribution of each factor
    factor_contributions = zeros(num_patterns, 1);
    for ipattern = 1:num_patterns
        % Reconstruct data using only this factor
        L_approx = W(:, ipattern) * H(ipattern, :);
        % Compute its Frobenius norm (squared)
        factor_contributions(ipattern) = norm(L_approx, 'fro')^2;
    end
    
    % Normalize contributions to sum to 1 (percentage)
    factor_contributions = factor_contributions / sum(factor_contributions);
    
    % Order factors by their contributions
    [sorted_contributions, sort_idx] = sort(factor_contributions, 'descend');
    W = W(:, sort_idx);
    H = H(sort_idx, :);
    
    % Plot the spatial factors ordered by contribution
    figure
    for ipattern = 1:num_patterns
        subplot(num_patterns, 1, ipattern)
        plot(H(ipattern, :))
        title(sprintf('Factor %d (%.2f%%)', ipattern, sorted_contributions(ipattern) * 100))
        ylabel('Intensity')
        xline(reward_zone_start_bins)
    end
    xlabel('Spatial Bins')
    sgtitle(sprintf('Licking Patterns NMF - Animal %d', ianimal))
    
    % Plot the trial factors ordered by contribution
    figure
    for ipattern = 1:num_patterns
        subplot(num_patterns, 1, ipattern)
        shadedErrorBar(1:size(trial_lick_positions, 1), movmean(W(:, ipattern), 10), movstd(W(:, ipattern), 10)/sqrt(10))
        title(sprintf('Factor %d (%.2f%%)', ipattern, sorted_contributions(ipattern) * 100))
        ylabel('Intensity')
        xline(preprocessed_data(ianimal).change_point_mean, 'LineWidth', 1, 'Color', 'k')
    end
    xlabel('Trials')
    sgtitle(sprintf('Licking Patterns NMF - Animal %d', ianimal))
end