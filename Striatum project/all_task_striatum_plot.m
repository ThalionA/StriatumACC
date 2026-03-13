for ianimal = 1:n_animals_task

    is_dms = task_data(ianimal).is_dms;
    is_dls = task_data(ianimal).is_dls;
    is_acc = task_data(ianimal).is_acc;

    current_activity = task_data(ianimal).spatial_binned_fr_all;
    [~, ~, trials] = size(current_activity);


    try
        change_point = min([task_data(ianimal).change_point_mean, trials]);
    catch
        change_point = trials;
    end

    current_activity = task_data(ianimal).spatial_binned_fr_all(:, :, 1:change_point);
    current_activity_dms = current_activity(is_dms, :, :);
    current_activity_dls = current_activity(is_dls, :, :);
    current_activity_acc = current_activity(is_acc, :, :);

    [neurons, bins, trials] = size(current_activity);


    figure
    t = tiledlayout('flow');
    if sum(is_dms) > 1
        nexttile
        hold on
        h = shadedErrorBar(1:trials, squeeze(mean(current_activity_dms, [1, 2], 'omitmissing'))', mean(squeeze(sem(current_activity_dms, [1, 2]))), 'lineprops', {'Color', color_dms});
        axis tight
        title('DMS')
        xline(learning_points{ianimal})
    end
    if any(is_dls)
        nexttile
        hold on
        g = shadedErrorBar(1:trials, squeeze(mean(current_activity_dls, [1, 2], 'omitmissing'))', mean(squeeze(sem(current_activity_dls, [1, 2]))), 'lineprops', {'Color', color_dls});
        axis tight
        title('DLS')
        xline(learning_points{ianimal})
    end
    if any(is_acc)
        nexttile
        hold on
        j = shadedErrorBar(1:trials, squeeze(mean(current_activity_acc, [1, 2], 'omitmissing'))', mean(squeeze(sem(current_activity_acc, [1, 2]))), 'lineprops', {'Color', color_acc});
        axis tight
        title('ACC')
        xline(learning_points{ianimal})
    end
    % legend([h.mainLine, g.mainLine, j.mainLine], {'DMS', 'DLS', 'ACC'})
    xlabel(t, 'trials')
    ylabel(t, 'firing rate')


    % Pre-post

    % --- Define the bins dimension (if not already defined)
    bins = size(current_activity_dms, 2);

    pre_trials = 4:13;
    post_trials = learning_points{ianimal}:learning_points{ianimal}+9;

    % --- AVERAGE ACROSS FIRST 3 TRIALS ---
    avg_dms_first = squeeze(mean(current_activity_dms(:,:,1:3), [1,3], 'omitmissing'));
    sem_dms_first = mean(squeeze(sem(current_activity_dms(:,:,1:3), [1,3])));

    avg_dls_first = squeeze(mean(current_activity_dls(:,:,1:3), [1,3], 'omitmissing'));
    sem_dls_first = mean(squeeze(sem(current_activity_dls(:,:,1:3), [1,3])));

    avg_acc_first = squeeze(mean(current_activity_acc(:,:,1:3), [1,3], 'omitmissing'));
    sem_acc_first = mean(squeeze(sem(current_activity_acc(:,:,1:3), [1,3])));

    % --- AVERAGE ACROSS TRIALS 21-30 (PRE) ---
    avg_dms_pre = squeeze(mean(current_activity_dms(:,:,pre_trials), [1,3], 'omitmissing'));
    sem_dms_pre = mean(squeeze(sem(current_activity_dms(:,:,pre_trials), [1,3])));

    avg_dls_pre = squeeze(mean(current_activity_dls(:,:,pre_trials), [1,3], 'omitmissing'));
    sem_dls_pre = mean(squeeze(sem(current_activity_dls(:,:,pre_trials), [1,3])));

    avg_acc_pre = squeeze(mean(current_activity_acc(:,:,pre_trials), [1,3], 'omitmissing'));
    sem_acc_pre = mean(squeeze(sem(current_activity_acc(:,:,pre_trials), [1,3])));

    % --- AVERAGE ACROSS TRIALS 33-42 (POST) ---
    avg_dms_post = squeeze(mean(current_activity_dms(:,:,post_trials), [1,3], 'omitmissing'));
    sem_dms_post = mean(squeeze(sem(current_activity_dms(:,:,post_trials), [1,3])));

    avg_dls_post = squeeze(mean(current_activity_dls(:,:,post_trials), [1,3], 'omitmissing'));
    sem_dls_post = mean(squeeze(sem(current_activity_dls(:,:,post_trials), [1,3])));

    avg_acc_post = squeeze(mean(current_activity_acc(:,:,post_trials), [1,3], 'omitmissing'));
    sem_acc_post = mean(squeeze(sem(current_activity_acc(:,:,post_trials), [1,3])));

    % --- CREATE FIGURE ---
    figure

    % Row 1 (first 3 trials)
    % DMS
    if sum(is_dms) > 1
        subplot(3,3,1)
        shadedErrorBar(1:bins, avg_dms_first, sem_dms_first, 'lineprops', {'Color', color_dms})
        title('DMS - First 3 Trials')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

    % DLS
    if sum(is_dls) > 1
        subplot(3,3,2)
        shadedErrorBar(1:bins, avg_dls_first, sem_dls_first, 'lineprops', {'Color', color_dls})
        title('DLS - First 3 Trials')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

    % ACC
    if sum(is_acc) > 1
        subplot(3,3,3)
        shadedErrorBar(1:bins, avg_acc_first, sem_acc_first, 'lineprops', {'Color', color_acc})
        title('ACC - First 3 Trials')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

    % Row 2 (pre)
    % DMS
    if sum(is_dms) > 1
        subplot(3,3,4)
        shadedErrorBar(1:bins, avg_dms_pre, sem_dms_pre, 'lineprops', {'Color', color_dms})
        title('DMS - Pre (4–31)')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

    % DLS
    if sum(is_dls) > 1
        subplot(3,3,5)
        shadedErrorBar(1:bins, avg_dls_pre, sem_dls_pre, 'lineprops', {'Color', color_dls})
        title('DLS - Pre (4–31)')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

    % ACC
    if sum(is_acc) > 1
        subplot(3,3,6)
        shadedErrorBar(1:bins, avg_acc_pre, sem_acc_pre, 'lineprops', {'Color', color_acc})
        title('ACC - Pre (4–31)')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

    % Row 3 (post)
    % DMS
    if sum(is_dms) > 1
        subplot(3,3,7)
        shadedErrorBar(1:bins, avg_dms_post, sem_dms_post, 'lineprops', {'Color', color_dms})
        title('DMS - Post (32–end)')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

    % DLS
    if sum(is_dls) > 1
        subplot(3,3,8)
        shadedErrorBar(1:bins, avg_dls_post, sem_dls_post, 'lineprops', {'Color', color_dls})
        title('DLS - Post (32–end)')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

    % ACC
    if sum(is_acc) > 1
        subplot(3,3,9)
        shadedErrorBar(1:bins, avg_acc_post, sem_acc_post, 'lineprops', {'Color', color_acc})
        title('ACC - Post (32–end)')
        axis tight; box off
        if exist("visual_zone_start_bins", 'var')
            xline([visual_zone_start_bins, reward_zone_start_bins], 'k', 'LineWidth', 2)
        end
    end

end


%%

% Preallocate arrays to store the per-neuron averages for each area and period
dms_early = [];
dms_pre   = [];
dms_post  = [];

dls_early = [];
dls_pre   = [];
dls_post  = [];

acc_early = [];
acc_pre   = [];
acc_post  = [];

for ianimal = 1:n_animals_task
    % Extract boolean indices for the three areas
    is_dms = task_data(ianimal).is_dms;
    is_dls = task_data(ianimal).is_dls;
    is_acc = task_data(ianimal).is_acc;

    % Get the spatially binned firing rate: dimensions are neurons x bins x trials
    current_activity = task_data(ianimal).spatial_binned_fr_all;
    [~, ~, n_trials] = size(current_activity);

    % Define trial ranges for the three periods
    early_range = 1:min(3, n_trials);         % first 3 trials
    pre_range   = 4:min(13, n_trials);          % trials 4 to 13

    % Post-learning period: first 10 trials following the learning point
    % (Ensure that we do not exceed the available trials)
    learning_pt = learning_points{ianimal};
    post_start = learning_pt;
    post_end   = min(learning_pt + 10 - 1, n_trials);
    post_range = post_start:post_end;

    % For the DMS area
    if any(is_dms)
        % Select the DMS neurons
        data_dms = current_activity(is_dms, :, :);

        % Compute the mean firing rate for each neuron in each period
        mean_early = squeeze(mean(mean(data_dms(:, :, early_range), 3, 'omitnan'), 2, 'omitnan'));
        mean_pre   = squeeze(mean(mean(data_dms(:, :, pre_range),   3, 'omitnan'), 2, 'omitnan'));
        mean_post  = squeeze(mean(mean(data_dms(:, :, post_range),  3, 'omitnan'), 2, 'omitnan'));

        % Concatenate the data across animals
        dms_early = [dms_early; mean_early];
        dms_pre   = [dms_pre;   mean_pre];
        dms_post  = [dms_post;  mean_post];
    end

    % For the DLS area
    if any(is_dls)
        data_dls = current_activity(is_dls, :, :);

        mean_early = squeeze(mean(mean(data_dls(:, :, early_range), 3, 'omitnan'), 2, 'omitnan'));
        mean_pre   = squeeze(mean(mean(data_dls(:, :, pre_range),   3, 'omitnan'), 2, 'omitnan'));
        mean_post  = squeeze(mean(mean(data_dls(:, :, post_range),  3, 'omitnan'), 2, 'omitnan'));

        dls_early = [dls_early; mean_early];
        dls_pre   = [dls_pre;   mean_pre];
        dls_post  = [dls_post;  mean_post];
    end

    % For the ACC area
    if any(is_acc)
        data_acc = current_activity(is_acc, :, :);

        mean_early = squeeze(mean(mean(data_acc(:, :, early_range), 3, 'omitnan'), 2, 'omitnan'));
        mean_pre   = squeeze(mean(mean(data_acc(:, :, pre_range),   3, 'omitnan'), 2, 'omitnan'));
        mean_post  = squeeze(mean(mean(data_acc(:, :, post_range),  3, 'omitnan'), 2, 'omitnan'));

        acc_early = [acc_early; mean_early];
        acc_pre   = [acc_pre;   mean_pre];
        acc_post  = [acc_post;  mean_post];
    end
end

figure
t = tiledlayout('flow', 'TileSpacing', 'compact');

% --- DMS subplot ---
nexttile
my_simple_errorbar_plot([dms_early, dms_pre, dms_post])
title('DMS')
xticklabels({'early', 'pre', 'post'})
xlabel('Period')
ylabel('Firing rate')
hold on

% Prepare the data for ANOVA for DMS
data = [dms_early(:); dms_pre(:); dms_post(:)];
groups = [repmat({'early'}, numel(dms_early), 1); ...
          repmat({'pre'},   numel(dms_pre),   1); ...
          repmat({'post'},  numel(dms_post),  1)];
% Ensure the group order is as desired
groups = categorical(groups, {'early', 'pre', 'post'}, 'Ordinal', true);

% Run ANOVA and multiple comparisons (without displaying the ANOVA figure)
[~, ~, stats] = anova1(data, groups, 'off');
[c,~,~,~] = multcompare(stats, 'Display', 'off');
% For three groups, multcompare returns comparisons in the order:
% 1 vs 2, 1 vs 3, and 2 vs 3. The 5th column of c contains the p-values.
p12 = c(1,5);
p13 = c(2,5);
p23 = c(3,5);
% Plot significance markers with sigstar at x positions 1, 2 and 3
sigstar({[1,2], [1,3], [2,3]}, [p12, p13, p23]);

% --- DLS subplot ---
nexttile
my_simple_errorbar_plot([dls_early, dls_pre, dls_post])
title('DLS')
xticklabels({'early', 'pre', 'post'})
xlabel('Period')
ylabel('Firing rate')
hold on

% Prepare the data for ANOVA for DLS
data = [dls_early(:); dls_pre(:); dls_post(:)];
groups = [repmat({'early'}, numel(dls_early), 1); ...
          repmat({'pre'},   numel(dls_pre),   1); ...
          repmat({'post'},  numel(dls_post),  1)];
groups = categorical(groups, {'early', 'pre', 'post'}, 'Ordinal', true);

[~, ~, stats] = anova1(data, groups, 'off');
[c,~,~,~] = multcompare(stats, 'Display', 'off');
p12 = c(1,5);
p13 = c(2,5);
p23 = c(3,5);
sigstar({[1,2], [1,3], [2,3]}, [p12, p13, p23]);

% --- ACC subplot ---
nexttile
my_simple_errorbar_plot([acc_early, acc_pre, acc_post])
title('ACC')
xticklabels({'early', 'pre', 'post'})
xlabel('Period')
ylabel('Firing rate')
hold on

% Prepare the data for ANOVA for ACC
data = [acc_early(:); acc_pre(:); acc_post(:)];
groups = [repmat({'early'}, numel(acc_early), 1); ...
          repmat({'pre'},   numel(acc_pre),   1); ...
          repmat({'post'},  numel(acc_post),  1)];
groups = categorical(groups, {'early', 'pre', 'post'}, 'Ordinal', true);

[~, ~, stats] = anova1(data, groups, 'off');
[c,~,~,~] = multcompare(stats, 'Display', 'off');
p12 = c(1,5);
p13 = c(2,5);
p23 = c(3,5);
sigstar({[1,2], [1,3], [2,3]}, [p12, p13, p23]);