%% Average activity

ianimal = 3;
current_activity = preprocessed_data(ianimal).spatial_binned_fr_all;
current_activity_dms = current_activity(preprocessed_data(ianimal).is_dms, :, :);
current_activity_dls = current_activity(preprocessed_data(ianimal).is_dls, :, :);
current_activity_acc = current_activity(preprocessed_data(ianimal).is_acc, :, :);

mean_current_activity = squeeze(mean(current_activity, 2, 'omitmissing'));

[neurons, bins, trials] = size(current_activity);

try
    change_point = min([preprocessed_data(ianimal).change_point_mean, trials]);
catch
    change_point = trials;
end

is_dms = preprocessed_data(ianimal).is_dms;
is_dls = preprocessed_data(ianimal).is_dls;
is_acc = preprocessed_data(ianimal).is_acc;

zscored_lick_errors = preprocessed_data(ianimal).zscored_lick_errors(1:trials);

figure
t = tiledlayout('flow');
if sum(is_dms) > 1
    nexttile
    hold on
    h = shadedErrorBar(1:trials, squeeze(mean(current_activity_dms, [1, 2], 'omitmissing'))', mean(squeeze(sem(current_activity_dms, [1, 2]))), 'lineprops', {'Color', color_dms});
    axis tight
    title('DMS')
end
if any(is_dls)
    nexttile
    hold on
    g = shadedErrorBar(1:trials, squeeze(mean(current_activity_dls, [1, 2], 'omitmissing'))', mean(squeeze(sem(current_activity_dls, [1, 2]))), 'lineprops', {'Color', color_dls});
    axis tight
    title('DLS')
end
if any(is_acc)
    nexttile
    hold on
    j = shadedErrorBar(1:trials, squeeze(mean(current_activity_acc, [1, 2], 'omitmissing'))', mean(squeeze(sem(current_activity_acc, [1, 2]))), 'lineprops', {'Color', color_acc});
    axis tight
    title('ACC')
end
% legend([h.mainLine, g.mainLine, j.mainLine], {'DMS', 'DLS', 'ACC'})
xlabel(t, 'trials')
ylabel(t, 'firing rate')


%% Heatmap
avg_dms = squeeze(mean(current_activity_dms, 1, 'omitmissing'));
figure
subplot(1, 4, 1)
imagesc(avg_dms')
set(gca, 'YDir', 'reverse')
colormap("parula")
colorbar
xlabel('Spatial bin')
ylabel('Trial')
if exist("visual_zone_start_bins", "var")
    xline([visual_zone_start_bins, reward_zone_start_bins], 'w', 'LineWidth', 2)
end
title('DMS')


% 2) DLS
avg_dls = squeeze(mean(current_activity_dls, 1, 'omitmissing'));

subplot(1, 4, 2)
imagesc(avg_dls')
set(gca, 'YDir', 'reverse')
colormap("parula")
colorbar
xlabel('Spatial bin')
ylabel('Trial')
if exist("visual_zone_start_bins", "var")
    xline([visual_zone_start_bins, reward_zone_start_bins], 'w', 'LineWidth', 2)
end
title('DLS')



% 3) ACC
avg_acc = squeeze(mean(current_activity_acc, 1, 'omitmissing'));

subplot(1, 4, 3)
imagesc(avg_acc')
set(gca, 'YDir', 'reverse')
colormap("parula")
colorbar
xlabel('Spatial bin')
ylabel('Trial')
if exist("visual_zone_start_bins", "var")
    xline([visual_zone_start_bins, reward_zone_start_bins], 'w', 'LineWidth', 2)
end
title('ACC')

subplot(1, 4, 4)
plot(zscored_lick_errors, 1:trials)
set(gca, 'YDir', 'reverse')
yticks([])
box off
xlabel('lick error')
title('Behaviour')


%% Pre-post

% --- Define the bins dimension (if not already defined)
bins = size(current_activity_dms, 2);

pre_trials = 4:31;
post_trials = 32:min(trials, change_point);

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
%% Fano Factors

% Define bin and trial ranges
bins = size(current_activity_dms, 2);  % number of spatial bins
first_trials = 1:3;
pre_trials   = 4:31;
post_trials  = 32:trials;

% DMS
mean_dms_activity_first = mean(current_activity_dms(:, :, first_trials), 3);
variance_dms_activity_first = var(current_activity_dms(:, :, first_trials), [], 3);
ff_dms_first = variance_dms_activity_first ./ mean_dms_activity_first;  % [neurons x bins]

mean_dms_activity_pre = mean(current_activity_dms(:, :, pre_trials), 3);
variance_dms_activity_pre = var(current_activity_dms(:, :, pre_trials), [], 3);
ff_dms_pre = variance_dms_activity_pre ./ mean_dms_activity_pre;

mean_dms_activity_post = mean(current_activity_dms(:, :, post_trials), 3);
variance_dms_activity_post = var(current_activity_dms(:, :, post_trials), [], 3);
ff_dms_post = variance_dms_activity_post ./ mean_dms_activity_post;

% DLS
mean_dls_activity_first = mean(current_activity_dls(:, :, first_trials), 3);
variance_dls_activity_first = var(current_activity_dls(:, :, first_trials), [], 3);
ff_dls_first = variance_dls_activity_first ./ mean_dls_activity_first;

mean_dls_activity_pre = mean(current_activity_dls(:, :, pre_trials), 3);
variance_dls_activity_pre = var(current_activity_dls(:, :, pre_trials), [], 3);
ff_dls_pre = variance_dls_activity_pre ./ mean_dls_activity_pre;

mean_dls_activity_post = mean(current_activity_dls(:, :, post_trials), 3);
variance_dls_activity_post = var(current_activity_dls(:, :, post_trials), [], 3);
ff_dls_post = variance_dls_activity_post ./ mean_dls_activity_post;

% ACC
mean_acc_activity_first = mean(current_activity_acc(:, :, first_trials), 3);
variance_acc_activity_first = var(current_activity_acc(:, :, first_trials), [], 3);
ff_acc_first = variance_acc_activity_first ./ mean_acc_activity_first;

mean_acc_activity_pre = mean(current_activity_acc(:, :, pre_trials), 3);
variance_acc_activity_pre = var(current_activity_acc(:, :, pre_trials), [], 3);
ff_acc_pre = variance_acc_activity_pre ./ mean_acc_activity_pre;

mean_acc_activity_post = mean(current_activity_acc(:, :, post_trials), 3);
variance_acc_activity_post = var(current_activity_acc(:, :, post_trials), [], 3);
ff_acc_post = variance_acc_activity_post ./ mean_acc_activity_post;

figure

%--- DMS row ---
subplot(3,3,1)
shadedErrorBar(1:bins, mean(ff_dms_first, 'omitmissing'), sem(ff_dms_first), ...
    'lineprops', {'Color','b'})
title('DMS - First')
axis tight; box off

subplot(3,3,2)
shadedErrorBar(1:bins, mean(ff_dms_pre, 'omitmissing'), sem(ff_dms_pre), ...
    'lineprops', {'Color','b'})
title('DMS - Pre')
axis tight; box off

subplot(3,3,3)
shadedErrorBar(1:bins, mean(ff_dms_post, 'omitmissing'), sem(ff_dms_post), ...
    'lineprops', {'Color','b'})
title('DMS - Post')
axis tight; box off

%--- DLS row ---
subplot(3,3,4)
shadedErrorBar(1:bins, mean(ff_dls_first, 'omitmissing'), sem(ff_dls_first), ...
    'lineprops', {'Color','r'})
title('DLS - First')
axis tight; box off

subplot(3,3,5)
shadedErrorBar(1:bins, mean(ff_dls_pre, 'omitmissing'), sem(ff_dls_pre), ...
    'lineprops', {'Color','r'})
title('DLS - Pre')
axis tight; box off

subplot(3,3,6)
shadedErrorBar(1:bins, mean(ff_dls_post, 'omitmissing'), sem(ff_dls_post), ...
    'lineprops', {'Color','r'})
title('DLS - Post')
axis tight; box off

%--- ACC row ---
subplot(3,3,7)
shadedErrorBar(1:bins, mean(ff_acc_first, 'omitmissing'), sem(ff_acc_first), ...
    'lineprops', {'Color','g'})
title('ACC - First')
axis tight; box off

subplot(3,3,8)
shadedErrorBar(1:bins, mean(ff_acc_pre, 'omitmissing'), sem(ff_acc_pre), ...
    'lineprops', {'Color','g'})
title('ACC - Pre')
axis tight; box off

subplot(3,3,9)
shadedErrorBar(1:bins, mean(ff_acc_post, 'omitmissing'), sem(ff_acc_post), ...
    'lineprops', {'Color','g'})
title('ACC - Post')
axis tight; box off

linkaxes(findall(gcf,'type','axes'), 'xy');  % sync all subplots

%% Population fano factor

population_ff_dms_first = nan(1, bins);
population_ff_dms_pre = nan(1, bins);
population_ff_dms_post = nan(1, bins);

population_ff_dls_first = nan(1, bins);
population_ff_dls_pre = nan(1, bins);
population_ff_dls_post = nan(1, bins);

population_ff_acc_first = nan(1, bins);
population_ff_acc_pre = nan(1, bins);
population_ff_acc_post = nan(1, bins);

for ibin = 1:bins
    p = polyfit(mean_dms_activity_first(:, ibin), variance_dms_activity_first(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_dms_first(ibin) = p(1);

    p = polyfit(mean_dms_activity_pre(:, ibin), variance_dms_activity_pre(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_dms_pre(ibin) = p(1);

    p = polyfit(mean_dms_activity_post(:, ibin), variance_dms_activity_post(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_dms_post(ibin) = p(1);

    p = polyfit(mean_dls_activity_first(:, ibin), variance_dls_activity_first(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_dls_first(ibin) = p(1);

    p = polyfit(mean_dls_activity_pre(:, ibin), variance_dls_activity_pre(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_dls_pre(ibin) = p(1);

    p = polyfit(mean_dls_activity_post(:, ibin), variance_dls_activity_post(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_dls_post(ibin) = p(1);

    p = polyfit(mean_acc_activity_first(:, ibin), variance_acc_activity_first(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_acc_first(ibin) = p(1);

    p = polyfit(mean_acc_activity_pre(:, ibin), variance_acc_activity_pre(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_acc_pre(ibin) = p(1);

    p = polyfit(mean_acc_activity_post(:, ibin), variance_acc_activity_post(:, ibin), 1);  % Fit a 1st-degree polynomial (a straight line)
    population_ff_acc_post(ibin) = p(1);
end

% figure
% hold on
% plot(population_ff_dms_first, 'LineWidth', 1)
% plot(population_ff_dms_pre, 'LineWidth', 1)
% plot(population_ff_dms_post, 'LineWidth', 1)

% Create a matching condition group vector:
condition_labels = [ones(bins, 1);
    2*ones(bins, 1);
    3*ones(bins, 1)];



figure
subplot(1, 3, 1)
% Concatenate into one long vector:
data_to_plot = [population_ff_dms_first(:); population_ff_dms_pre(:); population_ff_dms_post(:)];  % [3*bins x 1]
data_to_plot = data_to_plot(:);  % ensure column

my_errorbar_plot([population_ff_dms_first', population_ff_dms_pre', population_ff_dms_post'])
[~, ~, stats] = anovan(data_to_plot, condition_labels, 'varnames', {'trial_group'}, 'display', 'on');
[comp, ~] = multcompare(stats, 'Display', 'off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6));
xticklabels({'First', 'Pre', 'Post'})
title('DMS')
ylabel('population FF')

subplot(1, 3, 2)
% Concatenate into one long vector:
data_to_plot = [population_ff_dls_first(:); population_ff_dls_pre(:); population_ff_dls_post(:)];  % [3*bins x 1]
data_to_plot = data_to_plot(:);  % ensure column

my_errorbar_plot([population_ff_dls_first', population_ff_dls_pre', population_ff_dls_post'])
[~, ~, stats] = anovan(data_to_plot, condition_labels, 'varnames', {'trial_group'}, 'display', 'on');
[comp, ~] = multcompare(stats, 'Display', 'off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6));
xticklabels({'First', 'Pre', 'Post'})
title('DLS')

subplot(1, 3, 3)
% Concatenate into one long vector:
data_to_plot = [population_ff_acc_first(:); population_ff_acc_pre(:); population_ff_acc_post(:)];  % [3*bins x 1]
data_to_plot = data_to_plot(:);  % ensure column

my_errorbar_plot([population_ff_acc_first', population_ff_acc_pre', population_ff_acc_post'])
[~, ~, stats] = anovan(data_to_plot, condition_labels, 'varnames', {'trial_group'}, 'display', 'on');
[comp, ~] = multcompare(stats, 'Display', 'off');
comp_groups = num2cell(comp(:, 1:2), 2);
sig_ind = comp(:, 6) < 0.05;
sigstar(comp_groups(sig_ind), comp(sig_ind, 6));
xticklabels({'First', 'Pre', 'Post'})
title('ACC')

linkaxes

%% PCA

first_trials = 1:3;
pre_trials   = 4:31;
post_trials  = 32:trials;
num_components = 3;

saturation = linspace(0.1, 1, bins)';  % Goes from 0.1 to 1 across bins

% Hue values for each condition (in HSV)
H_first = 0.6667;   % Blue
H_pre   = 0.0833;   % Yellow
H_post  = 0.333;    % Green

HSV_first = [repmat(H_first, bins, 1), saturation, ones(bins,1)];
colors_first = hsv2rgb(HSV_first);
HSV_pre   = [repmat(H_pre, bins, 1), saturation, ones(bins,1)];
colors_pre = hsv2rgb(HSV_pre);
HSV_post  = [repmat(H_post, bins, 1), saturation, ones(bins,1)];
colors_post = hsv2rgb(HSV_post);

figure

% 1) ALL neurons
subplot(2,2,1)
current_activity_reshaped = reshape(current_activity, neurons, bins*trials);
[coeff, score, ~, ~, ~, ~] = pca(current_activity_reshaped', ...
    'NumComponents', num_components, 'Centered', true);
score_reshaped = reshape(score, [bins, trials, num_components]);

mean_score_first = squeeze(mean(score_reshaped(:, first_trials, :), 2));  % [bins x num_components]
sem_score_first = squeeze(sem(score_reshaped(:, first_trials, :), 2));  % [bins x num_components]

mean_score_pre   = squeeze(mean(score_reshaped(:, pre_trials, :), 2));
sem_score_pre = squeeze(sem(score_reshaped(:, pre_trials, :), 2));  % [bins x num_components]

mean_score_post  = squeeze(mean(score_reshaped(:, post_trials, :), 2));
sem_score_post = squeeze(sem(score_reshaped(:, post_trials, :), 2));  % [bins x num_components]


% Plot the lines
hold on
plot_pca_trajectory(mean_score_first, colors_first);
plot_pca_trajectory(mean_score_pre,   colors_pre);
plot_pca_trajectory(mean_score_post,  colors_post);
hold off
xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
title('All Neurons')
grid on, rotate3d on, view(-25,45)

% 2) DMS
subplot(2,2,2)
[neurons_dms, bins_dms, trials_dms] = size(current_activity_dms);
dms_reshaped = reshape(current_activity_dms, neurons_dms, bins_dms*trials_dms);
[~, score_dms, ~, ~, ~, ~] = pca(dms_reshaped', ...
    'NumComponents', num_components, 'Centered', true);
score_dms_reshaped = reshape(score_dms, [bins_dms, trials_dms, num_components]);

mean_score_first_dms = squeeze(mean(score_dms_reshaped(:, first_trials, :), 2));
sem_score_first_dms = squeeze(sem(score_dms_reshaped(:, first_trials, :), 2));  % [bins x num_components]

mean_score_pre_dms   = squeeze(mean(score_dms_reshaped(:, pre_trials, :), 2));
sem_score_pre_dms = squeeze(sem(score_dms_reshaped(:, pre_trials, :), 2));  % [bins x num_components]

mean_score_post_dms  = squeeze(mean(score_dms_reshaped(:, post_trials, :), 2));
sem_score_post_dms = squeeze(sem(score_dms_reshaped(:, post_trials, :), 2));  % [bins x num_components]


hold on
plot_pca_trajectory(mean_score_first_dms, colors_first);
plot_pca_trajectory(mean_score_pre_dms,   colors_pre);
plot_pca_trajectory(mean_score_post_dms,  colors_post);
hold off
xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
title('DMS')
grid on, rotate3d on, view(-25,45)

% 3) DLS
subplot(2,2,3)
[neurons_dls, bins_dls, trials_dls] = size(current_activity_dls);
dls_reshaped = reshape(current_activity_dls, neurons_dls, bins_dls*trials_dls);
[~, score_dls, ~, ~, ~, ~] = pca(dls_reshaped', ...
    'NumComponents', num_components, 'Centered', true);
score_dls_reshaped = reshape(score_dls, [bins_dls, trials_dls, num_components]);

mean_score_first_dls = squeeze(mean(score_dls_reshaped(:, first_trials, :), 2));
sem_score_first_dls = squeeze(sem(score_dls_reshaped(:, first_trials, :), 2));  % [bins x num_components]

mean_score_pre_dls   = squeeze(mean(score_dls_reshaped(:, pre_trials, :), 2));
sem_score_pre_dls = squeeze(sem(score_dls_reshaped(:, pre_trials, :), 2));  % [bins x num_components]

mean_score_post_dls  = squeeze(mean(score_dls_reshaped(:, post_trials, :), 2));
sem_score_post_dls = squeeze(sem(score_dls_reshaped(:, post_trials, :), 2));  % [bins x num_components]

hold on
plot_pca_trajectory(mean_score_first_dls, colors_first);
plot_pca_trajectory(mean_score_pre_dls,   colors_pre);
plot_pca_trajectory(mean_score_post_dls,  colors_post);
hold off
xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
title('DLS')
grid on, rotate3d on, view(-25,45)

% 4) ACC
subplot(2,2,4)
[neurons_acc, bins_acc, trials_acc] = size(current_activity_acc);
acc_reshaped = reshape(current_activity_acc, neurons_acc, bins_acc*trials_acc);
[~, score_acc, ~, ~, ~, ~] = pca(acc_reshaped', ...
    'NumComponents', num_components, 'Centered', true);
score_acc_reshaped = reshape(score_acc, [bins_acc, trials_acc, num_components]);

mean_score_first_acc = squeeze(mean(score_acc_reshaped(:, first_trials, :), 2));
sem_score_first_acc = squeeze(sem(score_acc_reshaped(:, first_trials, :), 2));  % [bins x num_components]

mean_score_pre_acc   = squeeze(mean(score_acc_reshaped(:, pre_trials, :), 2));
sem_score_pre_acc = squeeze(sem(score_acc_reshaped(:, pre_trials, :), 2));  % [bins x num_components]

mean_score_post_acc  = squeeze(mean(score_acc_reshaped(:, post_trials, :), 2));
sem_score_post_acc = squeeze(sem(score_acc_reshaped(:, post_trials, :), 2));  % [bins x num_components]

hold on
plot_pca_trajectory(mean_score_first_acc, colors_first);
plot_pca_trajectory(mean_score_pre_acc,   colors_pre);
plot_pca_trajectory(mean_score_post_acc,  colors_post);
hold off
xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
title('ACC')
grid on, rotate3d on, view(-25,45)

function plot_pca_trajectory(mean_scores, color_array)
    % mean_scores: [bins x 3] (PC1, PC2, PC3)
    % color_array: [bins x 3] (RGB)
    bins = size(mean_scores, 1);
    for iBin = 1:bins-1
        x1 = mean_scores(iBin,   1);  x2 = mean_scores(iBin+1, 1);
        y1 = mean_scores(iBin,   2);  y2 = mean_scores(iBin+1, 2);
        z1 = mean_scores(iBin,   3);  z2 = mean_scores(iBin+1, 3);
        line([x1 x2], [y1 y2], [z1 z2], ...
            'Color', color_array(iBin, :), 'LineWidth', 2);
    end
end

figure
subplot(2, 2, 1)
hold on
shadedErrorBar(1:bins, mean_score_first(:, 1), sem_score_first(:, 1), 'lineprops', {'Color', colors_first(end, :)})
shadedErrorBar(1:bins, mean_score_pre(:, 1), sem_score_pre(:, 1), 'lineprops', {'Color', colors_pre(end, :)})
shadedErrorBar(1:bins, mean_score_post(:, 1), sem_score_post(:, 1), 'lineprops', {'Color', colors_post(end, :)})
title('All')
subplot(2, 2, 2)
hold on
shadedErrorBar(1:bins, mean_score_first_dms(:, 1), sem_score_first_dms(:, 1), 'lineprops', {'Color', colors_first(end, :)})
shadedErrorBar(1:bins, mean_score_pre_dms(:, 1), sem_score_pre_dms(:, 1), 'lineprops', {'Color', colors_pre(end, :)})
shadedErrorBar(1:bins, mean_score_post_dms(:, 1), sem_score_post_dms(:, 1), 'lineprops', {'Color', colors_post(end, :)})
title('DMS')
subplot(2, 2, 3)
hold on
shadedErrorBar(1:bins, mean_score_first_dls(:, 1), sem_score_first_dls(:, 1), 'lineprops', {'Color', colors_first(end, :)})
shadedErrorBar(1:bins, mean_score_pre_dls(:, 1), sem_score_pre_dls(:, 1), 'lineprops', {'Color', colors_pre(end, :)})
shadedErrorBar(1:bins, mean_score_post_dls(:, 1), sem_score_post_dls(:, 1), 'lineprops', {'Color', colors_post(end, :)})
title('DLS')
subplot(2, 2, 4)
hold on
shadedErrorBar(1:bins, mean_score_first_acc(:, 1), sem_score_first_acc(:, 1), 'lineprops', {'Color', colors_first(end, :)})
shadedErrorBar(1:bins, mean_score_pre_acc(:, 1), sem_score_pre_acc(:, 1), 'lineprops', {'Color', colors_pre(end, :)})
shadedErrorBar(1:bins, mean_score_post_acc(:, 1), sem_score_post_acc(:, 1), 'lineprops', {'Color', colors_post(end, :)})
title('ACC')

%% Generalised variance

figure
tiledlayout('flow', 'TileSpacing', 'compact')
nexttile
genvar = estimate_trialwise_variance(current_activity(:, :, 1:change_point));
title('All units')
nexttile
genvar_dms = estimate_trialwise_variance(current_activity_dms(:, :, 1:change_point));
title('DMS')
nexttile
genvar_dls = estimate_trialwise_variance(current_activity_dls(:, :, 1:change_point));
title('DLS')
nexttile
genvar_acc = estimate_trialwise_variance(current_activity_acc(:, :, 1:change_point));
title('ACC')

%% TCA

[best_mdl, variance_explained, mean_cv_errors, sem_cv_errors] = tca_with_cv(current_activity, 'cp_nmu', 'min-max', 5, 10, 100, true);

% best_mdl = preprocessed_data(ianimal).tca_best_mdl;

tca_model = best_mdl;
n_Factors = size(tca_model.U{2}, 2);

figure
t = tiledlayout(n_Factors, 3, 'TileSpacing', 'compact', 'TileIndexing', 'rowmajor');

for iFactor = 1:n_Factors
    neuron_pattern = tca_model.U{1}(:, iFactor);
    spatial_pattern = tca_model.U{2}(:, iFactor);
    trial_pattern = tca_model.U{3}(:, iFactor);

    nexttile
    bar(neuron_pattern)

    nexttile
    plot(spatial_pattern)

    nexttile
    shadedErrorBar(1:trials, movmean(trial_pattern, 5, 'omitmissing'), movstd(trial_pattern, 5, 'omitmissing')/sqrt(5))

end

%% Decoding

% Choice of model assumption: Poisson
% Performance metric: we store predicted bin index for each test trial bin.

mask_none = false(1, neurons);

doShuffle = false;  % set to true for shuffle control
ablation_mask = mask_none;  % set to mask_dms, mask_dls, mask_acc, etc.

predicted_bins = NaN(trials, bins);  % store final predictions
actual_bins = repmat(1:bins, [trials, 1]);  % ground truth: bin i is labelled i

all_log_likelihoods = nan(trials, bins, bins);

for testTrial = 1:trials
    fprintf('decoding trial %d/%d\n', testTrial, trials)
    %----------------------------------------------------------------------
    % 1) Define training data (all trials except testTrial)
    %----------------------------------------------------------------------
    trainTrials = setdiff(1:trials, testTrial);

    %----------------------------------------------------------------------
    % 2) Exclude ablated neurons
    %----------------------------------------------------------------------
    % ablation_mask is 1 for neurons we want to exclude
    useNeurons = ~ablation_mask;  % useNeurons is 1 for kept neurons

    % Extract the training data for these neurons
    trainData = current_activity(useNeurons, :, trainTrials);
    % trainData size is [nUsedNeurons x nBins x (nTrials-1)]

    %----------------------------------------------------------------------
    % 3) Estimate mean firing rates for each neuron, each bin
    %    across the training trials
    %----------------------------------------------------------------------
    % We'll average along the 3rd dimension (trials)
    % So we get [nUsedNeurons x nBins]
    meanFR = mean(trainData, 3, 'omitnan');

    %----------------------------------------------------------------------
    % 4) Shuffle control:
    %    If doShuffle==true, permute the bin labels for each neuron
    %----------------------------------------------------------------------
    if doShuffle
        for ineuron = 1:size(meanFR,1)
            % random permutation of bin axis for that neuron
            meanFR(ineuron, :) = meanFR(ineuron, randperm(bins));
        end
    end

    %----------------------------------------------------------------------
    % 5) Decode the left-out trial: for each bin in testTrial, compute
    %    likelihood for each candidate bin, choose the best.
    %----------------------------------------------------------------------
    testData = current_activity(useNeurons, :, testTrial);  % [nUsedNeurons x nBins]

    for iBin = 1:bins
        % Observed firing in that bin: [nUsedNeurons x 1]
        observedCounts = testData(:, iBin);

        % We'll compute log-likelihood under each candidate bin
        % for each neuron, then sum across neurons.
        % Poisson: p(r|lambda) = exp(-lambda) * lambda^r / r!
        % We'll do log(p(r|lambda)) to avoid underflow:
        % log p(r|lambda) = -lambda + r*log(lambda) - log(r!)
        % We'll skip the log(r!) term as it doesn't affect argmax.

        % meanFR is [nUsedNeurons x nBins], so for candidate bin c
        % the mean rate is meanFR(:, c).

        logLikelihood = zeros(bins, 1);  % store log-likelihood for each candidate bin
        for c = 1:bins
            lambda_c = meanFR(:, c);  % [nUsedNeurons x 1]
            % Avoid zero or negative
            lambda_c(lambda_c<=0) = 1e-6;

            % observedCounts is r, also [nUsedNeurons x 1]
            % Summation over neurons of r*log(lambda) - lambda
            ll = observedCounts .* log(lambda_c) - lambda_c;
            logLikelihood(c) = sum(ll, 'omitnan');
        end

        % Choose the bin c that maximises the log-likelihood
        [~, bestBin] = max(logLikelihood);
        predicted_bins(testTrial, iBin) = bestBin;

        all_log_likelihoods(testTrial, iBin, :) = logLikelihood;
    end
end

% Now "predicted_bins" has the predicted bin index for each bin of each trial.
% "actual_bins" is the true bin index (1..nBins).

errors = predicted_bins - actual_bins;  % [nTrials x nBins]
abs_err = abs(errors);
mean_abs_err = mean(abs_err(:), 'omitnan');
rmse = mean(errors.^2, 2);
fprintf('Mean absolute bin error = %.2f\n', mean_abs_err);

figure
scatter(predicted_bins(:), actual_bins(:), 50, 'filled', 'MarkerEdgeColor', 'w')
identity_line
xlabel('predicted position')
ylabel('true position')

figure
shadedErrorBar(1:trials, mean(abs_err, 2, 'omitmissing'), sem(abs_err, 2))
xlabel('trials')
ylabel('average absolute error')

figure
shadedErrorBar(1:bins, mean(abs_err, 1, 'omitmissing'), sem(abs_err, 1))
xlabel('position')
ylabel('average absolute error')
if exist('visual_zone_start_bins', 'var')
    xline([visual_zone_start_bins, reward_zone_start_bins])
end

figure
imagesc(abs_err)
xlabel('position')
ylabel('trials')

figure
gridx1 = 0:.1:bins;
gridx2 = gridx1;
[x1,x2] = meshgrid(gridx1, gridx2);
x1 = x1(:);
x2 = x2(:);
xi = [x1 x2];
x = [predicted_bins(:), actual_bins(:)];
ksdensity(x, xi, 'PlotFcn', 'contour');
contourObj = findobj(gca, 'Type', 'Contour');
set(contourObj, 'LineWidth', 1.2);
box off
axis tight
xlabel('predicted position')
ylabel('true position')
identity_line

%% Area correlations

% 1) Compute average activity [bins x trials] for each area
% current_activity_dms, current_activity_dls, current_activity_acc
% are each [neurons x bins x trials].

% Mean across neurons, ignoring NaNs if needed
avg_dms = squeeze(mean(current_activity_dms, 1, 'omitmissing')); % [bins x trials]
avg_dls = squeeze(mean(current_activity_dls, 1, 'omitmissing')); % [bins x trials]
avg_acc = squeeze(mean(current_activity_acc, 1, 'omitmissing')); % [bins x trials]

% 2) For each bin, compute correlation across trials
% We'll create three vectors, each [bins x 1].
corr_dms_dls = NaN(bins, 1);
corr_dms_acc = NaN(bins, 1);
corr_dls_acc = NaN(bins, 1);

for iBin = 1:bins
    % Extract the [1 x trials] slice for that bin in each area
    dms_bin = avg_dms(iBin, :);
    dls_bin = avg_dls(iBin, :);
    acc_bin = avg_acc(iBin, :);

    % Compute Pearson's correlation across trials
    corr_dms_dls(iBin) = corr(dms_bin', dls_bin', 'type', 'Pearson', 'rows','complete');
    corr_dms_acc(iBin) = corr(dms_bin', acc_bin', 'type', 'Pearson', 'rows','complete');
    corr_dls_acc(iBin) = corr(dls_bin', acc_bin', 'type', 'Pearson', 'rows','complete');
end

% 3) Plot the correlation traces over bins
figure
hold on
plot(corr_dms_dls, 'LineWidth', 2, 'DisplayName', 'DMS vs DLS')
plot(corr_dms_acc, 'LineWidth', 2, 'DisplayName', 'DMS vs ACC')
plot(corr_dls_acc, 'LineWidth', 2, 'DisplayName', 'DLS vs ACC')
xlabel('Spatial bin')
ylabel('Correlation (r)')
title('Inter-area correlations across bins')
legend('Location','best')
axis tight
box off
xline([visual_zone_start_bins, reward_zone_start_bins])



% 2) For each trial, compute correlation across bins
corr_dms_dls_bins = NaN(trials, 1);
corr_dms_acc_bins = NaN(trials, 1);
corr_dls_acc_bins = NaN(trials, 1);

for iTrial = 1:trials
    % Extract bin-vectors for that trial
    dms_vec = avg_dms(:, iTrial);  % [nBins x 1]
    dls_vec = avg_dls(:, iTrial);
    acc_vec = avg_acc(:, iTrial);

    % Correlation across bins for each pair
    corr_dms_dls_bins(iTrial) = corr(dms_vec, dls_vec, 'type','Pearson','rows','complete');
    corr_dms_acc_bins(iTrial) = corr(dms_vec, acc_vec, 'type','Pearson','rows','complete');
    corr_dls_acc_bins(iTrial) = corr(dls_vec, acc_vec, 'type','Pearson','rows','complete');
end

% 3) Plot correlation across trials
wnd_sz = 5;
figure
subplot(3, 1, 1)
shadedErrorBar(1:change_point, movmean(corr_dms_dls_bins(1:change_point), wnd_sz), movstd(corr_dms_dls_bins(1:change_point), wnd_sz)/sqrt(wnd_sz))
title('DMS vs DLS')
box off
axis tight
subplot(3, 1, 2)
shadedErrorBar(1:change_point, movmean(corr_dms_acc_bins(1:change_point), wnd_sz), movstd(corr_dms_acc_bins(1:change_point), wnd_sz)/sqrt(wnd_sz))
ylabel('Correlation (r) across bins')
title('DMS vs ACC')
box off
axis tight
subplot(3, 1, 3)
shadedErrorBar(1:change_point, movmean(corr_dls_acc_bins(1:change_point), wnd_sz), movstd(corr_dls_acc_bins(1:change_point), wnd_sz)/sqrt(wnd_sz))
title('DLS vs ACC')
xlabel('Trial')
box off
axis tight
linkaxes
sgtitle('Inter-area correlations (across bins) for each trial')


figure
subplot(2, 2, 1)
scatter(corr_dms_dls_bins, zscored_lick_errors)
lsline
title('DMS-DLS')
subplot(2, 2, 2)
scatter(corr_dms_acc_bins, zscored_lick_errors)
lsline
title('DMS-ACC')
subplot(2, 2, 3)
scatter(corr_dls_acc_bins, zscored_lick_errors)
lsline
title('DLS-ACC')

%% Correlation across trials (with shuffled)

% 1) Mean firing [bins x trials] for each area
avg_dms = squeeze(mean(current_activity_dms, 1, 'omitmissing')); % [bins x trials]
avg_dls = squeeze(mean(current_activity_dls, 1, 'omitmissing')); % [bins x trials]
avg_acc = squeeze(mean(current_activity_acc, 1, 'omitmissing')); % [bins x trials]

[bins, trials] = size(avg_dms);

% 2) Unshuffled correlation
corr_dms_dls = NaN(bins, 1);
corr_dms_acc = NaN(bins, 1);
corr_dls_acc = NaN(bins, 1);

for iBin = 1:bins
    dms_bin = avg_dms(iBin, :);  % [1 x trials]
    dls_bin = avg_dls(iBin, :);
    acc_bin = avg_acc(iBin, :);

    corr_dms_dls(iBin) = corr(dms_bin', dls_bin','rows','complete');
    corr_dms_acc(iBin) = corr(dms_bin', acc_bin','rows','complete');
    corr_dls_acc(iBin) = corr(dls_bin', acc_bin','rows','complete');
end

% 3) 100 shuffles: shuffle the trial dimension for each bin independently
%    Then re-compute correlation. We'll store them in arrays:
nShuffles = 100;
corr_dms_dls_shuf = NaN(nShuffles, bins);
corr_dms_acc_shuf = NaN(nShuffles, bins);
corr_dls_acc_shuf = NaN(nShuffles, bins);

rng('default')  % for reproducibility (optional)

for s = 1:nShuffles
    for iBin = 1:bins
        % For each bin, we shuffle each area's trial vector independently
        % so that the cross-trial alignment is broken.
        perm_dms = avg_dms(iBin, randperm(trials));
        perm_dls = avg_dls(iBin, randperm(trials));
        perm_acc = avg_acc(iBin, randperm(trials));

        corr_dms_dls_shuf(s, iBin) = corr(perm_dms', perm_dls','rows','complete');
        corr_dms_acc_shuf(s, iBin) = corr(perm_dms', perm_acc','rows','complete');
        corr_dls_acc_shuf(s, iBin) = corr(perm_dls', perm_acc','rows','complete');
    end
end

% 4) Compute shuffle stats (e.g. mean and 5th-95th percentile) over the 100 shuffles
shuf_mean_dms_dls = mean(corr_dms_dls_shuf, 1, 'omitnan');
shuf_lo_dms_dls   = prctile(corr_dms_dls_shuf, 5, 1);
shuf_hi_dms_dls   = prctile(corr_dms_dls_shuf, 95, 1);

shuf_mean_dms_acc = mean(corr_dms_acc_shuf, 1, 'omitnan');
shuf_lo_dms_acc   = prctile(corr_dms_acc_shuf, 5, 1);
shuf_hi_dms_acc   = prctile(corr_dms_acc_shuf, 95, 1);

shuf_mean_dls_acc = mean(corr_dls_acc_shuf, 1, 'omitnan');
shuf_lo_dls_acc   = prctile(corr_dls_acc_shuf, 5, 1);
shuf_hi_dls_acc   = prctile(corr_dls_acc_shuf, 95, 1);

% 5) Plot real correlation with shuffle distribution
figure
hold on

% --- DMS vs DLS (real) ---
plot(1:bins, corr_dms_dls, 'b-', 'LineWidth', 1.5, 'DisplayName','DMS–DLS Real')

% --- DMS vs DLS (shuffle) as a shaded region
xvals = 1:bins;
fill([xvals, fliplr(xvals)], ...
    [shuf_lo_dms_dls, fliplr(shuf_hi_dms_dls)], ...
    'b', 'FaceAlpha', 0.2, 'EdgeColor','none', ...
    'DisplayName','DMS–DLS Shuffle 5%-95%');
plot(xvals, shuf_mean_dms_dls, 'b--', 'LineWidth',1, ...
    'DisplayName','DMS–DLS Shuffle Mean')

% --- DMS vs ACC (real) ---
plot(1:bins, corr_dms_acc, 'r-', 'LineWidth', 1.5, 'DisplayName','DMS–ACC Real')

% DMS vs ACC shuffle
fill([xvals, fliplr(xvals)], ...
    [shuf_lo_dms_acc, fliplr(shuf_hi_dms_acc)], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor','none', ...
    'DisplayName','DMS–ACC Shuffle 5%-95%');
plot(xvals, shuf_mean_dms_acc, 'r--', 'LineWidth',1, ...
    'DisplayName','DMS–ACC Shuffle Mean')

% --- DLS vs ACC (real) ---
plot(1:bins, corr_dls_acc, 'g-', 'LineWidth', 1.5, 'DisplayName','DLS–ACC Real')

% DLS vs ACC shuffle
fill([xvals, fliplr(xvals)], ...
    [shuf_lo_dls_acc, fliplr(shuf_hi_dls_acc)], ...
    'g', 'FaceAlpha', 0.2, 'EdgeColor','none', ...
    'DisplayName','DLS–ACC Shuffle 5%-95%');
plot(xvals, shuf_mean_dls_acc, 'g--', 'LineWidth',1, ...
    'DisplayName','DLS–ACC Shuffle Mean')

xlabel('Spatial bin')
ylabel('Correlation (r) across trials')
title('Inter-Area Correlation vs Bins (100 Shuffles)')
legend('Location','best')
box off
axis tight
if exist('visual_start_bins', 'var')
    xline([visual_zone_start_bins, reward_zone_start_bins], 'k--');
end

%% Correlation across bins (with shuffled)

% 1) avg_dms, avg_dls, avg_acc are [bins x trials]
[bins, trials] = size(avg_dms);

% 2) Unshuffled correlation across bins, for each trial
corr_dms_dls_bins = NaN(trials, 1);
corr_dms_acc_bins = NaN(trials, 1);
corr_dls_acc_bins = NaN(trials, 1);

for iTrial = 1:trials
    dms_vec = avg_dms(:, iTrial); % [bins x 1]
    dls_vec = avg_dls(:, iTrial);
    acc_vec = avg_acc(:, iTrial);

    corr_dms_dls_bins(iTrial) = corr(dms_vec, dls_vec, 'rows','complete');
    corr_dms_acc_bins(iTrial) = corr(dms_vec, acc_vec, 'rows','complete');
    corr_dls_acc_bins(iTrial) = corr(dls_vec, acc_vec, 'rows','complete');
end

% 3) 100 Shuffles: shuffle the bin dimension for each trial independently
nShuffles = 100;
corr_dms_dls_shuf2 = NaN(nShuffles, trials);
corr_dms_acc_shuf2 = NaN(nShuffles, trials);
corr_dls_acc_shuf2 = NaN(nShuffles, trials);

rng('default')

for s = 1:nShuffles
    for iTrial = 1:trials
        perm_dms = avg_dms(randperm(bins), iTrial);
        perm_dls = avg_dls(randperm(bins), iTrial);
        perm_acc = avg_acc(randperm(bins), iTrial);

        corr_dms_dls_shuf2(s, iTrial) = corr(perm_dms, perm_dls, 'rows','complete');
        corr_dms_acc_shuf2(s, iTrial) = corr(perm_dms, perm_acc, 'rows','complete');
        corr_dls_acc_shuf2(s, iTrial) = corr(perm_dls, perm_acc, 'rows','complete');
    end
end

% 4) Shuffle stats
shuf_mean_dms_dls_bins = mean(corr_dms_dls_shuf2, 1, 'omitnan');
shuf_lo_dms_dls_bins   = prctile(corr_dms_dls_shuf2, 5, 1);
shuf_hi_dms_dls_bins   = prctile(corr_dms_dls_shuf2, 95, 1);

shuf_mean_dms_acc_bins = mean(corr_dms_acc_shuf2, 1, 'omitnan');
shuf_lo_dms_acc_bins   = prctile(corr_dms_acc_shuf2, 5, 1);
shuf_hi_dms_acc_bins   = prctile(corr_dms_acc_shuf2, 95, 1);

shuf_mean_dls_acc_bins = mean(corr_dls_acc_shuf2, 1, 'omitnan');
shuf_lo_dls_acc_bins   = prctile(corr_dls_acc_shuf2, 5, 1);
shuf_hi_dls_acc_bins   = prctile(corr_dls_acc_shuf2, 95, 1);

% 5) Plot real vs shuffle
figure
hold on

xvals = 1:trials;

% Real correlation
plot(xvals, corr_dms_dls_bins, 'b-', 'LineWidth',1.5, 'DisplayName','DMS–DLS Real')
plot(xvals, corr_dms_acc_bins, 'r-', 'LineWidth',1.5, 'DisplayName','DMS–ACC Real')
plot(xvals, corr_dls_acc_bins, 'g-', 'LineWidth',1.5, 'DisplayName','DLS–ACC Real')

% Shuffle fill for DMS–DLS
fill([xvals, fliplr(xvals)], ...
    [shuf_lo_dms_dls_bins, fliplr(shuf_hi_dms_dls_bins)], ...
    'b', 'FaceAlpha',0.2, 'EdgeColor','none','DisplayName','Shuffle 5%-95% (DMS–DLS)');
plot(xvals, shuf_mean_dms_dls_bins, 'b--', 'LineWidth',1, 'DisplayName','Shuffle Mean (DMS–DLS)')

% Shuffle fill for DMS–ACC
fill([xvals, fliplr(xvals)], ...
    [shuf_lo_dms_acc_bins, fliplr(shuf_hi_dms_acc_bins)], ...
    'r', 'FaceAlpha',0.2, 'EdgeColor','none','DisplayName','Shuffle 5%-95% (DMS–ACC)');
plot(xvals, shuf_mean_dms_acc_bins, 'r--', 'LineWidth',1, 'DisplayName','Shuffle Mean (DMS–ACC)')

% Shuffle fill for DLS–ACC
fill([xvals, fliplr(xvals)], ...
    [shuf_lo_dls_acc_bins, fliplr(shuf_hi_dls_acc_bins)], ...
    'g', 'FaceAlpha',0.2, 'EdgeColor','none','DisplayName','Shuffle 5%-95% (DLS–ACC)');
plot(xvals, shuf_mean_dls_acc_bins, 'g--', 'LineWidth',1, 'DisplayName','Shuffle Mean (DLS–ACC)')

xlabel('Trial')
ylabel('Correlation (r) across bins')
title('Inter-area correlations vs Trials (100 Shuffles)')
legend('Location','best')
box off
axis tight
%% All pairwise correlations

for t = 1:trials
    dms_trial = squeeze(current_activity_dms(:, :, t));
    dls_trial = squeeze(current_activity_dls(:, :, t));
    acc_trial = squeeze(current_activity_acc(:, :, t));

    corr_dms_dls(t) = computeAvgCorr(dms_trial, dls_trial);
    corr_dms_acc(t) = computeAvgCorr(dms_trial, acc_trial);
    corr_dls_acc(t) = computeAvgCorr(dls_trial, acc_trial);
end

%-----------------------
% Plot the results
%-----------------------
figure
hold on
plot(corr_dms_dls, 'LineWidth', 2, 'DisplayName','DMS–DLS')
plot(corr_dms_acc, 'LineWidth', 2, 'DisplayName','DMS–ACC')
plot(corr_dls_acc, 'LineWidth', 2, 'DisplayName','DLS–ACC')
xlabel('Trial')
ylabel('Average pairwise correlation (across bins)')
legend('Location','best')
title('Inter-area average pairwise correlation per trial')
grid on
box off
axis tight

function avgCorrVal = computeAvgCorr(actA, actB)
% actA: [nA x nBins], actB: [nB x nBins]
% Returns the mean correlation across bins for all pairs (i in A, j in B).

nA = size(actA, 1);
nB = size(actB, 1);
allCorrVals = NaN(nA, nB);

for iNeuronA = 1:nA
    for jNeuronB = 1:nB
        allCorrVals(iNeuronA, jNeuronB) = corr(...
            actA(iNeuronA, :)', ...
            actB(jNeuronB, :)', ...
            'type','Pearson','rows','complete');
    end
end

avgCorrVal = mean(allCorrVals(:), 'omitnan');
end

%% 

figure
plot(mean_current_activity')