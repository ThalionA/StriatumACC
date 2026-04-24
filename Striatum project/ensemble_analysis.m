%% Ensemble assignment and purity

neuron_factors_all = best_mdl.u{1};
areas_all = labels_valid.area_labels;

% figure;
% t = tiledlayout('flow');
% for ifactor = 1:best_n_factors
%     nexttile
%     histogram(neuron_factos_all(:, ifactor))
% end

[~, ensemble_assignments] = max(neuron_factors_all, [], 2);

total_ensembles = max(ensemble_assignments);

% Purity analysis
neuron_factors_norm = neuron_factors_all./sum(neuron_factors_all, 2);
neuron_purity = max(neuron_factors_norm, [], 2);

ensemble_entropy = -sum(neuron_factors_norm.*log2(neuron_factors_norm), 2);

figure
scatter(neuron_purity, ensemble_entropy)
lsline
xlabel('purity')
ylabel('entropy')

figure
hold on
histogram(neuron_purity, 25)
histogram(ensemble_entropy, 25)

purity_threshold = 0.5;
pure_fraction = sum(neuron_purity >= purity_threshold)/numel(neuron_purity);
is_pure = neuron_purity >= purity_threshold;

figure
histogram(ensemble_assignments)
xlabel('ensemble')
ylabel('unit no.')
xticks(1:total_ensembles)

figure
t = tiledlayout('flow');
for iensemble = 1:total_ensembles
    neuron_idx = ensemble_assignments == iensemble;
    % neuron_idx = ensemble_assignments == iensemble & is_pure;
    ensemble_areas = categorical([labels_valid.area_labels(neuron_idx)]);

    nexttile
    histogram(ensemble_areas)
    title(sprintf('ensemble %d', iensemble))
end

%% Analyze Ensemble Composition by Neuron Type


fprintf('--- Analyzing ensemble composition by neuron type ---\n');

% --- Configuration for this analysis ---
% Define the mapping from the numeric type in your data to a name
neurontype_map = containers.Map({1, 2, 3}, {'MSN', 'FSN', 'TAN'});
type_keys = cell2mat(neurontype_map.keys);
type_names = neurontype_map.values;
num_types = numel(type_names);
type_colors = lines(num_types); % Assign a distinct color to each type

% Get the ensemble assignments and neuron type labels
% These variables should already be in your workspace from Run_TCA_pipeline
% ensemble_assignments      (n_neurons x 1) vector
% labels_valid.neurontype_labels (n_neurons x 1) vector

total_ensembles = max(ensemble_assignments);

% --- Calculate Composition ---
% We will create a matrix where rows are ensembles and columns are neuron types,
% containing the percentage of each type within that ensemble.
ensemble_composition_matrix = zeros(total_ensembles, num_types);

for i_ensemble = 1:total_ensembles
    % Find the indices of all neurons belonging to this ensemble
    % neuron_idx_in_ensemble = (ensemble_assignments == i_ensemble);

    neuron_idx_in_ensemble = (ensemble_assignments == i_ensemble) & is_pure;
    
    % Get the type labels for only these neurons
    types_in_ensemble = labels_valid.neurontype_labels(neuron_idx_in_ensemble);
    
    % Total number of neurons in this ensemble
    total_neurons_in_ensemble = sum(neuron_idx_in_ensemble);
    
    if total_neurons_in_ensemble > 0
        % Count how many of each type are in this ensemble
        for i_type = 1:num_types
            type_code = type_keys(i_type);
            count = sum(types_in_ensemble == type_code);
            ensemble_composition_matrix(i_ensemble, i_type) = (count / total_neurons_in_ensemble) * 100;
        end
    end
end

% --- Visualize Composition ---
figure('Name', 'Neuron Type Composition of Ensembles', 'Position', [200, 200, 800, 500]);
b = bar(ensemble_composition_matrix, 'stacked');

% Apply colors to the bars
for i_type = 1:num_types
    b(i_type).FaceColor = type_colors(i_type, :);
end

% Formatting and Labels
xlabel('ensemble');
ylabel('proportion of units (%)');
% title('Neuron Type Composition by Ensemble');
legend(type_names, 'Location', 'northeastoutside', 'FontSize', 10);
xticks(1:total_ensembles);
xlim([0.5, total_ensembles + 0.5]);
ylim([0, 100]);
grid on;
box on;
set(gca, 'FontSize', 12);

fprintf('Generated plot for neuron type composition of ensembles.\n');

% --- Optional: Statistical Test ---
% You could perform a chi-squared test to see if the distribution of neuron 
% types is significantly different from the overall distribution across ensembles.

% Overall distribution of neuron types across all 'pure' neurons
overall_counts = zeros(1, num_types);
for i_type = 1:num_types
    overall_counts(i_type) = sum(labels_valid.neurontype_labels == type_keys(i_type));
end
overall_proportions = overall_counts / sum(overall_counts);

fprintf('\n--- Chi-Squared Test for Ensemble Composition ---\n');
for i_ensemble = 1:total_ensembles
    neuron_idx_in_ensemble = (ensemble_assignments == i_ensemble);
    types_in_ensemble = labels_valid.neurontype_labels(neuron_idx_in_ensemble);
    total_in_ensemble = numel(types_in_ensemble);

    observed_counts = zeros(1, num_types);
    for i_type = 1:num_types
        observed_counts(i_type) = sum(types_in_ensemble == type_keys(i_type));
    end
   
end

%% ================= Analyze Unit-to-Ensemble Coupling ===============================
fprintf('--- Analyzing unit-to-ensemble coupling ---\n');

% --- Get data dimensions and initialize storage ---
[nNeurons, nBins, nTrials] = size(supermouse_combined_valid);
ensembles = unique(ensemble_assignments);
nEnsembles = numel(ensembles);

ensemble_colours = colorGradient([0 0 0], [1, 0.2, 0.3], nEnsembles);

% Matrix to store the correlation of each unit with its ensemble for each trial
unit_ensemble_coupling = nan(nNeurons, nTrials);

% --- Calculate Coupling for Each Neuron ---
for iNeuron = 1:nNeurons
    % Identify the neuron's assigned ensemble
    ensemble_id = ensemble_assignments(iNeuron);
    
    % Find indices of all OTHER neurons in the same ensemble (leave-one-out)
    other_units_mask = (ensemble_assignments == ensemble_id) & ((1:nNeurons)' ~= iNeuron);
    
    % If the neuron is the only one in its ensemble, we can't calculate coupling
    if sum(other_units_mask) == 0
        continue;
    end
    
    % Get the activity of the current neuron across all trials [bins x trials]
    unit_activity = squeeze(supermouse_combined_valid(iNeuron, :, :));
    
    % Calculate the mean activity of the rest of the ensemble [bins x trials]
    ensemble_mean_activity = squeeze(mean(supermouse_combined_valid(other_units_mask, :, :), 1, 'omitnan'));
    
    % Calculate the spatial correlation for each trial
    for iTrial = 1:nTrials
        % Correlate the spatial tuning vectors for this specific trial
        r_val = corr(unit_activity(:, iTrial), ensemble_mean_activity(:, iTrial), 'rows', 'pairwise');
        unit_ensemble_coupling(iNeuron, iTrial) = r_val;
    end
end

fprintf('  Coupling calculation complete.\n');

% --- Visualize the Results ---

% 1. Heatmap of all unit couplings over all trials
figure('Name', 'Unit-to-Ensemble Coupling Heatmap');
% Sort by ensemble for better visualization
[sorted_assignments, sort_idx] = sort(ensemble_assignments);
imagesc(unit_ensemble_coupling(sort_idx, :));
colorbar;
xlabel('Aligned Trial Number');
ylabel('Neuron (Sorted by Ensemble)');
title('Unit-to-Ensemble Spatial Coupling (Trial-by-Trial)');
set(gca, 'CLim', [-1 1]); % Set color limits for correlation

% 2. Average coupling per ensemble over time
figure('Name', 'Average Ensemble Coupling Over Time');
hold on;
legend_handles = gobjects(nEnsembles, 1);
for iensemble = 1:nEnsembles
    ensemble_id = ensembles(iensemble);
    ensemble_mask = (ensemble_assignments == ensemble_id);
    
    % Get coupling values for all units in this ensemble
    coupling_data = unit_ensemble_coupling(ensemble_mask, :);
    
    % Calculate mean and SEM across units
    mean_coupling = mean(coupling_data, 1, 'omitnan');
    sem_coupling = std(coupling_data, 0, 1, 'omitnan') / sqrt(sum(ensemble_mask));
    
    % Plot with shaded error bar
    h = shadedErrorBar(1:nTrials, mean_coupling, sem_coupling, 'lineprops', {'Color', ensemble_colours(iensemble, :)});
    legend_handles(iensemble) = h.mainLine;
end
hold off;
xlabel('Aligned Trial Number');
ylabel('Mean Spatial Coupling (r)');
title('Average Unit-to-Ensemble Coupling Evolution');
legend(legend_handles, compose('Ensemble %d', ensembles), 'Location', 'best');
grid on;

% 3. Identify outliers
% Define an outlier as a unit whose average coupling is very low
mean_unit_coupling = mean(unit_ensemble_coupling, 2, 'omitnan');
coupling_threshold = prctile(mean_unit_coupling, 10); % e.g., bottom 10%
outlier_indices = find(mean_unit_coupling < coupling_threshold);

figure('Name', 'Distribution of Average Coupling');
histogram(mean_unit_coupling, 50);
xline(coupling_threshold, 'r--', sprintf('Outlier Threshold (p10 = %.2f)', coupling_threshold));
xlabel('Mean Coupling per Unit (across trials)');
ylabel('Number of Units');
title('Distribution of Unit Coupling Strengths');

fprintf('  Identified %d potential outlier units (mean coupling < %.2f).\n\n', numel(outlier_indices), coupling_threshold);
%% Analyze Joint Distribution of Area and Neuron Type per Ensemble 
% This section creates a detailed breakdown of each ensemble, showing the
% proportion of neurons from each area, further subdivided by neuron type.

fprintf('--- Analyzing joint distribution of area and neuron type per ensemble ---\n');

% --- Configuration ---
% These variables should already be in the workspace from previous sections.
% ensemble_assignments: (n_neurons x 1) vector of ensemble IDs for each neuron.
% labels_valid.area_labels: (n_neurons x 1) cell array of area names.
% labels_valid.neurontype_labels: (n_neurons x 1) vector of numerical neuron type IDs.
% neurontype_map: Map from numerical IDs to type names (e.g., 'MSN', 'FSN').

% Get unique area and type information
area_names = unique(labels_valid.area_labels);
num_areas = numel(area_names);
type_keys = cell2mat(neurontype_map.keys);
type_names = neurontype_map.values;
num_types = numel(type_names);
total_ensembles = max(ensemble_assignments);

% --- Calculate Joint Composition ---
% We will create a 3D matrix to hold the percentages: (Ensemble x Area x Type)
joint_composition_matrix = zeros(total_ensembles, num_areas, num_types);

for i_ensemble = 1:total_ensembles
    % Find indices of all neurons belonging to this ensemble
    neuron_idx_in_ensemble = (ensemble_assignments == i_ensemble);
    total_neurons_in_ensemble = sum(neuron_idx_in_ensemble);

    if total_neurons_in_ensemble == 0
        continue; % Skip empty ensembles
    end

    % Get the area and type labels for only these neurons
    areas_in_ensemble = labels_valid.area_labels(neuron_idx_in_ensemble);
    types_in_ensemble = labels_valid.neurontype_labels(neuron_idx_in_ensemble);

    % Count the occurrences for each area/type combination
    for i_area = 1:num_areas
        for i_type = 1:num_types
            
            % Find neurons that match the current area AND type within this ensemble
            count = sum(strcmp(areas_in_ensemble, area_names{i_area}) & (types_in_ensemble == type_keys(i_type)));
            
            % Store as a percentage of the ensemble's total neuron count
            joint_composition_matrix(i_ensemble, i_area, i_type) = (count / total_neurons_in_ensemble) * 100;
        end
    end
end

% --- Visualize the Joint Distribution ---
figure('Name', 'Joint Area and Neuron Type Composition of Ensembles', 'Position', [100, 100, 1600, 900]);
t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');

% Define a color scheme for neuron types for consistency
type_colors = lines(num_types);

for i_ensemble = 1:total_ensembles
    nexttile;
    
    % Extract the (num_areas x num_types) matrix for this ensemble
    data_for_plot = squeeze(joint_composition_matrix(i_ensemble, :, :));
    
    % Create the grouped bar chart
    b = bar(data_for_plot, 'grouped');
    
    % Apply consistent colors
    for i_type = 1:num_types
        b(i_type).FaceColor = type_colors(i_type, :);
    end
    
    % Formatting for each subplot
    title(sprintf('Ensemble %d', i_ensemble));
    set(gca, 'xtick', 1:num_areas, 'xticklabel', area_names, 'FontSize', 10);
    xtickangle(45);
    grid on;
    box on;
end

linkaxes

% Add shared figure elements
title(t, 'Joint Distribution of Area and Neuron Type Across Ensembles', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(t, 'Proportion of Neurons in Ensemble (%)', 'FontSize', 12);

% Add a single legend to the entire figure
lg = legend(b, type_names, 'FontSize', 12);
lg.Layout.Tile = 'east';

fprintf('Generated plot for joint area/type composition of ensembles.\n');

%% ================= Analyze Purity by Neuron Type =====================================
fprintf('--- Analyzing neuron purity by cell type ---\n');

% --- Check for necessary variables ---
if ~exist('neuron_purity', 'var') || ~isfield(labels_valid, 'neurontype_labels')
    warning('Skipping purity analysis: Required variables not found.');
else
    % --- Get neuron type info from config and labels ---
    if ~exist('neurontype_map', 'var')
        neurontype_map = containers.Map({1, 2, 3}, {'MSN', 'FSN', 'TAN'});
    end
    type_keys = cell2mat(neurontype_map.keys);
    type_names = neurontype_map.values;
    
    % Create a cell array of string labels for all neurons
    nNeurons = numel(neuron_purity);
    type_labels_all = cell(nNeurons, 1);
    for i_type = 1:numel(type_keys)
        type_mask = (labels_valid.neurontype_labels == type_keys(i_type));
        type_labels_all(type_mask) = type_names(i_type);
    end
    
    % --- Filter out units with no assigned type ---
    % Find units that have a valid type label (i.e., the cell is not empty)
    valid_type_mask = ~cellfun(@isempty, type_labels_all); %<-- NEW
    
    % Create filtered lists for plotting and stats
    purity_filtered = neuron_purity(valid_type_mask); %<-- NEW
    type_labels_filtered = type_labels_all(valid_type_mask); %<-- NEW
    
    fprintf('  Analyzing %d / %d units with valid type labels.\n', sum(valid_type_mask), nNeurons);

    % --- Visualize the distribution of purity for each type ---
    figure('Name', 'Neuron Purity by Cell Type');
    boxplot(purity_filtered, type_labels_filtered, 'Notch', 'on'); %<-- USE FILTERED DATA
    title('Distribution of Ensemble Purity by Neuron Type');
    xlabel('Neuron Type');
    ylabel('Neuron Purity (Max Factor Weight)');
    grid on;
    
    % --- Perform statistical analysis (ANOVA) ---
    if numel(unique(type_labels_filtered)) > 1
        [p_val, anova_table, stats] = anova1(purity_filtered, type_labels_filtered, 'off'); %<-- USE FILTERED DATA
        fprintf('  ANOVA test for purity across neuron types:\n');
        fprintf('  F-statistic = %.2f, p-value = %.4f\n', anova_table{2,5}, p_val);
        
        if p_val < 0.05
            fprintf('  ANOVA is significant. Performing post-hoc multiple comparison test...\n');
            figure('Name', 'Post-hoc Test for Purity');
            multcompare(stats, 'CType', 'tukey-kramer');
            title('Multiple Comparison of Mean Purity by Neuron Type');
        else
            fprintf('  No significant difference in purity found across neuron types.\n');
        end
    else
        fprintf('  Only one neuron type present in filtered data. Skipping statistical comparison.\n');
    end
    fprintf('\n');
end
%% t-SNE

lala = tsne(neuron_factors_all);
figure
gscatter(lala(is_pure, 1), lala(is_pure, 2), ensemble_assignments(is_pure), [], [], 15)

% figure
% gscatter(lala(:, 1), lala(:, 2), neuron_purity >= 0.4, [], [], 15)

%% All ensembles

% Parameters for patches
bin_size        = cfg.plot.zone_params.bin_size;
corridor_end_au = cfg.plot.zone_params.corridor_end_au;
bin_edges       = 0:bin_size:corridor_end_au;
if bin_edges(end) <= corridor_end_au, bin_edges(end+1)=bin_edges(end)+bin_size; end
bin_centres     = bin_edges(1:end-1)+diff(bin_edges)/2;

visual_start_idx  = find(bin_centres >= cfg.plot.zone_params.visual_zones_au(1) ,1,'first');
reward_start_idx  = find(bin_centres >= cfg.plot.zone_params.reward_zone_au(1) ,1,'first');
reward_end_idx    = find(bin_centres <= cfg.plot.zone_params.reward_zone_au(2)  ,1,'last');
visual_end_idx    = reward_start_idx-1;                         % visual ends where reward starts

for iensemble = 1:total_ensembles

    ensemble_unit_idx = ensemble_assignments == iensemble;
    ensemble_unit_idx = ensemble_unit_idx & is_pure;

    average_ensemble_activity = squeeze(mean(supermouse_combined_valid(ensemble_unit_idx, :, :)))';

    ymax_all = max(average_ensemble_activity, [], "all");
    ymin_all = min(average_ensemble_activity, [], 'all');

    figure
    imagesc(average_ensemble_activity)
    title(sprintf('ensemble %d', iensemble))

    figure('Position', [0.2 0.2 480 720])
    t = tiledlayout(6, 1);
    for igroup = 1:6
        nexttile
        mean_activity = mean(average_ensemble_activity((igroup-1)*5+1:5*igroup, :));
        sem_activity = sem(average_ensemble_activity((igroup-1)*5+1:5*igroup, :));
        shadedErrorBar(1:50, mean_activity, sem_activity)
        if igroup < 6
            xticks([])
        end
        % yticks([])
        % --- draw patches ---
        ylim([ymin_all, ymax_all])
        yl = ylim;         % returns [0 1] at this point, but we set a real one later
        if yl(1) == yl(2), yl = [0 1]; end   % paranoia for empty axes
        yPatch = [yl(1) yl(1) yl(2) yl(2)];

        if visual_start_idx <= visual_end_idx
            xVis = [visual_start_idx-0.5 visual_end_idx+0.5 visual_end_idx+0.5 visual_start_idx-0.5];
            patch(xVis, yPatch, [0 0.447 0.741], 'EdgeColor','none', 'FaceAlpha',0.15);  % blue
        end
        if reward_start_idx <= reward_end_idx
            xRew = [reward_start_idx-0.5 reward_end_idx+0.5 reward_end_idx+0.5 reward_start_idx-0.5];
            patch(xRew, yPatch, [0.850 0.325 0.098], 'EdgeColor','none', 'FaceAlpha',0.15); % red
        end

        % tighten axes & bring lines to front
        yl = ylim;              % update real limits
        set(findobj(gca,'Type','patch'),'YData',[yl(1) yl(1) yl(2) yl(2)]); % stretch patches

        [~, peak_idx] = max(mean_activity);              % x-index of the peak
        dy            = 0.1 * range(yl);                % arrow length = 10 % of axis height

        hold on
        % arrow starts a bit below the top edge and points down towards the trace
        quiver(peak_idx, yl(2) - 0.05*range(yl),  ...
               0,                         -dy, ...       % (dx = 0, dy < 0 → vertical)
               'MaxHeadSize',1, ...
               'Color','k', 'LineWidth',1, 'MarkerSize', 1);

    end
    title(t, sprintf('ensemble %d', iensemble))
    xlabel(t, 'spatial bin')
    ylabel(t, 'FR')

end

%% Area-specific ensembles

area_to_plot = 'V1';

for iensemble = 1:total_ensembles

    ensemble_unit_idx = ensemble_assignments == iensemble & strcmp(areas_all, area_to_plot);
    % ensemble_unit_idx = ensemble_unit_idx & is_pure;

    average_ensemble_activity = squeeze(mean(supermouse_combined_valid(ensemble_unit_idx, :, :)))';

    ymax_all = max(average_ensemble_activity, [], "all");
    ymin_all = min(average_ensemble_activity, [], 'all');

    figure
    imagesc(average_ensemble_activity)
    title(sprintf([area_to_plot, '- ensemble %d'], iensemble))

    figure('Position', [0.2 0.2 480 720])
    t = tiledlayout(6, 1);
    for igroup = 1:6
        nexttile
        mean_activity = mean(average_ensemble_activity((igroup-1)*5+1:5*igroup, :));
        sem_activity = sem(average_ensemble_activity((igroup-1)*5+1:5*igroup, :));
        shadedErrorBar(1:50, mean_activity, sem_activity)
        if igroup < 6
            xticks([])
        end
        % yticks([])
        % --- draw patches ---
        ylim([ymin_all, ymax_all])
        yl = ylim;         % returns [0 1] at this point, but we set a real one later
        if yl(1) == yl(2), yl = [0 1]; end   % paranoia for empty axes
        yPatch = [yl(1) yl(1) yl(2) yl(2)];

        if visual_start_idx <= visual_end_idx
            xVis = [visual_start_idx-0.5 visual_end_idx+0.5 visual_end_idx+0.5 visual_start_idx-0.5];
            patch(xVis, yPatch, [0 0.447 0.741], 'EdgeColor','none', 'FaceAlpha',0.15);  % blue
        end
        if reward_start_idx <= reward_end_idx
            xRew = [reward_start_idx-0.5 reward_end_idx+0.5 reward_end_idx+0.5 reward_start_idx-0.5];
            patch(xRew, yPatch, [0.850 0.325 0.098], 'EdgeColor','none', 'FaceAlpha',0.15); % red
        end

        % tighten axes & bring lines to front
        yl = ylim;              % update real limits
        set(findobj(gca,'Type','patch'),'YData',[yl(1) yl(1) yl(2) yl(2)]); % stretch patches

    end
    title(t, sprintf([area_to_plot, '- ensemble %d'], iensemble))
    xlabel(t, 'spatial bin')
    ylabel(t, 'FR')

end
%% Group by lick error

animal_ensemble_data = cell(numel(unique(labels_valid.mouse_labels)), total_ensembles);

ensemble_activity_good_post = cell(1, total_ensembles);
ensemble_activity_bad_pre = cell(1, total_ensembles);
ensemble_activity_bad_post = cell(1, total_ensembles);

for ianimal = unique(labels_valid.mouse_labels)'
    current_data = task_data_to_pass(ianimal).spatial_binned_fr_all;
    [n_units, n_bins, n_trials] = size(current_data);
    all_idx_to_keep = false(1, n_units);

    is_dms = task_data_to_pass(ianimal).is_dms;
    is_dls = task_data_to_pass(ianimal).is_dls;
    is_acc = task_data_to_pass(ianimal).is_acc;
    is_v1 = task_data_to_pass(ianimal).is_v1;

    if any(contains(cfg.areas_to_include, 'DMS'))
        all_idx_to_keep(is_dms) = true;
    end

    if any(contains(cfg.areas_to_include, 'DLS'))
        all_idx_to_keep(is_dls) = true;
    end

    if any(contains(cfg.areas_to_include, 'ACC'))
        all_idx_to_keep(is_acc) = true;
    end

    if any(contains(cfg.areas_to_include, 'V1'))
        all_idx_to_keep(is_v1) = true;
    end


    current_data_keep = current_data(all_idx_to_keep, :, :);

    lick_errors = task_data_to_pass(ianimal).zscored_lick_errors;

    pre_learning_trials = false(1, n_trials);
    pre_learning_trials(1:task_lps_to_pass{ianimal}-1) = true;

    post_learning_trials = false(1, n_trials);
    post_learning_trials(task_lps_to_pass{ianimal}:end) = true;
    
    good_trials_post = (lick_errors <= -2) & post_learning_trials;
    bad_trials_pre = (lick_errors > -1) & pre_learning_trials;
    bad_trials_post = (lick_errors > -1) & post_learning_trials;

    for iensemble = 1:total_ensembles
        ensemble_assignments_animal = ensemble_assignments(labels_valid.mouse_labels == ianimal);
        % ensemble_assignments_animal = ensemble_assignments(labels_valid.mouse_labels == ianimal) & is_pure(labels_valid.mouse_labels == ianimal);
        animal_ensemble_data{ianimal, iensemble} = current_data_keep(ensemble_assignments_animal == iensemble, :, :);

        ensemble_activity_good_post{iensemble}(ianimal, :) = mean(animal_ensemble_data{ianimal, iensemble}(:, :, good_trials_post), [1, 3], 'omitmissing');
        ensemble_activity_bad_pre{iensemble}(ianimal, :) = mean(animal_ensemble_data{ianimal, iensemble}(:, :, bad_trials_pre), [1, 3], 'omitmissing');
        ensemble_activity_bad_post{iensemble}(ianimal, :) = mean(animal_ensemble_data{ianimal, iensemble}(:, :, bad_trials_post), [1, 3], 'omitmissing');

    end
end

figure('Position', [100, 100, 480, 720])
t = tiledlayout(total_ensembles, 1, "TileSpacing", "compact");
for iensemble = 1:total_ensembles
    nexttile
    hold on
    h = shadedErrorBar(1:n_bins, mean(ensemble_activity_good_post{iensemble}, 'omitmissing'), sem(ensemble_activity_good_post{iensemble}), 'lineprops', {'Color', 'g'});
    g = shadedErrorBar(1:n_bins, mean(ensemble_activity_bad_pre{iensemble}, 'omitmissing'), sem(ensemble_activity_good_post{iensemble}), 'lineprops', {'Color', 'r'});
    k = shadedErrorBar(1:n_bins, mean(ensemble_activity_bad_post{iensemble}, 'omitmissing'), sem(ensemble_activity_good_post{iensemble}), 'lineprops', {'Color', 'm'});
    
    if iensemble < total_ensembles
        xticks([])
    end
    if iensemble == total_ensembles
        legend([h.mainLine, g.mainLine, k.mainLine], {'good', 'bad pre', 'bad post'})
    end
    title(sprintf('ensemble %d', iensemble))
end
xlabel(t, 'spatial bin')
ylabel(t, 'firing rate')

%% Plot licks and velocity in aligned trials

all_licks = nan(numel(unique(labels_valid.mouse_labels)), n_bins, 30);
all_velocity = nan(numel(unique(labels_valid.mouse_labels)), n_bins, 30);

for ianimal = unique(labels_valid.mouse_labels)'
    licks_temp = task_data_to_pass(ianimal).spatial_binned_data.licks;
    durations_temp = task_data_to_pass(ianimal).spatial_binned_data.durations;
    velocity_temp = cfg.plot.zone_params.bin_size*1.25./durations_temp;
    lick_rate = licks_temp./durations_temp;
    learning_point = learning_points_task{ianimal};
    all_licks(ianimal, :, 1:10) = lick_rate(1:10, :)';
    all_licks(ianimal, :, 11:20) = lick_rate(learning_point-9:learning_point, :)';
    all_licks(ianimal, :, 21:30) = lick_rate(learning_point+1:learning_point+10, :)';

    all_velocity(ianimal, :, 1:10) = velocity_temp(1:10, :)';
    all_velocity(ianimal, :, 11:20) = velocity_temp(learning_point-9:learning_point, :)';
    all_velocity(ianimal, :, 21:30) = velocity_temp(learning_point+1:learning_point+10, :)';
end

figure
imagesc(squeeze(mean(all_licks, 'omitmissing'))')
colorbar
xlabel('trial')
ylabel('lick rate')

figure
imagesc(squeeze(mean(all_velocity, 'omitmissing'))')
colorbar
xlabel('trial')
ylabel('velocity')

mean_licks = squeeze(mean(all_licks, 'omitmissing'))';
figure
hold on
h = shadedErrorBar(1:n_bins, mean(mean_licks(1:10, :)), sem(mean_licks(1:10, :)), 'lineprops', {'Color', 'b'});
g = shadedErrorBar(1:n_bins, mean(mean_licks(11:20, :)), sem(mean_licks(11:20, :)),  'lineprops', {'Color', 'g'});
k = shadedErrorBar(1:n_bins, mean(mean_licks(21:30, :)), sem(mean_licks(21:30, :)),  'lineprops', {'Color', 'r'});
legend([h.mainLine, g.mainLine, k.mainLine], {'Naive', 'Intermediate', 'Expert'})
xlabel('spatial bin')
ylabel('lick rate')

mean_velocity = squeeze(mean(all_velocity, 'omitmissing'))';
figure
hold on
h = shadedErrorBar(1:n_bins, mean(mean_velocity(1:10, :)), sem(mean_velocity(1:10, :)), 'lineprops', {'Color', 'b'});
g = shadedErrorBar(1:n_bins, mean(mean_velocity(11:20, :)), sem(mean_velocity(11:20, :)),  'lineprops', {'Color', 'g'});
k = shadedErrorBar(1:n_bins, mean(mean_velocity(21:30, :)), sem(mean_velocity(21:30, :)),  'lineprops', {'Color', 'r'});
legend([h.mainLine, g.mainLine, k.mainLine], {'Naive', 'Intermediate', 'Expert'})
xlabel('spatial bin')
ylabel('velocity')

ymin_all = min(mean_licks, [], 'all');
ymax_all = 0.9*max(mean_licks, [], 'all');
figure('Position', [0.2, 0.2, 480 720])
t = tiledlayout(6, 1);
for igroup = 1:6
        nexttile
        mean_licks_group = mean(mean_licks((igroup-1)*5+1:5*igroup, :));
        sem_licks_group = sem(mean_licks((igroup-1)*5+1:5*igroup, :));
        shadedErrorBar(1:n_bins, mean_licks_group, sem_licks_group)

        % --- draw patches ---
        ylim([ymin_all, ymax_all])
        yl = ylim;         % returns [0 1] at this point, but we set a real one later
        if yl(1) == yl(2), yl = [0 1]; end   % paranoia for empty axes
        yPatch = [yl(1) yl(1) yl(2) yl(2)];

        if visual_start_idx <= visual_end_idx
            xVis = [visual_start_idx-0.5 visual_end_idx+0.5 visual_end_idx+0.5 visual_start_idx-0.5];
            patch(xVis, yPatch, [0 0.447 0.741], 'EdgeColor','none', 'FaceAlpha',0.15);  % blue
        end
        if reward_start_idx <= reward_end_idx
            xRew = [reward_start_idx-0.5 reward_end_idx+0.5 reward_end_idx+0.5 reward_start_idx-0.5];
            patch(xRew, yPatch, [0.850 0.325 0.098], 'EdgeColor','none', 'FaceAlpha',0.15); % red
        end

        % tighten axes & bring lines to front
        yl = ylim;              % update real limits
        set(findobj(gca,'Type','patch'),'YData',[yl(1) yl(1) yl(2) yl(2)]); % stretch patches

        if igroup < 6
            xticks([])
        end
end
linkaxes
xlabel(t, 'spatial bin')
ylabel(t, 'lick rate')

ymin_all = min(mean_velocity, [], 'all');
ymax_all = 0.8*max(mean_velocity, [], 'all');

figure('Position', [0.2, 0.2, 480 720])
t = tiledlayout(6, 1);
for igroup = 1:6
        nexttile
        mean_velocity_group = mean(mean_velocity((igroup-1)*5+1:5*igroup, :));
        sem_velocity_group = sem(mean_velocity((igroup-1)*5+1:5*igroup, :));
        shadedErrorBar(1:n_bins, mean_velocity_group, sem_velocity_group)

                % --- draw patches ---
        ylim([ymin_all, ymax_all])
        yl = ylim;         % returns [0 1] at this point, but we set a real one later
        if yl(1) == yl(2), yl = [0 1]; end   % paranoia for empty axes
        yPatch = [yl(1) yl(1) yl(2) yl(2)];

        if visual_start_idx <= visual_end_idx
            xVis = [visual_start_idx-0.5 visual_end_idx+0.5 visual_end_idx+0.5 visual_start_idx-0.5];
            patch(xVis, yPatch, [0 0.447 0.741], 'EdgeColor','none', 'FaceAlpha',0.15);  % blue
        end
        if reward_start_idx <= reward_end_idx
            xRew = [reward_start_idx-0.5 reward_end_idx+0.5 reward_end_idx+0.5 reward_start_idx-0.5];
            patch(xRew, yPatch, [0.850 0.325 0.098], 'EdgeColor','none', 'FaceAlpha',0.15); % red
        end

        % tighten axes & bring lines to front
        yl = ylim;              % update real limits
        set(findobj(gca,'Type','patch'),'YData',[yl(1) yl(1) yl(2) yl(2)]); % stretch patches

        if igroup < 6
            xticks([])
        end
end
linkaxes
xlabel(t, 'spatial bin')
ylabel(t, 'velocity')

% Correlate with activity
for iensemble = 1:6
    ensemble_unit_idx = ensemble_assignments == iensemble;

    average_ensemble_activity = squeeze(mean(supermouse_combined_valid(ensemble_unit_idx, :, :)))';

    figure('Position', [100 100 480 720])
    t = tiledlayout(6, 2, 'TileSpacing', 'compact');
    for igroup = 1:6
        mean_velocity_group = mean(mean_velocity((igroup-1)*5+1:5*igroup, :));
        mean_licks_group = mean(mean_licks((igroup-1)*5+1:5*igroup, :));

        mean_activity = mean(average_ensemble_activity((igroup-1)*5+1:5*igroup, :));
        [lick_rho, lick_pval] = corr(mean_activity', mean_licks_group');
        [vel_rho, vel_pval] = corr(mean_activity', mean_velocity_group');
        nexttile
        scatter(mean_activity', mean_licks_group')
        lsline
        title(sprintf('rho = %.2f - pval = %.3f', lick_rho, lick_pval))
        xlabel('activity')
        ylabel('licks')
        nexttile
        scatter(mean_activity', mean_velocity_group')
        lsline
        title(sprintf('rho = %.2f - pval = %.3f', vel_rho, vel_pval))
        xlabel('activity')
        ylabel('velocity')
    end
    title(t, sprintf('ensemble %d', iensemble))
end

%% Disengagement

all_licks_dis = nan(numel(unique(labels_valid.mouse_labels)), n_bins, 20);
all_velocity_dis = nan(numel(unique(labels_valid.mouse_labels)), n_bins, 20);

green   = [0.466 0.674 0.188];
magenta = [0.494 0.184 0.556];

n_animals = max(unique(labels_valid.mouse_labels)');
disengaged_aligned_data = cell(n_animals, total_ensembles);


for ianimal = unique(labels_valid.mouse_labels)'
    current_data = task_data_to_pass(ianimal).spatial_binned_fr_all;
    [n_units, n_bins, n_trials] = size(current_data);
    all_idx_to_keep = false(1, n_units);

    is_dms = task_data_to_pass(ianimal).is_dms;
    is_dls = task_data_to_pass(ianimal).is_dls;
    is_acc = task_data_to_pass(ianimal).is_acc;
    is_v1 = task_data_to_pass(ianimal).is_v1;

    if any(contains(cfg.areas_to_include, 'DMS'))
        all_idx_to_keep(is_dms) = true;
    end

    if any(contains(cfg.areas_to_include, 'DLS'))
        all_idx_to_keep(is_dls) = true;
    end

    if any(contains(cfg.areas_to_include, 'ACC'))
        all_idx_to_keep(is_acc) = true;
    end

    if any(contains(cfg.areas_to_include, 'V1'))
        all_idx_to_keep(is_v1) = true;
    end


    current_data_keep = current_data(all_idx_to_keep, :, :);

    if ~isnan(task_data_to_pass(ianimal).change_point_mean)
        ensemble_assignments_animal = ensemble_assignments(labels_valid.mouse_labels == ianimal);
        for iensemble = 1:total_ensembles
            engaged_aligned_data{ianimal, iensemble}(:, :, 1:10) = current_data_keep(ensemble_assignments_animal == iensemble, :, 1:10);
            engaged_aligned_data{ianimal, iensemble}(:, :, 11:20) = current_data_keep(ensemble_assignments_animal == iensemble, :, learning_points_task{ianimal}-10:learning_points_task{ianimal}-1);
            engaged_aligned_data{ianimal, iensemble}(:, :, 21:30) = current_data_keep(ensemble_assignments_animal == iensemble, :, learning_points_task{ianimal}+1:learning_points_task{ianimal}+10);
            disengaged_aligned_data{ianimal, iensemble} = current_data_keep(ensemble_assignments_animal == iensemble, :, task_data_to_pass(ianimal).change_point_mean-9:task_data_to_pass(ianimal).change_point_mean+10);
        end

        licks_temp = task_data_to_pass(ianimal).spatial_binned_data.licks;
        durations_temp = task_data_to_pass(ianimal).spatial_binned_data.durations;
        velocity_temp = cfg.plot.zone_params.bin_size*1.25./durations_temp;
        lick_rate = licks_temp./durations_temp;

        all_licks_dis(ianimal, :, :) = lick_rate(task_data_to_pass(ianimal).change_point_mean-9:task_data_to_pass(ianimal).change_point_mean+10, :)';
    
        all_velocity_dis(ianimal, :, :) = velocity_temp(task_data_to_pass(ianimal).change_point_mean-9:task_data_to_pass(ianimal).change_point_mean+10, :)';
    end


end

figure
t = tiledlayout('flow', 'TileSpacing', 'compact');
for iensemble = 1:total_ensembles
    all_disengaged_ensemble = cat(1, disengaged_aligned_data{:, iensemble});

    nexttile
    imagesc(squeeze(mean(all_disengaged_ensemble, 'omitmissing'))')
    yticks(0:5:20)
    yticklabels(-10:5:10)
    title(sprintf('ensemble %d', iensemble))
end
ylabel(t, 'trials to disengament')
xlabel(t, 'spatial bin')

figure
t = tiledlayout('flow', 'TileSpacing', 'compact');   
imagesc(squeeze(mean(all_licks_dis, 'omitmissing'))')
yticks(0:5:20)
yticklabels(-10:5:10)
title('licks')
ylabel('trials to disengament')
xlabel(t, 'spatial bin')

figure
average_licks_dis = squeeze(mean(all_licks_dis, 'omitmissing'));
hold on
h = shadedErrorBar(1:n_bins, mean(average_licks_dis(:,1:10),2,'omitmissing'), sem(average_licks_dis(:,1:10),2), 'lineprops',{'Color',green});
g = shadedErrorBar(1:n_bins, mean(average_licks_dis(:,11:end),2,'omitmissing'), sem(average_licks_dis(:,11:end),2), 'lineprops',{'Color',magenta});
legend([h.mainLine g.mainLine],{'pre-disengagement' 'post-disengagement'})
xlabel('spatial bin')
ylabel('lick rate')

figure
t = tiledlayout('flow', 'TileSpacing', 'compact');   
imagesc(squeeze(mean(all_velocity_dis, 'omitmissing'))')
yticks(0:5:20)
yticklabels(-10:5:10)
title('velocity')
ylabel('trials to disengament')
xlabel(t, 'spatial bin')

figure
average_vel_dis = squeeze(mean(all_velocity_dis, 'omitmissing'));
hold on
h = shadedErrorBar(1:n_bins, mean(average_vel_dis(:,1:10),2,'omitmissing'), sem(average_vel_dis(:,1:10),2), 'lineprops',{'Color',green});
g = shadedErrorBar(1:n_bins, mean(average_vel_dis(:,11:end),2,'omitmissing'), sem(average_vel_dis(:,11:end),2), 'lineprops',{'Color',magenta});
legend([h.mainLine g.mainLine],{'pre-disengagement' 'post-disengagement'})
xlabel('spatial bin')
ylabel('velocity')

% Parameters for patches
bin_size        = cfg.plot.zone_params.bin_size;
corridor_end_au = cfg.plot.zone_params.corridor_end_au;
bin_edges       = 0:bin_size:corridor_end_au;
if bin_edges(end) <= corridor_end_au, bin_edges(end+1)=bin_edges(end)+bin_size; end
bin_centres     = bin_edges(1:end-1)+diff(bin_edges)/2;

% visual_start_idx  = find(bin_centres >= cfg.plot.zone_params.visual_start_au ,1,'first');
% reward_start_idx  = find(bin_centres >= cfg.plot.zone_params.reward_start_au ,1,'first');
% reward_end_idx    = find(bin_centres <= cfg.plot.zone_params.reward_end_au  ,1,'last');
% visual_end_idx    = reward_start_idx-1;                         % visual ends where reward starts

figure
t = tiledlayout('flow','TileSpacing','compact');

for iensemble = 1:total_ensembles
    nexttile
    hold on

    all_disengaged_ensemble = cat(1, disengaged_aligned_data{:,iensemble});
    average_ensemble_activity = squeeze(mean(all_disengaged_ensemble,'omitmissing'));

    h = shadedErrorBar(1:n_bins, ...
        mean(average_ensemble_activity(:,1:10),2,'omitmissing'), ...
        sem(average_ensemble_activity(:,1:10),2), ...
        'lineprops',{'Color',green});

    g = shadedErrorBar(1:n_bins, ...
        mean(average_ensemble_activity(:,11:end),2,'omitmissing'), ...
        sem(average_ensemble_activity(:,11:end),2), ...
        'lineprops',{'Color',magenta});

    axis tight

    % --- draw patches ---
    yl = ylim;         % returns [0 1] at this point, but we set a real one later
    if yl(1) == yl(2), yl = [0 1]; end   % paranoia for empty axes
    yPatch = [yl(1) yl(1) yl(2) yl(2)];

    if visual_start_idx <= visual_end_idx
        xVis = [visual_start_idx-0.5 visual_end_idx+0.5 visual_end_idx+0.5 visual_start_idx-0.5];
        patch(xVis, yPatch, [0 0.447 0.741], 'EdgeColor','none', 'FaceAlpha',0.15);  % blue
    end
    if reward_start_idx <= reward_end_idx
        xRew = [reward_start_idx-0.5 reward_end_idx+0.5 reward_end_idx+0.5 reward_start_idx-0.5];
        patch(xRew, yPatch, [0.850 0.325 0.098], 'EdgeColor','none', 'FaceAlpha',0.15); % red
    end

    % tighten axes & bring lines to front
    yl = ylim;              % update real limits
    set(findobj(gca,'Type','patch'),'YData',[yl(1) yl(1) yl(2) yl(2)]); % stretch patches
    uistack(h.mainLine,'top'); uistack(g.mainLine,'top');

    title(sprintf('ensemble %d',iensemble))
    if iensemble==6
        legend([h.mainLine g.mainLine],{'pre-disengagement' 'post-disengagement'})
    end
end

ylabel(t,'FR')
xlabel(t,'spatial bin')

%% Naive/expert/disengaged

figure
t = tiledlayout('flow');

for iensemble = 1:total_ensembles
    nexttile
    title(sprintf('ensemble %d', iensemble))
    ensemble_unit_idx = ensemble_assignments == iensemble;
    % ensemble_unit_idx = ensemble_unit_idx & is_pure;

    average_ensemble_activity = squeeze(mean(supermouse_combined_valid(ensemble_unit_idx, :, :)))';

    all_disengaged_ensemble = cat(1, disengaged_aligned_data{:, iensemble});

    hold on
    plot(mean(average_ensemble_activity(1:10, :)), 'Color', cfg.plot.colors.epoch_early, LineWidth=1)

    plot(mean(average_ensemble_activity(21:30, :)), 'Color', cfg.plot.colors.epoch_expert, LineWidth=1)

    plot(squeeze(mean(all_disengaged_ensemble(:, :, 11:20), [1, 3], 'omitnan'))', 'Color', magenta, LineWidth=1)
    axis tight
end

legend({'naive', 'expert', 'disengaged'}, 'Location', 'best')
%% Decoding ablation analysis

% After Run_TCA_pipeline finished and you have:
%   supermouse_combined_valid   (neurons × bins × trials)
%   ensemble_assignments        (neurons × 1)

results = decode_ensemble_ablation(supermouse_combined_valid,...
                                   ensemble_assignments);
fprintf('Baseline RMSE  = %.2f\n', results.baseline.rmse);
fprintf('Shuffle  RMSE  = %.2f ± %.2f\n', ...
        results.shuffle.rmse_mean, results.shuffle.rmse_sem);

%% Plot decoding results
figure
hold on
plot(results.baseline.abs_error_trial, 'k', 'LineWidth', 2)
shadedErrorBar(1:30, results.shuffle.abs_error_trial_mean, results.shuffle.abs_error_trial_sem, 'lineprops', {'Color', 'r', 'LineWidth', 1})
for iensemble = 1:total_ensembles
    plot(results.knockout(iensemble).abs_error_trial)
end
title('KO')
xlabel('trial')
ylabel('abs decoding error')

figure
hold on
plot(results.baseline.abs_error_trial, 'k', 'LineWidth', 2)
shadedErrorBar(1:30, results.shuffle.abs_error_trial_mean, results.shuffle.abs_error_trial_sem, 'lineprops', {'Color', 'r', 'LineWidth', 1})
for iensemble = 1:total_ensembles
    plot(results.only(iensemble).abs_error_trial)
end
title('single')
xlabel('trial')
ylabel('abs decoding error')


figure
hold on
plot(results.baseline.abs_error_space, 'k', 'LineWidth', 2)
shadedErrorBar(1:50, results.shuffle.abs_error_space_mean, results.shuffle.abs_error_space_sem, 'lineprops', {'Color', 'r', 'LineWidth', 1})
for iensemble = 1:total_ensembles
    plot(results.knockout(iensemble).abs_error_space)
end
title('KO')
xlabel('spatial bin')
ylabel('abs decoding error')


figure
hold on
plot(results.baseline.abs_error_space, 'k', 'LineWidth', 2)
shadedErrorBar(1:50, results.shuffle.abs_error_space_mean, results.shuffle.abs_error_space_sem, 'lineprops', {'Color', 'r', 'LineWidth', 1})
for iensemble = 1:total_ensembles
    plot(results.only(iensemble).abs_error_space)
end
title('single')
xlabel('spatial bin')
ylabel('abs decoding error')

figure
boxplot([results.baseline.abs_error_trial, results.shuffle.abs_error_trial_mean', [results.knockout(:).abs_error_trial]])
title('KO')
xticklabels([{'full', 'shuffle'}, strsplit(num2str(1:total_ensembles))])
xlabel('ensemble')
ylabel('abs decoding error')

figure
boxplot([results.knockout(:).abs_error_trial] - results.baseline.abs_error_trial)
title('KO')
delta_errors = [results.knockout(:).abs_error_trial] - results.baseline.abs_error_trial;
[h, pval] = ttest(delta_errors);
sigstar({[1, 1], 2*[1, 1], 3*[1, 1], 4*[1, 1], 5*[1, 1]}, pval)
yline(0, 'k--')
xlabel('ensemble')
ylabel('\Delta decoding error')

figure
boxplot([results.baseline.abs_error_trial, results.shuffle.abs_error_trial_mean', [results.only(:).abs_error_trial]])
title('single')
xticklabels([{'full', 'shuffle'}, strsplit(num2str(1:total_ensembles))])
xlabel('ensemble')
ylabel('abs decoding error')


%% Reliability/Stability of ensembles

win = 5;                                         % window size (odd)
halfwin = floor(win/2);
[nNeurons, nBins, nTrials] = size(supermouse_combined_valid);
ensembles = unique(ensemble_assignments(:)');
E = numel(ensembles);

unit_r = nan(nNeurons, nTrials);
for n = 1:nNeurons
    for t = 1:nTrials
        idxWin = max(1, t-halfwin) : min(nTrials, t+halfwin);
        L = numel(idxWin);
        if L < 2, continue, end

        data_block = squeeze(supermouse_combined_valid(n, :, idxWin));
        R = corr(data_block);

        % Extract the upper triangle of the correlation matrix (excluding the diagonal)
        upper_triangle = triu(R, 1);
        upper_vals = upper_triangle(upper_triangle ~= 0);

        % Compute the average correlation
        unit_r(n, t) = mean(upper_vals, 'omitnan');

        % dat = squeeze(supermouse_combined_valid(n,:,idxWin));   % bins × L
        % p = 1; rvals = nan(L*(L-1)/2,1);
        % for i = 1:L-1
        %     for j = i+1:L
        %         rvals(p) = corr(dat(:,i), dat(:,j), 'rows','pairwise');
        %         p = p + 1;
        %     end
        % end
        % unit_r(n,t) = mean(rvals, 'omitnan');
    end
end

ens_r_unitavg = nan(E, nTrials);
for k = 1:E
    idx = ensemble_assignments == ensembles(k);
    ens_r_unitavg(k,:) = mean(unit_r(idx,:), 1, 'omitnan');
    ens_r_unitsem(k,:) = mean(unit_r(idx,:), 1, 'omitnan');
end

figure('Name','Unit‑level trial correlation'); hold on
for k = 1:E
    plot(ens_r_unitavg(k,:), 'DisplayName', sprintf('ensemble%d', ensembles(k)), 'LineWidth', 1.2);
end
xlabel('trial'); ylabel('mean r (unit)'); legend show

%% 3. PAIRWISE ensemble correlation *per trial*  (average‑activity vector)
%    Returns pair_r(k1,k2,t) and plots heat‑map over trials.
pair_r = nan(E,E,nTrials);
for k1 = 1:E
    idx1 = ensemble_assignments==ensembles(k1);
    if ~any(idx1), continue, end
    A = squeeze(mean(supermouse_combined_valid(idx1,:,:),1,'omitnan'));  % bins × trials
    for k2 = k1+1:E
        idx2 = ensemble_assignments==ensembles(k2);
        if ~any(idx2), continue, end
        B = squeeze(mean(supermouse_combined_valid(idx2,:,:),1,'omitnan'));
        for t = 1:nTrials
            pair_r(k1,k2,t) = corr(A(:,t),B(:,t),'rows','pairwise');
        end
        pair_r(k2,k1,:) = pair_r(k1,k2,:);
    end
end

% Plot: heat‑map (pairs × trials)
pairs = find(triu(true(E),1));
[pIdx_row,pIdx_col] = ind2sub([E E],pairs);
pair_matrix = squeeze(pair_r(:,:,1)); %# placeholder to size figure
pairTime = nan(numel(pairs), nTrials);
for p = 1:numel(pairs)
    pairTime(p,:) = squeeze(pair_r(pIdx_row(p), pIdx_col(p), :));
end
figure('Name','Pairwise ensemble corr vs trial');
imagesc(pairTime); colorbar
yticks(1:numel(pairs));
% yticklabels(compose('E%d–E%d',ensembles(pIdx_row),ensembles(pIdx_col)));
xlabel('Trial'); title('Pairwise corr (avg activity)');

% Also compute across‑trial mean for back‑compatibility
a = pair_r; a(isnan(a)) = 0; pair_mean = squeeze(sum(a,3)) ./ sum(~isnan(pair_r),3);

figure
plot(pairTime')

%% 4. ANIMATED NETWORK  (edge thickness = |corr|, colour = sign)
%    Saves to GIF in current folder.
gif_name = 'ensemble_corr_evolution.gif';
frame_delay = 0.3;               % seconds between frames
max_w     = max(abs(pair_r(:)),[],'omitnan');

% precompute node positions (circle)
ang = linspace(0,2*pi,E+1)'; ang(end)=[];
xy = [cos(ang) sin(ang)];

f = figure('Name','Ensemble correlation network'); clf;
for t = 1:nTrials
    W = pair_r(:,:,t);
    W(isnan(W)) = 0;                    % treat NaN as 0 weight
    G = graph(W, 'upper', 'omitselfloops');  % undirected, no self-loops
    labels = compose('E%d', ensembles);      % labels for plotting
    p = plot(G, 'XData',xy(:,1),'YData',xy(:,2), ...
             'NodeLabel',labels, ...
             'MarkerSize',8);
    % scale edges
    LWidths = 1 + 4*abs(G.Edges.Weight)/max_w; % 1–5 px
    p.LineWidth = LWidths;
    % colour: positive = red, negative = blue, zero = gray
    wcol = zeros(numedges(G),3);
    w = G.Edges.Weight;
    wcol(w<0,:) = repmat([0 0 1], sum(w<0),1);   % blue
    wcol(w>0,:) = repmat([1 0 0], sum(w>0),1);   % red
    wcol(w==0,:)= repmat([0.6 0.6 0.6], sum(w==0),1);
    p.EdgeColor = wcol;
    title(sprintf('Trial %d',t)); axis off equal; drawnow;
    % capture frame
    fr = getframe(f);
    [im,map] = rgb2ind(fr.cdata,256);
    if t==1
        imwrite(im,map,gif_name,'gif','LoopCount',Inf,'DelayTime',frame_delay);
    else
        imwrite(im,map,gif_name,'gif','WriteMode','append','DelayTime',frame_delay);
    end
end
fprintf('Animated GIF saved to %s\n', fullfile(pwd,gif_name));
%% ================= PCA on State-Space Trajectories by Ensemble ==============================
fprintf('--- Plotting average trajectory in PCA space for each ensemble ---\n');

% --- Main Configuration ---
% Ensure there are enough trials for the defined epochs (naive, intermediate, expert)
if size(supermouse_combined_valid, 3) < 30
    warning('PCA trajectory plot requires at least 30 trials. Skipping ensemble PCA.');
else
    total_ensembles = max(ensemble_assignments);
    
    % --- Create Figure and Tiled Layout ---
    figure('Name', 'Average Trajectory by Ensemble in Separate PCA Spaces', 'Position', [50, 50, 1200, 800]);
    t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'normal');
    
    % Define epochs and their corresponding colors and names from cfg
    epoch_trials = {1:10, 11:20, 21:30};
    epoch_names = cfg.plot.epoch_names;
    epoch_colors = {cfg.plot.colors.epoch_early, cfg.plot.colors.epoch_middle, cfg.plot.colors.epoch_expert};
    
    % --- Loop Through Each Ensemble ---
    for i_ensemble = 1:total_ensembles
        nexttile;
        hold on;
        
        % --- 1. Select Data for the Current Ensemble ---
        ensemble_unit_idx = (ensemble_assignments == i_ensemble);
        n_neurons_ens = sum(ensemble_unit_idx);
        
        % Check if there are enough neurons to perform PCA (we need at least 4 for 3 PCs)
        if n_neurons_ens < 4
            title(sprintf('Ensemble %d (N=%d, too few neurons)', i_ensemble, n_neurons_ens));
            text(0.5, 0.5, 'Skipped', 'HorizontalAlignment', 'center');
            axis off;
            hold off;
            continue; % Skip to the next ensemble
        end
        
        ensemble_tensor = supermouse_combined_valid(ensemble_unit_idx, :, :);
        [~, n_bins, n_trials] = size(ensemble_tensor);
        
        % --- 2. Perform PCA on this Ensemble's Data ---
        % Reshape from [Neurons x Bins x Trials] to [(Bins * Trials) x Neurons]
        data_for_pca = reshape(permute(ensemble_tensor, [2 3 1]), [n_bins * n_trials, n_neurons_ens]);
        
        % Run PCA. 'coeff' are the new axes, 'score' is the projected data.
        [coeff, score, ~, ~, explained] = pca(data_for_pca);
        
        % --- 3. Calculate Epoch-Averaged Trajectories in PC Space ---
        % Reshape the projected data (first 3 PCs) back to a trial structure
        score_3d = score(:, 1:3);
        trajectories_in_pc_space = reshape(score_3d, [n_bins, n_trials, 3]);
        
        avg_trajectories = cell(1, numel(epoch_trials));
        plot_handles = gobjects(numel(epoch_trials), 1);
        
        for i_epoch = 1:numel(epoch_trials)
            % Average trajectories across trials for the current epoch
            mean_traj = mean(trajectories_in_pc_space(:, epoch_trials{i_epoch}, :), 2);
            avg_trajectories{i_epoch} = squeeze(mean_traj); % Result is [Bins x 3]
            
            % --- 4. Plot the Trajectory for this Epoch ---
            traj = smoothdata(avg_trajectories{i_epoch}, 1, 'movmean', 10);
            plot_handles(i_epoch) = plot3(traj(:,1), traj(:,2), traj(:,3), ...
                'Color', epoch_colors{i_epoch}, 'LineWidth', 2);
                
            % Add markers for start (circle) and end (square) points
            scatter3(traj(1,1), traj(1,2), traj(1,3), 50, epoch_colors{i_epoch}, 'o', 'filled');
            scatter3(traj(end,1), traj(end,2), traj(end,3), 50, epoch_colors{i_epoch}, 's', 'filled');
        end
        
        hold off;
        
        % --- 5. Format Subplot ---
        title(sprintf('Ensemble %d (N=%d)', i_ensemble, n_neurons_ens));
        xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
        ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
        zlabel(sprintf('PC3 (%.1f%%)', explained(3)));
        grid on;
        axis tight;
        view(35, 25); % Set consistent viewing angle
    end
    
    % --- Add Shared Legend to the Figure ---
    lg = legend(plot_handles, epoch_names, 'FontSize', 10);
    lg.Layout.Tile = 'east'; % Place legend in its own tile on the side
    
    fprintf('  Displayed average trajectory plot for each ensemble.\n\n');
end