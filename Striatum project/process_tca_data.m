function [clusters, cluster_assignments, normalized_spatial_patterns, pattern_labels, all_trial_patterns] = process_tca_data(preprocessed_data, n_animals, areas, minimum_cluster_num)
    % Process the data to compute clusters and normalized patterns without plotting
    % Now accounts for multiple areas per animal and variable trial numbers.

    if nargin < 4
        minimum_cluster_num = 4;
    end
    % Extract z-scored lick errors for all animals
    zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

    % Initialize variables
    all_spatial_patterns = [];
    pattern_labels = []; % To keep track of animal, area, and component indices
    all_trial_patterns = [];
    trial_patterns_by_condition = [];
    factor_counter = 0;

    % Loop over animals and areas to collect patterns
    for ianimal = 1:n_animals
        n_trials = preprocessed_data(ianimal).n_trials;

        % Initialize indices
        temp_idx = false(1, n_trials);
        temp_idx(1:3) = true; % Handle cases where n_trials < 2
        first_idx = temp_idx;

        zscored_errors = zscored_lick_errors_all{ianimal};
        precise_idx = zscored_errors <= -2;
        random_idx = zscored_errors > -1;

        % Exclude the first two trials from precise and random indices
        exclude_trials = 1:3;
        precise_idx(exclude_trials) = false;
        random_idx(exclude_trials) = false;

        for iarea = 1:numel(areas)
            area = areas{iarea};
            tca_field_name = sprintf('tca_best_mdl_%s', area);

            if isfield(preprocessed_data(ianimal), tca_field_name) && ~isempty(preprocessed_data(ianimal).(tca_field_name))
                tca_model = preprocessed_data(ianimal).(tca_field_name);
                n_Factors = size(tca_model.U{2}, 2);

                for iFactor = 1:n_Factors
                    factor_counter = factor_counter + 1;
                    pattern = tca_model.U{2}(:, iFactor);
                    all_spatial_patterns = [all_spatial_patterns, pattern];
                    pattern_labels = [pattern_labels; struct('animal', ianimal, 'area', area, 'factor', iFactor)];

                    trial_pattern = tca_model.U{3}(:, iFactor);
                    max_trial_pattern = max(trial_pattern);
                    if max_trial_pattern ~= 0
                        trial_pattern = trial_pattern / max_trial_pattern;
                    end

                    trial_pattern_first = mean(trial_pattern(first_idx));
                    trial_pattern_precise = mean(trial_pattern(precise_idx));
                    trial_pattern_random = mean(trial_pattern(random_idx));

                    trial_patterns_by_condition = [trial_patterns_by_condition; ...
                        [trial_pattern_first, trial_pattern_precise, trial_pattern_random]];

                    all_trial_patterns = [all_trial_patterns, {trial_pattern}]; % Store as cell array
                end
            else
                warning('TCA model for area %s not found in animal %d', area, ianimal);
            end
        end
    end

    % Check if any patterns were collected
    if isempty(pattern_labels)
        error('No patterns were collected. Please ensure that TCA models exist for the specified animals and areas.');
    end

    % Normalize each spatial pattern to have unit norm
    normalized_spatial_patterns = all_spatial_patterns ./ vecnorm(all_spatial_patterns);

    % Proceed with hierarchical clustering
    similarity_matrix = corrcoef(normalized_spatial_patterns);
    distance_matrix = 1 - similarity_matrix; % Convert similarity to distance
    Z = linkage(squareform(distance_matrix), 'average');

    % Determine the optimal number of clusters using silhouette analysis
    num_patterns = size(all_spatial_patterns, 2);
    max_clusters = min(6, num_patterns); % Set a reasonable maximum
    silhouette_avg = zeros(max_clusters, 1);

    for k = 2:max_clusters
        % Assign clusters
        cluster_assignments_k = cluster(Z, 'maxclust', k);
        % Compute the silhouette values
        s = silhouette(normalized_spatial_patterns', cluster_assignments_k, 'correlation');
        % Average silhouette score
        silhouette_avg(k) = mean(s);
    end

    % Find the number of clusters that maximize the average silhouette score
    [~, optimal_num_clusters] = max(silhouette_avg(2:end));
    optimal_num_clusters = optimal_num_clusters + 1; % Adjust for indexing

    fprintf('Optimal number of clusters determined by silhouette analysis: %d\n', optimal_num_clusters);

    % Assign clusters using the optimal number
    cluster_assignments = cluster(Z, 'maxclust', optimal_num_clusters);

    % Initialize a cell array to hold clusters
    clusters = cell(optimal_num_clusters, 1);

    for iclust = 1:optimal_num_clusters
        % Find indices of patterns in the current cluster
        cluster_indices = find(cluster_assignments == iclust);
        clusters{iclust} = cluster_indices;

        % Get the animals and areas represented in this cluster
        animals_in_cluster = unique([pattern_labels(cluster_indices).animal]);
        areas_in_cluster = unique({pattern_labels(cluster_indices).area});

        % Display information about the cluster
        fprintf('Cluster %d contains patterns from animals: %s and areas: %s\n', ...
            iclust, num2str(animals_in_cluster'), strjoin(areas_in_cluster, ', '));
    end
end