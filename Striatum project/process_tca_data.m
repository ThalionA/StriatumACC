function [clusters, cluster_assignments, normalized_spatial_patterns, pattern_labels, all_trial_patterns] = process_tca_data(preprocessed_data, n_animals, highest_common_n_trials, areas)
    % Process the data to compute clusters and normalized patterns without plotting
    % Now accounts for multiple areas per animal.

    % Extract z-scored lick errors for all animals
    zscored_lick_errors_all = {preprocessed_data(:).zscored_lick_errors};

    % Initialize indices
    first_idx = cell(1, n_animals);
    temp_idx = false(1, highest_common_n_trials);
    temp_idx(1:2) = true;
    [first_idx{:}] = deal(temp_idx);

    precise_idx = cellfun(@(x) x<=-2, zscored_lick_errors_all, 'UniformOutput', false);
    random_idx = cellfun(@(x) x>-1, zscored_lick_errors_all, 'UniformOutput', false);
    for ianimal = 1:n_animals
        precise_idx{ianimal}(1:2) = false;
        random_idx{ianimal}(1:2) = false;
    end

    % Initialize variables
    all_spatial_patterns = [];
    pattern_labels = []; % To keep track of animal, area, and component indices
    all_trial_patterns = [];
    trial_patterns_by_condition = [];
    factor_counter = 0;

    % Loop over animals and areas to collect patterns
    for ianimal = 1:n_animals
        for iarea = 1:numel(areas)
            area = areas{iarea};
            tca_field_name = sprintf('tca_best_mdl_%s', area);
            if ~isempty(preprocessed_data(ianimal).(tca_field_name))
                tca_model = preprocessed_data(ianimal).(tca_field_name);
                n_Factors = size(tca_model.U{2}, 2);

                for iFactor = 1:n_Factors
                    factor_counter = factor_counter + 1;
                    pattern = tca_model.U{2}(:, iFactor);
                    all_spatial_patterns = [all_spatial_patterns, pattern];
                    pattern_labels = [pattern_labels; struct('animal', ianimal, 'area', area, 'factor', iFactor)];

                    trial_pattern = tca_model.U{3}(:, iFactor);
                    trial_pattern = trial_pattern / max(trial_pattern);

                    trial_pattern_first = mean(trial_pattern(first_idx{ianimal}));
                    trial_pattern_precise = mean(trial_pattern(precise_idx{ianimal}));
                    trial_pattern_random = mean(trial_pattern(random_idx{ianimal}));

                    trial_patterns_by_condition = [trial_patterns_by_condition; ...
                        [trial_pattern_first, trial_pattern_precise, trial_pattern_random]];

                    all_trial_patterns = [all_trial_patterns, trial_pattern];
                end
            else
                warning('TCA model for area %s not found in animal %d', area, ianimal);
            end
        end
    end

    % Normalize each spatial pattern to have unit norm
    normalized_spatial_patterns = all_spatial_patterns ./ vecnorm(all_spatial_patterns);

    % Normalize each trial pattern to have unit norm
    normalized_trial_patterns = all_trial_patterns ./ vecnorm(all_trial_patterns);

    % Compute the pairwise Pearson correlation coefficients
    similarity_matrix = corrcoef(normalized_spatial_patterns);

    % Perform hierarchical clustering using the distance matrix
    distance_matrix = 1 - similarity_matrix; % Convert similarity to distance
    Z = linkage(squareform(distance_matrix), 'average');

    % Determine the number of clusters
    num_clusters = 4; % Adjust as needed

    % Assign clusters
    cluster_assignments = cluster(Z, 'maxclust', num_clusters);

    % Initialize a cell array to hold clusters
    clusters = cell(num_clusters, 1);

    for iclust = 1:num_clusters
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