function plot_tca_results(clusters, cluster_assignments, normalized_spatial_patterns, pattern_labels, all_trial_patterns, reward_zone_start_bins, areas)
    % Plot the dendrogram and cluster patterns
    % Now accounts for multiple areas per animal and variable trial lengths.

    % Compute the linkage for dendrogram
    similarity_matrix = corrcoef(normalized_spatial_patterns);
    distance_matrix = 1 - similarity_matrix;
    Z = linkage(squareform(distance_matrix), 'average');

    % Determine the number of clusters
    num_clusters = max(cluster_assignments);
    cmap = lines(num_clusters); % Color map for clusters

    % Compute the color threshold for dendrogram
    c_thresh = median([Z(end - num_clusters + 2, 3), Z(end - num_clusters + 1, 3)]);

    % Plot the dendrogram with colored clusters
    figure;
    [H, ~, outperm] = dendrogram(Z, 0, 'ColorThreshold', c_thresh);
    ax = gca;
    set(ax, 'ColorOrder', cmap, 'NextPlot', 'replacechildren');
    xlabel('Patterns');
    ylabel('Distance');
    title('Hierarchical Clustering of Spatial Patterns');

    % Create legend for clusters
    hold on;
    hLegend = gobjects(num_clusters, 1);
    leg_entries = cell(num_clusters, 1);
    for iclust = 1:num_clusters
        hLegend(iclust) = plot(NaN, NaN, '-', 'Color', cmap(iclust, :), 'LineWidth', 2);
        leg_entries{iclust} = sprintf('Cluster %d', iclust);
    end
    hold off;
    legend(hLegend, leg_entries, 'Location', 'best');

    % Visualize clusters - Spatial Patterns
    figure;
    t = tiledlayout(numel(areas), num_clusters, 'TileSpacing', 'compact', 'Padding', 'compact');

    for iarea = 1:numel(areas)
        area = areas{iarea};
        for iclust = 1:num_clusters
            nexttile;
            hold on;
            title(sprintf('Area %s - Cluster %d', area, iclust));

            xline(reward_zone_start_bins, 'r');
            % Adjust xline as needed for your data

            % Find patterns in the current cluster and area
            cluster_indices = clusters{iclust};
            area_indices = find(strcmp({pattern_labels(cluster_indices).area}, area));
            indices = cluster_indices(area_indices);

            if ~isempty(indices)
                plot(normalized_spatial_patterns(:, indices), 'Color', [0.5 0.5 0.5]);
                plot(mean(normalized_spatial_patterns(:, indices), 2), 'Color', cmap(iclust, :), 'LineWidth', 2);
            end

            hold off;
            axis tight;
        end
    end

    xlabel(t, 'Spatial Bin');
    ylabel(t, 'Factor Loading');

    % Visualize clusters - Trial Patterns
    figure;
    t = tiledlayout(numel(areas), num_clusters, 'TileSpacing', 'compact', 'Padding', 'compact');

    for iarea = 1:numel(areas)
        area = areas{iarea};
        for iclust = 1:num_clusters
            nexttile;
            hold on;
            title(sprintf('Area %s - Cluster %d', area, iclust));

            % Find patterns in the current cluster and area
            cluster_indices = clusters{iclust};
            area_indices = find(strcmp({pattern_labels(cluster_indices).area}, area));
            indices = cluster_indices(area_indices);

            if ~isempty(indices)
                % Get the trial patterns for the selected indices
                trial_patterns = all_trial_patterns(indices); % This is a cell array
                num_patterns = length(trial_patterns);
                % Find the maximum trial length among the selected trial patterns
                max_length = max(cellfun(@length, trial_patterns));
                % Initialize a matrix to hold padded trial patterns
                trial_pattern_matrix = NaN(max_length, num_patterns);
                for i = 1:num_patterns
                    len = length(trial_patterns{i});
                    trial_pattern_matrix(1:len, i) = trial_patterns{i};
                end
                % Now plot mean and sem, ignoring NaNs
                mean_pattern = nanmean(trial_pattern_matrix, 2);
                sem_pattern = nanstd(trial_pattern_matrix, 0, 2) / sqrt(num_patterns);
                shadedErrorBar(1:max_length, mean_pattern, sem_pattern);
            end

            hold off;
            axis tight;
        end
    end

    xlabel(t, 'Trial');
    ylabel(t, 'Loading');
end