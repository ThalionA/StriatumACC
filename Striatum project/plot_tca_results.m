function plot_tca_results(clusters, cluster_assignments, normalized_spatial_patterns, pattern_labels, all_trial_patterns, reward_zone_start_bins, areas)
    % Plot the dendrogram and cluster patterns
    % Now accounts for multiple areas per animal.

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
            % Adjust xline(20, 'r'); as needed for your data

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
                % plot(all_trial_patterns(:, indices));
                shadedErrorBar(1:size(all_trial_patterns, 1), mean(all_trial_patterns(:, indices), 2), sem(all_trial_patterns(:, indices), 2))
            end

            hold off;
        end
    end

    xlabel(t, 'Trial');
    ylabel(t, 'Loading');
end