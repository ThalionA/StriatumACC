% File: plotFactorHeatmapsByMouseArea.m
function plotFactorHeatmapsByMouseArea(model, mouse_labels, area_labels, nFactors, nAnimalsTotal)
% Generates heatmaps showing mean neuron loading per factor for each mouse, faceted by area.

    fprintf('Generating heatmaps of mean factor loadings by mouse and area...\n');

    if isempty(model) || isempty(model.U{1})
        warning('plotFactorHeatmapsByMouseArea: Invalid model provided.');
        return;
    end
    if isempty(mouse_labels) || isempty(area_labels)
         warning('plotFactorHeatmapsByMouseArea: Mouse or area labels are missing.');
        return;
    end
    if numel(mouse_labels) ~= size(model.U{1}, 1) || numel(area_labels) ~= size(model.U{1}, 1)
         warning('plotFactorHeatmapsByMouseArea: Label dimension mismatch with neuron factors.');
        return;
    end

    neuron_loadings = model.U{1};
    unique_areas = unique(area_labels);
    unique_areas = unique_areas(~strcmp(unique_areas, 'Unknown')); % Exclude 'Unknown'
    nAreas = numel(unique_areas);
    unique_mice = unique(mouse_labels);
    % Ensure nAnimalsTotal matches the actual mice present in labels if different from tensor_info
    nAnimalsPresent = numel(unique_mice);
    if nAnimalsPresent ~= nAnimalsTotal
        fprintf('  Note: %d animals present in valid labels, tensor_info reported %d total.\n', nAnimalsPresent, nAnimalsTotal);
        % Use nAnimalsPresent for x-axis ticks if they differ significantly
    end


    if nAreas == 0
        warning('plotFactorHeatmapsByMouseArea: No valid areas found in labels.');
        return;
    end

    % Calculate mean loadings: Factors x Mouse x Area
    mean_loadings_faceted = nan(nFactors, nAnimalsTotal, nAreas); % Initialize based on total potential animals

    area_labels_cat = categorical(area_labels); % Use categorical for faster comparison

    for r = 1:nFactors
        for m_idx = 1:nAnimalsPresent % Iterate through mice actually present
            m = unique_mice(m_idx); % Get the actual mouse ID
            for a = 1:nAreas
                % Find indices for neurons from this mouse AND this area
                idx_combined = find(mouse_labels == m & area_labels_cat == unique_areas{a});
                if ~isempty(idx_combined)
                    % Place data in matrix using actual mouse ID (m) as index
                    mean_loadings_faceted(r, m, a) = mean(neuron_loadings(idx_combined, r), 'omitnan');
                % else, leave as NaN
                end
            end
        end
    end

    % Determine common color limits across all plotted data, ignoring NaNs
    clim_min = min(mean_loadings_faceted(:), [], 'omitnan');
    clim_max = max(mean_loadings_faceted(:), [], 'omitnan');
    if isempty(clim_min) || isempty(clim_max) || isnan(clim_min) || isnan(clim_max) || clim_min == clim_max
        warning('Could not determine valid color limits for heatmap. Using [0 1].');
        clim_min = 0; clim_max = 1; % Fallback
    end

    % --- Plotting ---
    figure;
    t_heatmap = tiledlayout(1, nAreas, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_heatmap, 'Mean Neuron Factor Loadings (Factors x Mouse ID) per Area');

    for a = 1:nAreas
        nexttile(t_heatmap);
        % Select data for the current area, potentially only showing mice present
        % area_data = mean_loadings_faceted(:, unique_mice, a); % Show only mice with data
        area_data = mean_loadings_faceted(:, 1:nAnimalsTotal, a); % Show all potential mouse slots

        imagesc(area_data);
        set(gca, 'YDir', 'normal'); % Factor 1 at the bottom
        colormap(gca, "parula"); % Use specific colormap
        clim([clim_min, clim_max]); % Apply consistent color limits

        % Add labels and ticks
        title(unique_areas{a});
        if a == 1 % Add Y label only to the first plot
            ylabel('Factor Index');
            yticks(1:nFactors);
            yticklabels(1:nFactors);
        else
             yticks([]); % No yticks on subsequent plots
        end
        xlabel('Mouse ID');
        xticks(unique_mice); % Tick only the mice present? Or 1:nAnimalsTotal?
        xtickangle(45);
        set(gca,'XTickLabel', arrayfun(@num2str, unique_mice, 'UniformOutput', false)); % Label with actual IDs
        % Consider showing all xticks 1:nAnimalsTotal if gaps are meaningful
        % xticks(1:nAnimalsTotal);
        % xtickangle(45);
    end

    % Add a single colorbar for the whole figure
    try % Catch error if layout has issues with colorbar positioning
        cb = colorbar(nexttile(t_heatmap, nAreas)); % Get handle from last tile
        cb.Layout.Tile = 'east'; % Place colorbar to the east
        ylabel(cb, 'Mean Neuron Loading');
    catch ME_cb
        warning('Could not automatically position colorbar: %s', "%s", ME_cb.message);
        % Add fallback colorbar to the last axis if layout fails
        colorbar;
    end
end