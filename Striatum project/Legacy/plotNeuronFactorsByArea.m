function plotNeuronFactorsByArea(model, area_labels, cfg, nFactors, do_downsample)
% Plots neuron factors (Mode 1) grouped by area, with ANOVA and optional downsampling.

    if isempty(model) || ~isempty(area_labels) && size(model.U{1},1) ~= numel(area_labels)
        disp('Skipping neuron factor by area plot: Invalid model or label mismatch.');
        return;
    end
    if isempty(area_labels)
         disp('Skipping neuron factor by area plot: No area labels provided.');
        return;
    end

    neuron_loadings = model.U{1};
    unique_areas = unique(area_labels);
    % Filter out 'Unknown' if present
    unique_areas = unique_areas(~strcmp(unique_areas, 'Unknown'));
    nAreas = numel(unique_areas);

    if nAreas < 2
        fprintf('Skipping ANOVA for neuron factors by area: Only %d area found.\n', nAreas);
        % Consider making just a simple plot if nAreas == 1
        return;
    end

    % Get colors for the areas present
    area_colors_present = cell2mat(values(cfg.plot.colors.area_map, unique_areas));

    % --- Plot Raw Data ---
    figure;
    t_raw = tiledlayout('flow', 'TileSpacing', 'compact');
    title(t_raw, 'Neuron Factors by Area (All Data)');
    ylabel(t_raw, 'Neuron Factor Loading');
    xlabel(t_raw, 'Brain Area');

    stats_results_raw = cell(1, nFactors); % Store stats for raw data

    for r = 1:nFactors
        ax_raw = nexttile(t_raw);
        data_by_area = cell(1, nAreas);
        data_vec = [];
        group_vec = [];

        for a = 1:nAreas
            idx = strcmp(area_labels, unique_areas{a});
            current_data = neuron_loadings(idx, r);
            data_by_area{a} = current_data;
            data_vec = [data_vec; current_data]; %#ok<AGROW>
            group_vec = [group_vec; repmat(unique_areas(a), numel(current_data), 1)]; %#ok<AGROW>
        end

        % Plot using error bar plot or boxplot
        try
            my_simple_errorbar_plot(data_by_area, area_colors_present); % Assumes this function exists
        catch
            warning('my_simple_errorbar_plot not found or failed. Using boxplot.');
            boxplot(data_vec, group_vec, 'Colors', area_colors_present, 'Symbol', 'o', 'Widths', 0.6);
        end
        set(gca, 'XTick', 1:nAreas, 'XTickLabel', unique_areas);
        title(sprintf('Factor %d', r));

        % Statistics (ANOVA and Multiple Comparisons)
        if nAreas > 1 && numel(unique(group_vec)) > 1 && numel(data_vec) > nAreas
            try
                group_cat = categorical(group_vec, unique_areas, 'Ordinal', false);
                [p_anova, tbl, stats] = anova1(data_vec, group_cat, 'off');
                stats_results_raw{r}.p_anova = p_anova;
                stats_results_raw{r}.table = tbl;
                stats_results_raw{r}.stats = stats;

                if p_anova < 0.05
                    mc = multcompare(stats, 'Display', 'off', 'Alpha', 0.05, 'CType', 'tukey-kramer');
                    stats_results_raw{r}.multcompare = mc;
                    sigPairs = {};
                    pvals = [];
                    for i = 1:size(mc, 1)
                        if mc(i, 6) < 0.05
                            pair = sort([mc(i, 1), mc(i, 2)]); % Indices match unique_areas order
                            sigPairs{end+1} = pair; %#ok<AGROW>
                            pvals(end+1) = mc(i, 6); %#ok<AGROW>
                        end
                    end
                    if ~isempty(sigPairs) && ~isempty(which('sigstar'))
                        [~, unique_idx] = unique(cellfun(@mat2str, sigPairs, 'UniformOutput', false));
                        sigstar(sigPairs(unique_idx), pvals(unique_idx));
                    elseif ~isempty(sigPairs)
                         fprintf('sigstar function not found. Skipping significance stars for Factor %d (Raw).\n', r);
                    end
                end
            catch ME_stats
                warning('Error during ANOVA/multcompare for Factor %d (Raw): %s', r, ME_stats.message);
            end
        end
    end % End factor loop raw

    % --- Optional Downsampled Analysis ---
    if do_downsample
        fprintf('Performing downsampled analysis for neuron factors by area...\n');

        % Get indices for each area
        indices_by_area = cell(1, nAreas);
        n_units_per_area = zeros(1, nAreas);
        for a = 1:nAreas
            indices_by_area{a} = find(strcmp(area_labels, unique_areas{a}));
            n_units_per_area(a) = numel(indices_by_area{a});
        end

        % Determine target number of units
        target_units = min(n_units_per_area);

        if target_units <= 1 % Cannot downsample reasonably
             fprintf('Skipping downsampled analysis: Minimum units per area (%d) is too low.\n', target_units);
             return;
        end
        fprintf('Downsampling to %d units per area.\n', target_units);

        % Perform downsampling (consider running multiple iterations later for robustness)
        downsampled_indices = [];
        for a = 1:nAreas
            ds_idx_rel = randsample(indices_by_area{a}, target_units);
            downsampled_indices = [downsampled_indices; ds_idx_rel]; %#ok<AGROW>
        end

        % Get labels for the downsampled set
        area_labels_ds = area_labels(downsampled_indices);
        neuron_loadings_ds = neuron_loadings(downsampled_indices, :);

        % Plot Downsampled Data
        figure;
        t_ds = tiledlayout('flow', 'TileSpacing', 'compact');
        title(t_ds, sprintf('Neuron Factors by Area (Downsampled to %d units/area)', target_units));
        ylabel(t_ds, 'Neuron Factor Loading');
        xlabel(t_ds, 'Brain Area');

        stats_results_ds = cell(1, nFactors); % Store stats for downsampled data

        for r = 1:nFactors
            ax_ds = nexttile(t_ds);
            data_by_area_ds = cell(1, nAreas);
            data_vec_ds = [];
            group_vec_ds = [];

            for a = 1:nAreas
                idx_ds = strcmp(area_labels_ds, unique_areas{a});
                current_data_ds = neuron_loadings_ds(idx_ds, r);
                data_by_area_ds{a} = current_data_ds;
                data_vec_ds = [data_vec_ds; current_data_ds]; %#ok<AGROW>
                group_vec_ds = [group_vec_ds; repmat(unique_areas(a), numel(current_data_ds), 1)]; %#ok<AGROW>
            end

            % Plot
            try
                my_simple_errorbar_plot(data_by_area_ds, area_colors_present);
            catch
                boxplot(data_vec_ds, group_vec_ds, 'Colors', area_colors_present, 'Symbol', 'o', 'Widths', 0.6);
            end
            set(gca, 'XTick', 1:nAreas, 'XTickLabel', unique_areas);
            title(sprintf('Factor %d', r));

            % Statistics
            if nAreas > 1 && numel(unique(group_vec_ds)) > 1 && numel(data_vec_ds) > nAreas
                 try
                    group_cat_ds = categorical(group_vec_ds, unique_areas, 'Ordinal', false);
                    [p_anova_ds, tbl_ds, stats_ds] = anova1(data_vec_ds, group_cat_ds, 'off');
                    stats_results_ds{r}.p_anova = p_anova_ds;
                    stats_results_ds{r}.table = tbl_ds;
                    stats_results_ds{r}.stats = stats_ds;

                    if p_anova_ds < 0.05
                        mc_ds = multcompare(stats_ds, 'Display', 'off', 'Alpha', 0.05, 'CType', 'tukey-kramer');
                         stats_results_ds{r}.multcompare = mc_ds;
                         sigPairs_ds = {};
                         pvals_ds = [];
                        for i = 1:size(mc_ds, 1)
                            if mc_ds(i, 6) < 0.05
                                pair_ds = sort([mc_ds(i, 1), mc_ds(i, 2)]);
                                sigPairs_ds{end+1} = pair_ds; %#ok<AGROW>
                                pvals_ds(end+1) = mc_ds(i, 6); %#ok<AGROW>
                            end
                        end
                        if ~isempty(sigPairs_ds) && ~isempty(which('sigstar'))
                            [~, unique_idx_ds] = unique(cellfun(@mat2str, sigPairs_ds, 'UniformOutput', false));
                            sigstar(sigPairs_ds(unique_idx_ds), pvals_ds(unique_idx_ds));
                         elseif ~isempty(sigPairs_ds)
                             fprintf('sigstar function not found. Skipping significance stars for Factor %d (Downsampled).\n', r);
                        end
                    end
                 catch ME_stats_ds
                    warning('Error during ANOVA/multcompare for Factor %d (Downsampled): %s', r, ME_stats_ds.message);
                end
            end % End stats check ds
        end % End factor loop ds
    end % End if do_downsample
end % End function