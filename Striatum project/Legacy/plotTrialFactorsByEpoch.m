% File: plotTrialFactorsByEpoch.m
function plotTrialFactorsByEpoch(model, cfg, nFactors)
% Analyzes and plots trial factor loadings grouped by learning epoch.

    fprintf('Analyzing and plotting trial factors by epoch...\n');

    if isempty(model) || length(model.U) < 3 || isempty(model.U{3})
        warning('plotTrialFactorsByEpoch: Invalid model or trial factors missing.');
        return;
    end

    trial_loadings = model.U{3};
    n_aligned_trials = size(trial_loadings, 1);

    % Define Epochs based on config (assuming 30 trials)
    if n_aligned_trials == 30
        n_early = numel(cfg.control_epoch_windows{1});
        n_pre = abs(diff(cfg.control_epoch_windows{2})) + 1;
        n_post = abs(diff(cfg.control_epoch_windows{3}));

        epoch_indices = {1:n_early, (n_early+1):(n_early+n_pre), (n_early+n_pre+1):(n_early+n_pre+n_post)};
        epoch_names = cfg.plot.epoch_names; % {'Naive', 'Intermediate', 'Expert'}
        epoch_colors = [cfg.plot.colors.epoch_early; cfg.plot.colors.epoch_middle; cfg.plot.colors.epoch_expert];
        nEpochs = numel(epoch_names);
    else
        warning('plotTrialFactorsByEpoch: Expected 30 aligned trials, found %d. Cannot define epochs reliably.', n_aligned_trials);
        return;
    end

    % Check for required stats functions
    has_stats_toolbox = ~isempty(which('anova1')) && ~isempty(which('multcompare'));
    has_sigstar = ~isempty(which('sigstar'));
    has_errorbar_plot = ~isempty(which('my_simple_errorbar_plot')); % Your custom function

    if ~has_errorbar_plot
        warning('Custom function my_simple_errorbar_plot not found. Using boxplot instead.');
    end

    % --- Setup Figure ---
    figure;
    t_epoch = tiledlayout('flow', 'TileSpacing', 'compact');
    title(t_epoch, 'Trial Factor Loadings by Learning Epoch');
    xlabel(t_epoch, 'Learning Epoch');
    ylabel(t_epoch, 'Loading');

    % --- Loop Through Factors ---
    for r = 1:nFactors
        nexttile(t_epoch);
        factor_data = trial_loadings(:, r); % Loadings for factor r

        % --- Extract data for each epoch ---
        data_for_plot = cell(1, nEpochs);
        data_vec = []; % Vector for all data points for ANOVA
        group_vec = []; % Grouping variable for ANOVA

        for e = 1:nEpochs
            epoch_data = factor_data(epoch_indices{e});
            data_for_plot{e} = epoch_data(~isnan(epoch_data)); % Remove NaNs per epoch for plotting/stats
            data_vec = [data_vec; data_for_plot{e}(:)]; %#ok<AGROW>
            group_vec = [group_vec; repmat(epoch_names(e), numel(data_for_plot{e}), 1)]; %#ok<AGROW>
        end

        % --- Plotting ---
        hold on;
        if has_errorbar_plot
            try
                my_simple_errorbar_plot(data_for_plot, epoch_colors);
            catch ME_plot
                warning('Error during my_simple_errorbar_plot for factor %d: %s\nUsing boxplot.', r, ME_plot.message);
                boxplot(data_vec, group_vec, 'Colors', epoch_colors, 'Symbol', 'o', 'Widths', 0.6);
            end
        else % Fallback boxplot
            boxplot(data_vec, group_vec, 'Colors', epoch_colors, 'Symbol', 'o', 'Widths', 0.6);
        end
        set(gca, 'XTick', 1:nEpochs, 'XTickLabel', epoch_names);
        xlim([0.5 nEpochs + 0.5]);
        title(sprintf('Factor %d', r));
        box off;

        % --- Statistics (ANOVA and Multiple Comparisons) ---
        if has_stats_toolbox
            unique_groups = unique(group_vec);
            if numel(unique_groups) > 1 && numel(data_vec) > numel(unique_groups) % Need >1 group and enough data
                try
                    group_cat = categorical(group_vec, epoch_names, 'Ordinal', false);
                    [p_anova, ~, stats] = anova1(data_vec, group_cat, 'off');

                    if p_anova < 0.05
                        mc = multcompare(stats, 'Display', 'off', 'Alpha', 0.05, 'CType', 'tukey-kramer');
                        sigPairs = {}; pvals = [];
                        for i = 1:size(mc, 1)
                            if mc(i, 6) < 0.05 % Check p-value (column 6)
                                pair = sort([mc(i, 1), mc(i, 2)]); % Indices match epoch_names order
                                sigPairs{end+1} = pair; %#ok<AGROW>
                                pvals(end+1) = mc(i, 6); %#ok<AGROW>
                            end
                        end

                        if ~isempty(sigPairs)
                            [~, unique_idx] = unique(cellfun(@mat2str, sigPairs, 'UniformOutput', false));
                            unique_pairs = sigPairs(unique_idx);
                            unique_pvals = pvals(unique_idx);
                            if has_sigstar
                                sigstar(unique_pairs, unique_pvals);
                            else
                                fprintf('sigstar not found. Significance for Factor %d: %s (p<0.05)\n', r, mat2str(cell2mat(unique_pairs(:))));
                                % Could add simple text markers as fallback
                            end
                        end % if ~isempty(sigPairs)
                    end % if p_anova < 0.05
                catch ME_stats
                     warning('Error during ANOVA/multcompare for factor %d: %s', r, ME_stats.message);
                end % try-catch stats
            else
                 fprintf('Skipping stats for factor %d: Not enough distinct groups or data points.\n', r);
            end % if enough groups/data
        else
             fprintf('Statistics Toolbox not found. Skipping ANOVA/multcompare for factor %d.\n', r);
        end % if has_stats_toolbox
        hold off;
    end % End factor loop
end