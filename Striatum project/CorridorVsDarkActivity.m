%% Corridor vs dark-period firing rate, and its evolution across learning
%
% The 5 s dark inter-trial period is cached alongside the corridor traversal
% (`temp_binned_dark_fr`, 50 x 100 ms bins) but has never been analysed. This
% script compares mean firing rate in the two states per area, and tracks both
% across the epoch windows used everywhere else in the project.
%
% ⚠ THE TWO RATES USE DIFFERENT ESTIMATORS. Corridor rate is spikes divided by
% *occupancy* (spatial_binning.m), which carries the known (k-1)*dt denominator
% defect and is therefore speed-dependent; dark rate is spikes divided by a
% fixed 0.1 s bin (ProcessStriatumTask.m:185) and is unbiased. Because running
% speed itself rises with learning, the corridor arm's bias is not even
% constant across epochs. Read the DARK evolution as clean, the CORRIDOR
% evolution as speed-confounded, and the corridor/dark ratio as carrying both.
%
% Control 2 is excluded: it is the dark-only cohort and has no corridor.
%
% Outputs: figures/CorridorVsDark_*.{svg,png} and
%          figures/corridor_vs_dark_by_animal.csv (per animal x area x epoch).
%
% Created 2026-08-27.

clearvars -except all_data; clc; close all;
% Build figures off-screen so a long run doesn't throw windows in front of
% whatever you're doing; they still save normally. Released at the end of
% the script (setenv('MATLAB_SHOW_FIGURES','1') to see them live).
fig_guard = figures_offscreen(); %#ok<NASGU>

cfg = project_cfg();
rng(cfg.seed, 'twister');

trials_per_epoch = cfg.trials_per_epoch;
naive_split      = 3;      % matches SpatioTemporalActivityEvolution's {1:3, 4:10}
epoch_names      = {sprintf('Trials 1-%d', naive_split), ...
                    sprintf('Trials %d-%d', naive_split + 1, trials_per_epoch), ...
                    'Intermediate', 'Expert'};
n_epochs   = numel(epoch_names);
areas      = {'DMS', 'DLS', 'ACC', 'V1', 'CA1'};
n_areas    = numel(areas);
group_files = {cfg.task_data_file, cfg.control_data_file};
group_names = {'Task', 'Control 1'};

% Learning points are not stored in the cache; they are recomputed from the
% z-scored lick errors by the shared helper, exactly as IntegratedAll_v1 does
% (find_learning_points, project rule: z <= -2, window 10, >= 7 sub-threshold).
lp_cfg = struct('lp_z_threshold', cfg.lp_z_threshold, ...
                'lp_window', cfg.lp_window, ...
                'lp_min_consecutive', cfg.lp_min_consecutive);

fprintf('--- Corridor vs dark firing rate ---\n');

rows = {};                                   % long-format export
mean_fr = nan(20, n_areas, n_epochs, 2, 2);  % animal x area x epoch x group x state

task_lps = [];
for g = 1:numel(group_files)
    S = load(group_files{g}, 'preprocessed_data');
    data = S.preprocessed_data;
    if g == 1
        [task_lps, avg_task_lp] = find_learning_points(data, lp_cfg);
        fprintf('  %d/%d task learners, mean LP %.1f\n', ...
                sum(isfinite(task_lps)), numel(task_lps), avg_task_lp);
    end
    for i = 1:numel(data)
        if ~isfield(data(i), 'temp_binned_dark_fr') || isempty(data(i).temp_binned_dark_fr)
            continue
        end
        corr_fr = data(i).spatial_binned_fr_all;   % units x bins x trials
        dark_fr = data(i).temp_binned_dark_fr;     % units x bins x trials
        n_tr = min(size(corr_fr, 3), size(dark_fr, 3));
        corr_fr = corr_fr(:, :, 1:n_tr);
        dark_fr = dark_fr(:, :, 1:n_tr);

        if g == 1
            lp = task_lps(i);
        else
            lp = avg_task_lp;                  % controls have no LP; yoked to task mean
        end

        idx = epoch_indices(lp, n_tr, struct('trials_per_epoch', trials_per_epoch, ...
                                             'naive_split', naive_split));

        % Per-unit mean over bins -> per-trial scalar per unit, per state.
        per_trial = {squeeze(mean(corr_fr, 2, 'omitnan')), ...   % units x trials
                     squeeze(mean(dark_fr, 2, 'omitnan'))};

        for a = 1:n_areas
            mask = is_area_safe(data(i), areas{a});
            if ~any(mask), continue; end
            for e = 1:n_epochs
                tr = idx{e};
                if isempty(tr), continue; end
                for st = 1:2
                    v = mean(per_trial{st}(mask, tr), 2, 'omitnan');  % per unit
                    mean_fr(i, a, e, g, st) = mean(v, 'omitnan');     % area mean
                end
                rows(end+1, :) = {group_names{g}, i, areas{a}, epoch_names{e}, ...
                                  sum(mask), mean_fr(i, a, e, g, 1), ...
                                  mean_fr(i, a, e, g, 2), ...
                                  mean_fr(i, a, e, g, 1) - mean_fr(i, a, e, g, 2)}; %#ok<SAGROW>
            end
        end
        fprintf('  %s animal %d: lp=%s, %d trials\n', group_names{g}, i, ...
                num2str(lp), n_tr);
    end
end

if ~isempty(rows)
    T = cell2table(rows, 'VariableNames', {'group', 'animal', 'area', 'epoch', ...
        'n_units', 'corridor_fr', 'dark_fr', 'corridor_minus_dark'});
    writetable(T, fullfile('figures', 'corridor_vs_dark_by_animal.csv'));
    fprintf('Wrote figures/corridor_vs_dark_by_animal.csv (%d rows).\n', height(T));
end

%% --- Figure 1: corridor vs dark, per area, animal-level -------------------
figure('Name', 'Corridor vs dark firing rate', 'Position', [100 100 1400 560], 'Color', 'w');
for g = 1:2
    subplot(1, 2, g); hold on;
    for a = 1:n_areas
        x = squeeze(mean(mean_fr(:, a, :, g, 2), 3, 'omitnan'));  % dark
        y = squeeze(mean(mean_fr(:, a, :, g, 1), 3, 'omitnan'));  % corridor
        ok = isfinite(x) & isfinite(y);
        scatter(x(ok), y(ok), 46, cfg.area_colors(a, :), 'filled', ...
                'MarkerEdgeColor', 'w', 'DisplayName', areas{a});
    end
    lim = [0 max([1; mean_fr(:)], [], 'omitnan') * 1.05];
    plot(lim, lim, 'k--', 'HandleVisibility', 'off');
    xlim(lim); ylim(lim); axis square; box on;
    xlabel('Dark-period FR (Hz)'); ylabel('Corridor FR (Hz)');
    title(sprintf('%s — one point per animal x area', group_names{g}));
    if g == 1, legend('Location', 'northwest'); end
end
save_to_svg('CorridorVsDark_scatter');

%% --- Figure 2: evolution across epochs, per area ---------------------------
figure('Name', 'Corridor and dark FR across epochs', 'Position', [100 100 1500 680], 'Color', 'w');
for g = 1:2
    for a = 1:n_areas
        subplot(2, n_areas, (g - 1) * n_areas + a); hold on;
        for st = 1:2
            M = squeeze(mean_fr(:, a, :, g, st));        % animals x epochs
            mu = mean(M, 1, 'omitnan');
            se = std(M, 0, 1, 'omitnan') ./ sqrt(sum(isfinite(M), 1));
            if all(isnan(mu)), continue; end
            errorbar(1:n_epochs, mu, se, '-o', 'LineWidth', 1.8, 'CapSize', 4, ...
                     'Color', cfg.area_colors(a, :) * (st == 1) + [0.45 0.45 0.45] * (st == 2), ...
                     'MarkerFaceColor', 'w', ...
                     'DisplayName', ternary(st == 1, 'corridor', 'dark'));
        end
        xlim([0.5 n_epochs + 0.5]); xticks(1:n_epochs); xticklabels(epoch_names);
        xtickangle(30); box on;
        n_an = sum(isfinite(mean_fr(:, a, 1, g, 1)));
        title(sprintf('%s — %s (N=%d)', group_names{g}, areas{a}, n_an), 'FontSize', 9);
        if a == 1, ylabel('Mean FR (Hz)'); end
        if a == 1 && g == 1, legend('Location', 'best', 'FontSize', 7); end
    end
end
save_to_svg('CorridorVsDark_epoch_evolution');

save_all_open_figures('corridordark');

% Restore figure visibility for interactive work.
clear fig_guard
fprintf('--- Done ---\n');

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
