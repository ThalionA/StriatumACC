%% Corridor vs dark-period firing rate: metric x aggregation x cell type
%
% The 5 s dark inter-trial period is cached alongside the corridor traversal
% (`temp_binned_dark_fr`, 50 x 100 ms bins) but had never been analysed. This
% script compares mean firing rate in the two states and tracks both across the
% epoch windows, in the same variant family as the other project figures:
%
%   metric      : Raw FR (Hz)  |  Z-Scored
%   aggregation : Hierarchical (animal is the unit of analysis, N = animals)
%                 Pooled       (unit is the unit of analysis, n = units)
%   cell type   : All, plus MSN/FS/TAN in striatum and FS/RS elsewhere
%
% ⚠ TWO THINGS THAT SHAPE INTERPRETATION
%
% 1. The two rates use DIFFERENT ESTIMATORS. Corridor rate is spikes divided by
%    *occupancy* (spatial_binning.m), which carries the known (k-1)*dt
%    denominator defect and is therefore speed-dependent; dark rate is spikes
%    divided by a fixed 0.1 s bin (ProcessStriatumTask.m:185) and is unbiased.
%    Running speed itself rises with learning, so the corridor arm's bias is
%    not constant across epochs. Read the DARK evolution as clean, the CORRIDOR
%    evolution as speed-confounded, and their difference as carrying both.
%
% 2. Z-SCORING IS COMMON ACROSS THE TWO STATES, deliberately. Each unit is
%    centred and scaled by its mean/SD pooled over corridor AND dark samples.
%    Z-scoring the states separately would force both means to zero and make
%    the corridor-vs-dark comparison vacuous by construction. The convention
%    otherwise matches SpatioTemporalActivityEvolution (per unit, over all
%    bins x trials).
%
% Pooled panels use an across-UNIT SEM: units within an animal share a session,
% a learning point and an occupancy artefact, so that band understates the true
% uncertainty. Treat Hierarchical as primary and Pooled as a sensitivity check.
%
% Control 2 is excluded: it is the dark-only cohort and has no corridor.
%
% Outputs: figures/CorridorVsDark_*.{svg,png}
%          figures/corridor_vs_dark_by_animal.csv (animal x area x type x epoch)
%
% Created 2026-08-27.

clearvars -except all_data; clc; close all;
cfg = project_cfg();
rng(cfg.seed, 'twister');

% Build figures off-screen so a long run doesn't throw windows in front of
% whatever you're doing; they still save normally.
fig_guard = figures_offscreen(); %#ok<NASGU>

trials_per_epoch = cfg.trials_per_epoch;
naive_split      = 3;      % matches SpatioTemporalActivityEvolution's {1:3, 4:10}
epoch_names      = {sprintf('Trials 1-%d', naive_split), ...
                    sprintf('Trials %d-%d', naive_split + 1, trials_per_epoch), ...
                    'Intermediate', 'Expert'};
n_epochs = numel(epoch_names);

areas       = {'DMS', 'DLS', 'ACC', 'V1', 'CA1'};
striatal    = {'DMS', 'DLS'};
n_areas     = numel(areas);
type_codes  = [NaN 1 2 3 5];                       % NaN = "All types"
type_names  = {'All', 'MSN', 'FS', 'TAN', 'RS'};
n_types     = numel(type_names);
striatal_types    = [NaN 1 2 3];                   % All, MSN, FS, TAN
nonstriatal_types = [NaN 2 5];                     % All, FS, RS
states      = {'corridor', 'dark'};
metrics     = {'Raw FR', 'Z-Scored'};
aggregations = {'Hierarchical', 'Pooled'};

group_files = {cfg.task_data_file, cfg.control_data_file};
group_names = {'Task', 'Control 1'};

lp_cfg = struct('lp_z_threshold', cfg.lp_z_threshold, ...
                'lp_window', cfg.lp_window, ...
                'lp_min_consecutive', cfg.lp_min_consecutive);

fprintf('--- Corridor vs dark firing rate ---\n');

% Per-unit records: one row per (unit, epoch), carrying both states and both
% metrics. Aggregation to animal or unit level happens at plot time.
U = struct('group', {}, 'animal', {}, 'area', {}, 'type', {}, ...
           'epoch', {}, 'raw_c', {}, 'raw_d', {}, 'z_c', {}, 'z_d', {});

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
        dark_fr = data(i).temp_binned_dark_fr;
        n_tr = min(size(corr_fr, 3), size(dark_fr, 3));
        corr_fr = corr_fr(:, :, 1:n_tr);
        dark_fr = dark_fr(:, :, 1:n_tr);

        if g == 1, lp = task_lps(i); else, lp = avg_task_lp; end
        idx = epoch_indices(lp, n_tr, struct('trials_per_epoch', trials_per_epoch, ...
                                             'naive_split', naive_split));

        % Per-unit mean over bins -> units x trials, per state.
        pt_c = squeeze(mean(corr_fr, 2, 'omitnan'));
        pt_d = squeeze(mean(dark_fr, 2, 'omitnan'));

        % COMMON z-score per unit over both states (see header note 2).
        both = [pt_c, pt_d];
        mu   = mean(both, 2, 'omitnan');
        sd   = std(both, 0, 2, 'omitnan');
        sd(~isfinite(sd) | sd == 0) = NaN;         % silent units -> NaN, not Inf
        z_c = (pt_c - mu) ./ sd;
        z_d = (pt_d - mu) ./ sd;

        % Cell-type code per unit (column 5 of final_neurontypes).
        if isfield(data(i), 'final_neurontypes') && ~isempty(data(i).final_neurontypes) ...
                && size(data(i).final_neurontypes, 2) >= 5
            tcode = data(i).final_neurontypes(:, 5);
        else
            tcode = nan(size(pt_c, 1), 1);
        end

        for a = 1:n_areas
            mask = is_area_safe(data(i), areas{a});
            if ~any(mask), continue; end
            uidx = find(mask);
            for e = 1:n_epochs
                tr = idx{e};
                if isempty(tr), continue; end
                for u = uidx(:)'
                    U(end+1) = struct( ...
                        'group', group_names{g}, 'animal', i, 'area', areas{a}, ...
                        'type', tcode(u), 'epoch', e, ...
                        'raw_c', mean(pt_c(u, tr), 'omitnan'), ...
                        'raw_d', mean(pt_d(u, tr), 'omitnan'), ...
                        'z_c',   mean(z_c(u, tr),  'omitnan'), ...
                        'z_d',   mean(z_d(u, tr),  'omitnan')); %#ok<SAGROW>
                end
            end
        end
        n_units_animal = numel(unique([U(strcmp({U.group}, group_names{g}) & ...
                                        [U.animal] == i).area]));
        fprintf('  %s animal %d: lp=%s, %d trials, %d areas with units\n', ...
                group_names{g}, i, num2str(lp), n_tr, n_units_animal);
    end
end
fprintf('  %d unit x epoch records\n', numel(U));

u_group = string({U.group})';  u_animal = [U.animal]';  u_area = string({U.area})';
u_type  = [U.type]';           u_epoch  = [U.epoch]';
% vals{metric} = [corridor, dark] columns, aligned with the U records.
vals = {[[U.raw_c]', [U.raw_d]'], [[U.z_c]', [U.z_d]']};

%% --- Export: animal-level table (hierarchical unit of analysis) -----------
rows = {};
for g = 1:2
    for a = 1:n_areas
        for t = 1:n_types
            for e = 1:n_epochs
                sel = u_group == group_names{g} & u_area == areas{a} & u_epoch == e & ...
                      type_sel(u_type, type_codes(t));
                if ~any(sel), continue; end
                for an = unique(u_animal(sel))'
                    s2 = sel & u_animal == an;
                    rows(end+1, :) = {group_names{g}, an, areas{a}, type_names{t}, ...
                        epoch_names{e}, sum(s2), ...
                        mean(vals{1}(s2, 1), 'omitnan'), ...
                        mean(vals{1}(s2, 2), 'omitnan'), ...
                        mean(vals{2}(s2, 1), 'omitnan'), ...
                        mean(vals{2}(s2, 2), 'omitnan')}; %#ok<SAGROW>
                end
            end
        end
    end
end
T = cell2table(rows, 'VariableNames', {'group', 'animal', 'area', 'cell_type', ...
    'epoch', 'n_units', 'corridor_fr', 'dark_fr', 'corridor_z', 'dark_z'});
writetable(T, fullfile('figures', 'corridor_vs_dark_by_animal.csv'));
fprintf('Wrote figures/corridor_vs_dark_by_animal.csv (%d rows).\n', height(T));

%% --- Figures: metric x aggregation, area-level ----------------------------
for m = 1:numel(metrics)
    V = vals{m};
    for ag = 1:numel(aggregations)
        figure('Name', sprintf('CorridorDark %s %s by area', metrics{m}, aggregations{ag}), ...
               'Position', [80 80 1500 660], 'Color', 'w');
        for g = 1:2
            for a = 1:n_areas
                subplot(2, n_areas, (g - 1) * n_areas + a); hold on;
                base = u_group == group_names{g} & u_area == areas{a};
                for st = 1:2
                    [mu, se, nn] = agg_epochs(V(:, st), base, u_epoch, u_animal, ...
                                              n_epochs, aggregations{ag});
                    if all(isnan(mu)), continue; end
                    col = cfg.area_colors(a, :);
                    if st == 2, col = [0.45 0.45 0.45]; end
                    errorbar(1:n_epochs, mu, se, '-o', 'LineWidth', 1.8, 'CapSize', 4, ...
                             'Color', col, 'MarkerFaceColor', 'w', 'DisplayName', states{st});
                end
                style_epoch_axis(n_epochs, epoch_names);
                title(sprintf('%s — %s (%s=%d)', group_names{g}, areas{a}, ...
                      unit_label(aggregations{ag}), nn), 'FontSize', 9);
                if a == 1, ylabel(y_label(metrics{m})); end
                if a == 1 && g == 1, legend('Location', 'best', 'FontSize', 7); end
            end
        end
        sgtitle(sprintf('Corridor vs dark — %s, %s', metrics{m}, aggregations{ag}));
        save_to_svg(sprintf('CorridorVsDark_%s_%s_byArea', ...
                    clean_tag(metrics{m}), aggregations{ag}));
    end
end

%% --- Figures: metric x aggregation, area x cell type ----------------------
for m = 1:numel(metrics)
    V = vals{m};
    for ag = 1:numel(aggregations)
        for g = 1:2
            figure('Name', sprintf('CorridorDark %s %s %s by type', ...
                   metrics{m}, aggregations{ag}, group_names{g}), ...
                   'Position', [60 60 1600 760], 'Color', 'w');
            for a = 1:n_areas
                allowed = ternary(ismember(areas{a}, striatal), striatal_types, nonstriatal_types);
                % NaN codes "All types"; ismember(NaN,...) is false, so test it
                % explicitly or the pooled column is gated out everywhere.
                keep = arrayfun(@(c) isnan(c) || ismember(c, allowed), type_codes);
                for t = 1:n_types
                    subplot(n_areas, n_types, (a - 1) * n_types + t); hold on;
                    if ~keep(t)
                        axis off;
                        text(0.5, 0.5, 'n/a', 'HorizontalAlignment', 'center', ...
                             'Color', [0.7 0.7 0.7], 'FontSize', 8);
                        continue
                    end
                    base = u_group == group_names{g} & u_area == areas{a} & ...
                           type_sel(u_type, type_codes(t));
                    nn = 0;
                    for st = 1:2
                        [mu, se, nn] = agg_epochs(V(:, st), base, u_epoch, u_animal, ...
                                                  n_epochs, aggregations{ag});
                        if all(isnan(mu)), continue; end
                        col = cfg.area_colors(a, :);
                        if st == 2, col = [0.45 0.45 0.45]; end
                        errorbar(1:n_epochs, mu, se, '-o', 'LineWidth', 1.5, ...
                                 'CapSize', 3, 'Color', col, 'MarkerFaceColor', 'w', ...
                                 'DisplayName', states{st});
                    end
                    style_epoch_axis(n_epochs, epoch_names);
                    title(sprintf('%s — %s (%s=%d)', areas{a}, type_names{t}, ...
                          unit_label(aggregations{ag}), nn), 'FontSize', 8);
                    if t == 1, ylabel(y_label(metrics{m}), 'FontSize', 8); end
                    if a == 1 && t == 1, legend('Location', 'best', 'FontSize', 6); end
                end
            end
            sgtitle(sprintf('%s: corridor vs dark by area x cell type — %s, %s', ...
                    group_names{g}, metrics{m}, aggregations{ag}));
            save_to_svg(sprintf('CorridorVsDark_%s_%s_%s_byType', ...
                        clean_tag(metrics{m}), aggregations{ag}, clean_tag(group_names{g})));
        end
    end
end

save_all_open_figures('corridordark');

% Restore figure visibility for interactive work.
clear fig_guard
fprintf('--- Done ---\n');

% ---------------------------------------------------------------------------
function sel = type_sel(u_type, code)
% NaN code means "all types"; otherwise match the cell-type code exactly.
    if isnan(code), sel = true(size(u_type)); else, sel = u_type == code; end
end

function [mu, se, n_unit] = agg_epochs(v, base, u_epoch, u_animal, n_epochs, how)
% Mean +- SEM per epoch, aggregating either across animals (Hierarchical: each
% animal contributes one value) or across units (Pooled).
    mu = nan(1, n_epochs); se = nan(1, n_epochs); n_unit = 0;
    for e = 1:n_epochs
        s = base & u_epoch == e & isfinite(v);
        if ~any(s), continue; end
        if strcmp(how, 'Hierarchical')
            ids = unique(u_animal(s));
            x = arrayfun(@(a) mean(v(s & u_animal == a), 'omitnan'), ids);
        else
            x = v(s);
        end
        x = x(isfinite(x));
        if isempty(x), continue; end
        mu(e) = mean(x); se(e) = std(x) / sqrt(numel(x));
        n_unit = max(n_unit, numel(x));
    end
end

function style_epoch_axis(n_epochs, epoch_names)
    xlim([0.5 n_epochs + 0.5]); xticks(1:n_epochs); xticklabels(epoch_names);
    xtickangle(30); box on;
end

function s = y_label(metric)
    if strcmp(metric, 'Raw FR'), s = 'Mean FR (Hz)'; else, s = 'Z-scored FR'; end
end

function s = unit_label(how)
    if strcmp(how, 'Hierarchical'), s = 'N'; else, s = 'n'; end
end

function s = clean_tag(x)
    s = regexprep(x, '[^A-Za-z0-9]', '');
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
