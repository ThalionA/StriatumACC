% compare_bin_sizes.m
% HISTORICAL ONE-OFF (2026-08-10): this comparison decided the pipeline's
% 5 cm standard (see PREDICTIONS.md outcome; figures/compare_bin_sizes.*).
% The 2.5 cm product is no longer emitted, so re-running requires temporarily
% setting bin_size = 2 in ProcessStriatumTask.m to regenerate
% preprocessed_data2p5cm.mat first.
%
% Empirical soundness comparison of the two spatial binnings (2.5 cm vs 5 cm)
% of the task cohort. Answers: does 2.5 cm binning cost reliability (noise),
% and does it carry any REAL structure below the 5 cm scale, or is the extra
% resolution pure noise?
%
% Requires: preprocessed_data2p5cm.mat and preprocessed_data5cm.mat in the
% working directory (both emitted by one run of ProcessStriatumTask.m).
%
% Metrics (per unit, then per-animal medians; animal = inferential unit):
%   r_fine / r_coarse : split-half (odd/even trials) tuning reliability
%   r_cross           : held-out fine curve vs interpolated coarse curve.
%                       r_fine > r_cross  => genuine sub-5cm structure.
%                       r_cross >= r_fine => fine grid adds noise, not signal.
%   zero_frac         : sparsity cost of small bins
%
% Outputs (figures/ dir): compare_bin_sizes.{svg,png}, per-animal summary CSV,
% full per-unit metrics .mat (raw data saved alongside figures per repo rules).
%
% Prediction registered in ../PREDICTIONS.md (2026-08-10) BEFORE first run.

clearvars; clc;

% --- Bin geometry: replicate ProcessStriatumTask exactly (incl. the widened last bin) ---
corridor_end_au = 200;
au2cm = 1.25;
mk_centres = @(bs) centres_of(bs, corridor_end_au) * au2cm;
centres_fine   = mk_centres(2);   % 2.5 cm grid
centres_coarse = mk_centres(4);   % 5 cm grid

% --- Load both products SEQUENTIALLY, keeping only the slim fields ---
% The full structs carry the 1 ms payloads (binned_spikes_trials, darkData,
% corridorData) and expand to tens of GB in RAM EACH; loading both at once
% OOM-killed MATLAB on 2026-08-10. Extract what the comparison needs, drop
% the rest before touching the second file.
slim = @(P) arrayfun(@(p) struct( ...
    'spatial_binned_fr_all', p.spatial_binned_fr_all, 'n_trials', p.n_trials, ...
    'is_dms', p.is_dms, 'is_dls', p.is_dls, 'is_acc', p.is_acc, ...
    'is_v1', p.is_v1, 'is_ca1', p.is_ca1, 'is_dg', p.is_dg), P);
S = load('preprocessed_data2p5cm.mat', 'preprocessed_data');
P_fine = slim(S.preprocessed_data); clear S
S = load('preprocessed_data5cm.mat', 'preprocessed_data');
P_coarse = slim(S.preprocessed_data); clear S
n_animals = numel(P_fine);
assert(numel(P_coarse) == n_animals, 'animal count differs between the two files');

areas = {'DMS', 'DLS', 'ACC', 'V1', 'CA1', 'DG'};
masks = {'is_dms', 'is_dls', 'is_acc', 'is_v1', 'is_ca1', 'is_dg'};
min_units = 5;

rows = [];          % per animal x area summary
all_metrics = {};   % full per-unit metrics for the .mat

for ia = 1:n_animals
    fr_fine_all   = P_fine(ia).spatial_binned_fr_all;
    fr_coarse_all = P_coarse(ia).spatial_binned_fr_all;
    assert(size(fr_fine_all, 1) == size(fr_coarse_all, 1), ...
        'unit count differs between binnings for animal %d', ia);
    % Trial counts MUST match: both files come from one dual-bin run over
    % identical trials. A mismatch means mis-paired files from different
    % runs, which would invalidate every paired metric.
    assert(size(fr_fine_all, 3) == size(fr_coarse_all, 3), ...
        'trial count differs for animal %d - products are from different runs', ia);
    n_tr = size(fr_fine_all, 3);
    for imask = 1:numel(masks)
        if isfield(P_fine, masks{imask}) && isfield(P_coarse, masks{imask})
            assert(isequal(P_fine(ia).(masks{imask}), P_coarse(ia).(masks{imask})), ...
                '%s mask differs between binnings for animal %d', masks{imask}, ia);
        end
    end
    nf = nnz(~isfinite(fr_fine_all)) + nnz(~isfinite(fr_coarse_all)) ...
       - nnz(isnan(fr_fine_all)) - nnz(isnan(fr_coarse_all));
    if nf > 0
        fprintf('animal %d: %d non-finite (Inf) rate entries treated as unvisited\n', ia, nf);
    end

    for iar = 1:numel(areas)
        if ~isfield(P_fine, masks{iar}), continue; end
        m = P_fine(ia).(masks{iar});
        if sum(m) < min_units, continue; end

        M = bin_comparison_metrics(fr_fine_all(m, :, 1:n_tr), ...
                                   fr_coarse_all(m, :, 1:n_tr), ...
                                   centres_fine, centres_coarse);
        M.animal = ia; M.area = areas{iar};
        all_metrics{end+1} = M; %#ok<SAGROW>

        med = @(x) median(x, 'omitnan');
        rows = [rows; {ia, areas{iar}, sum(m), ...
            med(M.r_fine), med(M.r_coarse), med(M.r_cross), ...
            med(M.r_coarse) - med(M.r_fine), ...     % reliability gain of 5 cm
            med(M.r_fine) - med(M.r_cross), ...      % genuine sub-5cm structure
            med(M.zero_frac_fine), med(M.zero_frac_coarse)}]; %#ok<AGROW>
    end
end

T = cell2table(rows, 'VariableNames', {'animal', 'area', 'n_units', ...
    'r_fine', 'r_coarse', 'r_cross', 'd_reliability_5cm_minus_2p5', ...
    'd_structure_fine_minus_cross', 'zero_frac_2p5cm', 'zero_frac_5cm'});

% --- Per-area stats across animals (animal = inferential unit) ---
fprintf('\n=== Bin-size comparison: per-area summary (median across animals) ===\n');
fprintf('%-5s %3s | r(2.5) r(5.0) r(x) | d_rel  p_rel | d_str  p_str | zero%%2.5 zero%%5\n', 'area', 'n');
verdicts = struct();
for iar = 1:numel(areas)
    A = T(strcmp(T.area, areas{iar}), :);
    n = height(A);
    if n < 3, continue; end
    d_rel = A.d_reliability_5cm_minus_2p5;
    d_str = A.d_structure_fine_minus_cross;
    p_rel = signrank(d_rel);   % Wilcoxon signed-rank, H0: median = 0
    p_str = signrank(d_str);
    fprintf('%-5s %3d | %5.2f  %5.2f  %5.2f | %+5.2f  %5.3f | %+5.2f  %5.3f | %6.1f  %6.1f\n', ...
        areas{iar}, n, median(A.r_fine), median(A.r_coarse), median(A.r_cross), ...
        median(d_rel), p_rel, median(d_str), p_str, ...
        100*median(A.zero_frac_2p5cm), 100*median(A.zero_frac_5cm));
    verdicts.(areas{iar}) = struct('n', n, 'd_rel', median(d_rel), 'p_rel', p_rel, ...
                                   'd_str', median(d_str), 'p_str', p_str);
end
fprintf(['\nWilcoxon signed-rank across animals (n as shown); note the p-floor at ', ...
    'small n (n=5 -> min p = 0.0625).\n']);
fprintf(['Reading: d_rel > 0 = 5 cm more reliable (2.5 cm noisier, as expected).\n', ...
    '         d_str > 0 = evidence of REAL sub-5cm structure (favours 2.5 cm);\n', ...
    '         d_str <= 0 = no sub-5cm structure DETECTABLE at this trial count\n', ...
    '                      (asymmetric test: the interpolated coarse predictor is\n', ...
    '                      less noisy, so this direction is expected even if modest\n', ...
    '                      fine structure exists; 5 cm is the safe default).\n']);

% --- Figure: paired reliability + structure test ---
fig = figure('Position', [100, 100, 1400, 550], 'Color', 'w');
present_areas = unique(T.area, 'stable');
n_pa = numel(present_areas);

subplot(1, 2, 1); hold on;
for iar = 1:n_pa
    A = T(strcmp(T.area, present_areas{iar}), :);
    x0 = iar - 0.15; x1 = iar + 0.15;
    plot([x0; x1], [A.r_fine'; A.r_coarse'], '-', 'Color', [0.75 0.75 0.75]);
    plot(x0, A.r_fine, 'o', 'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'none');
    plot(x1, A.r_coarse, 'o', 'MarkerFaceColor', [0.00 0.45 0.74], 'MarkerEdgeColor', 'none');
end
xticks(1:n_pa); xticklabels(present_areas); xlim([0.5, n_pa + 0.5]);
ylabel('Split-half tuning reliability (Pearson r)');
title('Reliability per animal: 2.5 cm (orange) vs 5 cm (blue) bins');
grid on;

subplot(1, 2, 2); hold on;
for iar = 1:n_pa
    A = T(strcmp(T.area, present_areas{iar}), :);
    xj = iar + 0.1 * randn(height(A), 1);
    plot(xj, A.d_structure_fine_minus_cross, 'o', ...
        'MarkerFaceColor', [0.47 0.67 0.19], 'MarkerEdgeColor', 'none');
    plot(iar + [-0.25, 0.25], median(A.d_structure_fine_minus_cross) * [1, 1], 'k-', 'LineWidth', 2);
end
yline(0, 'k--');
xticks(1:n_pa); xticklabels(present_areas); xlim([0.5, n_pa + 0.5]);
ylabel('r_{fine} - r_{cross} (per-animal median)');
title({'Sub-5cm structure test', '> 0 favours 2.5 cm; \leq 0 = none detectable at this trial count'});
grid on;

% --- Save outputs (svg + png pair, raw data alongside) ---
if ~exist('figures', 'dir'), mkdir('figures'); end
set(fig, 'PaperUnits', 'inches', 'PaperPosition', [0 0 14 5.5]);  % deterministic size
print(fig, fullfile('figures', 'compare_bin_sizes.svg'), '-dsvg');
print(fig, fullfile('figures', 'compare_bin_sizes.png'), '-dpng', '-r100');  % 1400x550 px (<=1600 rule)
writetable(T, fullfile('figures', 'compare_bin_sizes_summary.csv'));
save(fullfile('figures', 'compare_bin_sizes_metrics.mat'), 'all_metrics', 'T', 'verdicts', '-v7.3');
fprintf('\nSaved: figures/compare_bin_sizes.{svg,png}, _summary.csv, _metrics.mat\n');

% ---------------------------------------------------------------
function c = centres_of(bs, corridor_end)
    e = 0:bs:corridor_end;
    e(end) = corridor_end + bs;
    c = e(1:end-1) + diff(e)/2;
end
