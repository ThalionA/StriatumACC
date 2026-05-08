%% Visualise neuron type clustering

% Self-sufficient loading (2026-05-07).
if ~exist('cfg', 'var') || isempty(cfg)
    cfg = project_cfg();
end
if ~exist('control_data_raw', 'var') || isempty(control_data_raw)
    if isfile(cfg.control_data_file)
        fprintf('Loading control data from %s ...\n', cfg.control_data_file);
        S = load(cfg.control_data_file, 'preprocessed_data');
        control_data_raw = S.preprocessed_data;
    else
        error('neurontype_classification:NoControlData', ...
              'control_data_raw not in workspace and %s not found.', cfg.control_data_file);
    end
end

all_neurontype_info = {control_data_raw(:).final_neurontypes};
all_neurontype_info = cat(1, all_neurontype_info{:});

figure;
hold on
for itype = 1:4
    neurontype_idx = all_neurontype_info(:, 5) == itype;
    scatter3(all_neurontype_info(neurontype_idx, 4), all_neurontype_info(neurontype_idx, 3), all_neurontype_info(neurontype_idx, 2), 50, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.75);
end
xlabel('post spike suppression (ms)')
ylabel('peak to trough duration (ms)')
zlabel('prop ISI > 2s')
legend({'MSN', 'FS', 'TAN', 'UIN'})
grid on
ylim([0, 2])
view(60, 15)

%% Proportion per area

all_isdms = {control_data_raw(:).is_dms};
all_isdms = cat(2, all_isdms{:})';

all_isdls = {control_data_raw(:).is_dls};
all_isdls = cat(2, all_isdls{:})';

all_isacc = {control_data_raw(:).is_acc};
all_isacc = cat(2, all_isacc{:})';

% Optional probe-2 areas (V1, CA1, DG, ...) — pad missing animals with false.
% Generalised 2026-05-08 from a V1-only block.
all_isv1  = concat_optional_field(control_data_raw, 'is_v1');
all_isca1 = concat_optional_field(control_data_raw, 'is_ca1');
all_isdg  = concat_optional_field(control_data_raw, 'is_dg');

dms_neurontypes = [sum(all_neurontype_info(:, 5) == 1 & all_isdms), sum(all_neurontype_info(:, 5) == 2 & all_isdms), sum(all_neurontype_info(:, 5) == 3 & all_isdms), sum(all_neurontype_info(:, 5) == 4 & all_isdms)];
dls_neurontypes = [sum(all_neurontype_info(:, 5) == 1 & all_isdls), sum(all_neurontype_info(:, 5) == 2 & all_isdls), sum(all_neurontype_info(:, 5) == 3 & all_isdls), sum(all_neurontype_info(:, 5) == 4 & all_isdls)];
acc_neurontypes = [sum(all_neurontype_info(:, 5) == 1 & all_isacc), sum(all_neurontype_info(:, 5) == 2 & all_isacc), sum(all_neurontype_info(:, 5) == 3 & all_isacc), sum(all_neurontype_info(:, 5) == 4 & all_isacc)];
% Probe-2 areas (V1, CA1, DG) have no MSN/FSN/TAN classification — all
% units fall under "Unclassified" so the totals still reconcile.
v1_neurontypes  = [0, 0, 0, sum(all_isv1)];
ca1_neurontypes = [0, 0, 0, sum(all_isca1)];
dg_neurontypes  = [0, 0, 0, sum(all_isdg)];

figure
bar([dms_neurontypes', dls_neurontypes', acc_neurontypes', ...
     v1_neurontypes', ca1_neurontypes', dg_neurontypes']', 'stacked')

xticklabels({'DMS', 'DLS', 'ACC', 'V1', 'CA1', 'DG'})
legend({'MSN', 'FS', 'TAN', 'UIN'})
ylabel('unit count')

% --- Local helper for optional area-flag concatenation ---
function y = concat_optional_field(data_struct, fname)
    y = arrayfun(@(s) ...
        (isfield(s, fname) && ~isempty(s.(fname))) * logical(s.(fname)) ...
        + ~(isfield(s, fname) && ~isempty(s.(fname))) * false(size(s.is_dms)), ...
        data_struct, 'UniformOutput', false);
    y = cat(2, y{:});
    if size(y, 1) == 1, y = y'; end
end

%% Firing rate distribution
all_neurontypes = {'MSN', 'FS', 'TAN', 'UIN'};
figure
t = tiledlayout(4, 1);
lala = colororder;

for itype = 1:4
    nexttile
    histogram(all_neurontype_info(all_neurontype_info(:, 5) == itype, 1), 'Normalization', 'count', 'FaceColor', lala(itype, :))
    ax = gca;
    ax.XScale = 'log';
    if itype == 1
        ylim([0, 300])
    elseif itype == 2
        ylim([0, 30])
    else 
        axis tight
    end
    xlim([0.2, 1e2])

    box off
    title(all_neurontypes{itype})
end
ylabel(t, 'unit count')
xlabel(t, 'average firing rate (Hz)')
