%% Visualise neuron type clustering

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

dms_neurontypes = [sum(all_neurontype_info(:, 5) == 1 & all_isdms), sum(all_neurontype_info(:, 5) == 2 & all_isdms), sum(all_neurontype_info(:, 5) == 3 & all_isdms), sum(all_neurontype_info(:, 5) == 4 & all_isdms)];
dls_neurontypes = [sum(all_neurontype_info(:, 5) == 1 & all_isdls), sum(all_neurontype_info(:, 5) == 2 & all_isdls), sum(all_neurontype_info(:, 5) == 3 & all_isdls), sum(all_neurontype_info(:, 5) == 4 & all_isdls)];
acc_neurontypes = [sum(all_neurontype_info(:, 5) == 1 & all_isacc), sum(all_neurontype_info(:, 5) == 2 & all_isacc), sum(all_neurontype_info(:, 5) == 3 & all_isacc), sum(all_neurontype_info(:, 5) == 4 & all_isacc)];

figure
bar([dms_neurontypes', dls_neurontypes', acc_neurontypes']', 'stacked')

xticklabels({'DMS', 'DLS', 'ACC'})
legend({'MSN', 'FS', 'TAN', 'UIN'})
ylabel('unit count')

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
