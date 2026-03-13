%% Canonical correlation analysis

reduced_components = 1;


for ianimal = 1:12
    % ianimal = 6;
    animal_data = task_data(ianimal).spatial_binned_fr_all;
    [~, ~, num_trials] = size(animal_data);
    diseng_point = min([task_data(ianimal).change_point_mean, num_trials]);

    animal_data = task_data(ianimal).spatial_binned_fr_all(:, :, 1:diseng_point);
    [~, num_bins, num_trials] = size(animal_data);

    is_dms = task_data(ianimal).is_dms;
    is_dls = task_data(ianimal).is_dls;
    is_acc = task_data(ianimal).is_acc;

    n_dms_units = sum(is_dms);
    n_dls_units = sum(is_dls);
    n_acc_units = sum(is_acc);

    if any([n_dms_units, n_dls_units, n_acc_units] < reduced_components)
        continue
    end

    animal_data_dms = animal_data(is_dms, :, :);
    [~,dms_reduced,~] = pca(animal_data_dms(:, :)', NumComponents=reduced_components);
    dms_reduced = reshape(dms_reduced', reduced_components, num_bins, num_trials);

    animal_data_dls = animal_data(is_dls, :, :);
    [~,dls_reduced,~] = pca(animal_data_dls(:, :)', NumComponents=reduced_components);
    dls_reduced = reshape(dls_reduced', reduced_components, num_bins, num_trials);

    animal_data_acc = animal_data(is_acc, :, :);
    [~,acc_reduced,~] = pca(animal_data_acc(:, :)', NumComponents=reduced_components);
    acc_reduced = reshape(acc_reduced', reduced_components, num_bins, num_trials);


    learning_point = learning_points_task{ianimal};

    dms_spatial_activity_naive = dms_reduced(:, :, 1:10);
    dls_spatial_activity_naive = dls_reduced(:, :, 1:10);
    acc_spatial_activity_naive = acc_reduced(:, :, 1:10);

    dms_spatial_activity_inter = dms_reduced(:, :, learning_point - 10:learning_point -1);
    dls_spatial_activity_inter = dls_reduced(:, :, learning_point - 10:learning_point -1);
    acc_spatial_activity_inter = acc_reduced(:, :, learning_point - 10:learning_point -1);

    dms_spatial_activity_exp = dms_reduced(:, :, learning_point + 1:learning_point + 10);
    dls_spatial_activity_exp = dls_reduced(:, :, learning_point + 1:learning_point + 10);
    acc_spatial_activity_exp = acc_reduced(:, :, learning_point + 1:learning_point + 10);

    dms_dls_cca_all_bins = nan(reduced_components, num_bins);
    dms_acc_cca_all_bins = nan(reduced_components, num_bins);
    acc_dls_cca_all_bins = nan(reduced_components, num_bins);

    dms_dls_cca_all_trials = nan(reduced_components, num_trials);
    dms_acc_cca_all_trials = nan(reduced_components, num_trials);
    acc_dls_cca_all_trials = nan(reduced_components, num_trials);

    dms_dls_cca_naive_bins = nan(reduced_components, num_bins);
    dms_acc_cca_naive_bins = nan(reduced_components, num_bins);
    acc_dls_cca_naive_bins = nan(reduced_components, num_bins);

    dms_dls_cca_inter_bins = nan(reduced_components, num_bins);
    dms_acc_cca_inter_bins = nan(reduced_components, num_bins);
    acc_dls_cca_inter_bins = nan(reduced_components, num_bins);

    dms_dls_cca_exp_bins = nan(reduced_components, num_bins);
    dms_acc_cca_exp_bins = nan(reduced_components, num_bins);
    acc_dls_cca_exp_bins = nan(reduced_components, num_bins);


    for itrial = 1:num_trials
        trial_window = itrial + (-3:3);
        trial_window = trial_window(trial_window >= 1 & trial_window <= num_trials);

        non_nan_bins = ~any(isnan(acc_reduced(:, :, trial_window)), [1, 3]);

        dms_reduced_pooled = dms_reduced(:, non_nan_bins, trial_window);
        dls_reduced_pooled = dls_reduced(:, non_nan_bins, trial_window);
        acc_reduced_pooled = acc_reduced(:, non_nan_bins, trial_window);

        
        
            [A,B,r,U,V] = canoncorr(dms_reduced_pooled(:, :)', dls_reduced_pooled(:, :)');
            dms_dls_cca_all_trials(1:length(r), itrial) = r;

            [A,B,r,U,V] = canoncorr(dms_reduced_pooled(:, :)', acc_reduced_pooled(:, :)');
            dms_acc_cca_all_trials(1:length(r), itrial) = r;

            [A,B,r,U,V] = canoncorr(acc_reduced_pooled(:, :)', dls_reduced_pooled(:, :)');
            acc_dls_cca_all_trials(1:length(r), itrial) = r;
    end


    non_nan_trials = squeeze(~any(isnan(acc_reduced(:, :, :)), [1, 2]));

    for ibin = 1:num_bins

        bin_window = ibin + (-3:3);
        bin_window = bin_window(bin_window >= 1 & bin_window <= num_bins);

        dms_reduced_pooled = dms_reduced(:, bin_window, non_nan_trials);
        dls_reduced_pooled = dls_reduced(:, bin_window, non_nan_trials);
        acc_reduced_pooled = acc_reduced(:, bin_window, non_nan_trials);

        try
            [A,B,r,U,V] = canoncorr(dms_reduced_pooled(:, :)', dls_reduced_pooled(:, :)');
            dms_dls_cca_all_bins(1:length(r), ibin) = r;
            for idim = 1:reduced_components
                dms_dls_cca_naive_bins(idim, ibin) = corr(U(1:10, idim), V(1:10, idim));
                dms_dls_cca_inter_bins(idim, ibin) = corr(U(learning_point - 10:learning_point -1, idim), V(learning_point - 10:learning_point -1, idim));
                dms_dls_cca_exp_bins(idim, ibin) = corr(U(learning_point + 1:learning_point + 10, idim), V(learning_point + 1:learning_point + 10, idim));
            end


        catch
            warning('dms-dls cca failed on bin %d', ibin)
        end

        try
            [A,B,r,U,V] = canoncorr(dms_reduced_pooled(:, :)', acc_reduced_pooled(:, :)');
            dms_acc_cca_all_bins(1:length(r), ibin) = r;
            for idim = 1:reduced_components
                dms_acc_cca_naive_bins(idim, ibin) = corr(U(1:10, idim), V(1:10, idim));
                dms_acc_cca_inter_bins(idim, ibin) = corr(U(learning_point - 10:learning_point -1, idim), V(learning_point - 10:learning_point -1, idim));
                dms_acc_cca_exp_bins(idim, ibin) = corr(U(learning_point + 1:learning_point + 10, idim), V(learning_point + 1:learning_point + 10, idim));
            end
        catch
            warning('dms-acc cca failed on bin %d', ibin)
        end

        try
            [A,B,r,U,V] = canoncorr(acc_reduced_pooled(:, :)', dls_reduced_pooled(:, :)');
            acc_dls_cca_all_bins(1:length(r), ibin) = r;
            for idim = 1:reduced_components
                acc_dls_cca_naive_bins(idim, ibin) = corr(U(1:10, idim), V(1:10, idim));
                acc_dls_cca_inter_bins(idim, ibin) = corr(U(learning_point - 10:learning_point -1, idim), V(learning_point - 10:learning_point -1, idim));
                acc_dls_cca_exp_bins(idim, ibin) = corr(U(learning_point + 1:learning_point + 10, idim), V(learning_point + 1:learning_point + 10, idim));
            end
        catch
            warning('acc-dls cca failed on bin %d', ibin)
        end

    end



    % Plotting

    figure
    t = tiledlayout(3, 3, "TileSpacing", "compact", "Padding", "compact");
    nexttile
    plot(dms_dls_cca_naive_bins(1, :)')
    title('DMS-DLS naive')

    nexttile
    plot(dms_acc_cca_naive_bins(1, :)')
    title('DMS-ACC naive')

    nexttile
    plot(acc_dls_cca_naive_bins(1, :)')
    title('ACC-DLS naive')

    nexttile
    plot(dms_dls_cca_inter_bins(1, :)')
    title('DMS-DLS intermediate')

    nexttile
    plot(dms_acc_cca_inter_bins(1, :)')
    title('DMS-ACC intermediate')

    nexttile
    plot(acc_dls_cca_inter_bins(1, :)')
    title('ACC-DLS intermediate')

    nexttile
    plot(dms_dls_cca_exp_bins(1, :)')
    title('DMS-DLS expert')

    nexttile
    plot(dms_acc_cca_exp_bins(1, :)')
    title('DMS-ACC expert')

    nexttile
    plot(acc_dls_cca_exp_bins(1, :)')
    title('ACC-DLS expert')
    linkaxes

    title(t, sprintf('animal %d', ianimal))

    figure; t = tiledlayout('flow');
    nexttile; plot(1:num_bins, dms_dls_cca_all_bins(1, :)'); title('DMS-DLS'); xline([20, 25])
    nexttile; plot(1:num_bins, dms_acc_cca_all_bins(1, :)'); title('DMS-ACC'); xline([20, 25])
    nexttile; plot(1:num_bins, acc_dls_cca_all_bins(1, :)'); title('ACC-DLS'); xline([20, 25]); ylim([0, 1]); linkaxes
    title(t, sprintf('animal %d', ianimal))
    xlabel(t, 'spatial bins')
    


    figure; t = tiledlayout('flow'); nexttile; plot(dms_dls_cca_all_trials(1, 1:learning_point+10)'); title('DMS-DLS'); xline(learning_point, 'r')
    nexttile; plot(dms_acc_cca_all_trials(1, 1:learning_point+10)'); title('DMS-ACC'); xline(learning_point, 'r')
    nexttile; plot(acc_dls_cca_all_trials(1, 1:learning_point+10)'); title('ACC-DLS'); xline(learning_point, 'r'); ylim([0, 1]); linkaxes
    title(t, sprintf('animal %d', ianimal))
    xlabel(t, 'trials')
    
    animal_dms_dls_bins(ianimal, :) = dms_dls_cca_all_bins(1, :);
    animal_dms_acc_bins(ianimal, :) = dms_acc_cca_all_bins(1, :);
    animal_acc_dls_bins(ianimal, :) = dms_acc_cca_all_bins(1, :);

    animal_dms_dls_bins_trials(ianimal, :, :) = [dms_dls_cca_naive_bins(1, :); dms_dls_cca_inter_bins(1, :); dms_dls_cca_exp_bins(1, :)];
    animal_dms_acc_bins_trials(ianimal, :, :) = [dms_acc_cca_naive_bins(1, :); dms_acc_cca_inter_bins(1, :); dms_acc_cca_exp_bins(1, :)];
    animal_acc_dls_bins_trials(ianimal, :, :) = [acc_dls_cca_naive_bins(1, :); acc_dls_cca_inter_bins(1, :); acc_dls_cca_exp_bins(1, :)];


    animal_dms_dls_trials(ianimal, :) = dms_dls_cca_all_trials(1, [1:10, learning_point - 10:learning_point-1, learning_point + 1:learning_point+10]);
    animal_dms_acc_trials(ianimal, :) = dms_acc_cca_all_trials(1, [1:10, learning_point - 10:learning_point-1, learning_point + 1:learning_point+10]);
    animal_acc_dls_trials(ianimal, :) = acc_dls_cca_all_trials(1, [1:10, learning_point - 10:learning_point-1, learning_point + 1:learning_point+10]);

end

%%
figure
t = tiledlayout('flow');
nexttile
shadedErrorBar(1:num_bins, mean(animal_dms_dls_bins, 'omitmissing'), sem(animal_dms_dls_bins, 1))
title('DMS-DLS')
xline([20, 25])
nexttile
shadedErrorBar(1:num_bins, mean(animal_dms_acc_bins, 'omitmissing'), sem(animal_dms_acc_bins, 1))
title('DMS-ACC')
xline([20, 25])
nexttile
shadedErrorBar(1:num_bins, mean(animal_acc_dls_bins, 'omitmissing'), sem(animal_acc_dls_bins, 1))
title('ACC-DLS')
xline([20, 25])
linkaxes
xlabel(t, 'spatial bins')

figure
t = tiledlayout('flow');
nexttile
shadedErrorBar(1:30, mean(animal_dms_dls_trials, 'omitmissing'), sem(animal_dms_dls_trials, 1))
title('DMS-DLS')
xline([10, 20])
nexttile
shadedErrorBar(1:30, mean(animal_dms_acc_trials, 'omitmissing'), sem(animal_dms_acc_trials, 1))
title('DMS-ACC')
xline([10, 20])
nexttile
shadedErrorBar(1:30, mean(animal_acc_dls_trials, 'omitmissing'), sem(animal_acc_dls_trials, 1))
title('ACC-DLS')
xline([10, 20])
linkaxes
xlabel(t, 'trials')


figure
t = tiledlayout(3, 3);
for iepoch = 1:3
    nexttile
    shadedErrorBar(1:num_bins, mean(squeeze(animal_dms_dls_bins_trials(:, iepoch, :)), 'omitmissing'), sem(squeeze(animal_dms_dls_bins_trials(:, iepoch, :)), 1))
    title('DMS-DLS')
    xline([20, 25])



    nexttile
    shadedErrorBar(1:num_bins, mean(squeeze(animal_dms_acc_bins_trials(:, iepoch, :)), 'omitmissing'), sem(squeeze(animal_dms_acc_bins_trials(:, iepoch, :)), 1))
    title('DMS-ACC')
    xline([20, 25])


    nexttile
    shadedErrorBar(1:num_bins, mean(squeeze(animal_acc_dls_bins_trials(:, iepoch, :)), 'omitmissing'), sem(squeeze(animal_acc_dls_bins_trials(:, iepoch, :)), 1))
    title('ACC-DLS')
    xline([20, 25])
end
linkaxes
xlabel(t, 'spatial bins')
