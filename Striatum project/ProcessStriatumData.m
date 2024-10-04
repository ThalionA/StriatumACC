%% Load

if ~exist("all_data", 'var')
    load all_data.mat
end

%%
n_animals = size(all_data, 2);
plot_summary_fig = false;

if plot_summary_fig
    figure
    t = tiledlayout(7, n_animals, "TileIndexing", "columnmajor");
end


bin_size = 4; % 2.5cm bins
bin_edges = 0:bin_size:200;
bin_edges(end) = 202;
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;
num_bins = numel(bin_centres);



for ianimal = 1:n_animals

    % cut data per trial
    n_vr_datapoints = length(all_data(ianimal).corrected_vr_time);
    n_npx_datapoints = length(all_data(ianimal).npx_time);

    changeIdx_vr = [find(diff(all_data(ianimal).vr_trial) ~= 0), n_vr_datapoints];

    n_trials = numel(changeIdx_vr);

    trialStartIdx_vr = [1, changeIdx_vr(1:end-1) + 1];
    trialEndIdx_vr = changeIdx_vr;
    trialLengths_vr = changeIdx_vr - trialStartIdx_vr + 1;

    trialTimes_vr = mat2cell(all_data(ianimal).corrected_vr_time, 1, trialLengths_vr);
    trial_times_zeroed = cellfun(@(x) x - x(1), trialTimes_vr, UniformOutput=false);
    trialStartTimes_vr = all_data(ianimal).corrected_vr_time(trialStartIdx_vr);
    trialEndTimes_vr = all_data(ianimal).corrected_vr_time(trialEndIdx_vr);

    trialDurations_vr = (trialEndTimes_vr - trialStartTimes_vr)/1000;
    trial_licks = mat2cell(all_data(ianimal).corrected_licks, 1, trialLengths_vr);
    trial_position = mat2cell(all_data(ianimal).vr_position, 1, trialLengths_vr);
    trial_reward = mat2cell(all_data(ianimal).vr_reward, 1, trialLengths_vr);
    trial_world = mat2cell(all_data(ianimal).vr_world, 1, trialLengths_vr);

    npxStartIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialStartTimes_vr, 'nearest', 'extrap');
    npxEndIdx = interp1(all_data(ianimal).npx_time, 1:n_npx_datapoints, trialEndTimes_vr, 'nearest', 'extrap');

    binned_spikes_trials = arrayfun(@(s,e) all_data(ianimal).final_spikes(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
    npx_times_trials = cellfun(@(x) 0:size(x, 2)-1, binned_spikes_trials, 'UniformOutput', false);

    is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
    is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

    final_spikes_dms = all_data(ianimal).final_spikes(is_dms, :);
    final_spikes_acc = all_data(ianimal).final_spikes(is_acc, :);

    binned_spikes_trials_dms = arrayfun(@(s,e) final_spikes_dms(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);
    binned_spikes_trials_acc = arrayfun(@(s,e) final_spikes_acc(:, s:e), npxStartIdx, npxEndIdx, 'UniformOutput', false);

    trial_lick_no = cellfun(@sum, trial_licks);

    trial_success = cellfun(@max, trial_reward);

    trial_average_fr_dms = cellfun(@(x) mean(sum(x, 2)), binned_spikes_trials_dms)./trialDurations_vr;
    trial_sem_fr_dms = cellfun(@(x) sem(sum(x, 2)), binned_spikes_trials_dms)./trialDurations_vr;

    trial_average_fr_acc = cellfun(@(x) mean(sum(x, 2)), binned_spikes_trials_acc)./trialDurations_vr;
    trial_sem_fr_acc = cellfun(@(x) sem(sum(x, 2)), binned_spikes_trials_acc)./trialDurations_vr;

    mov_window_size = 5;
    [~, duration_peaks] = findpeaks(movmean(trialDurations_vr, mov_window_size), 'MinPeakProminence', 5, 'Annotate', 'peaks');
    trial_licks_change = find(movmean(trial_lick_no, 10) < 20, 1);
    trial_success_change = find(movmean(trial_success, 10) < 0.5, 1);

    [~, loc1] = min(abs(duration_peaks - trial_licks_change));
    [~, loc2] = min(abs(duration_peaks - trial_success_change));
    most_likely_change_duration = mean([duration_peaks(loc1), duration_peaks(loc2)]);

    all_data(ianimal).change_point_mean = floor(mean([trial_licks_change, trial_success_change, most_likely_change_duration]));

    corridor_start_idx_vr = cellfun(@(x) find(x > 6, 1), trial_world);
    corridor_start_time_vr = cellfun(@(x, y) x(find(y > 6, 1) + 1), trial_times_zeroed, trial_world);

    corridor_start_idx_npx = nan(size(corridor_start_idx_vr));
    binned_spikes_dark = cell(1, n_trials);
    binned_spikes_corridor = cell(1, n_trials);

    trial_position_corridor = cell(1, n_trials);
    trial_position_dark = cell(1, n_trials);

    trial_licks_corridor = cell(1, n_trials);
    trial_licks_dark = cell(1, n_trials);

    trial_reward_corridor = cell(1, n_trials);
    trial_reward_dark = cell(1, n_trials);

    trial_times_corridor = cell(1, n_trials);
    trial_times_dark = cell(1, n_trials);


    for itrial = 1:n_trials
        [~, corridor_start_idx_npx(itrial)] = min(abs(npx_times_trials{itrial} - corridor_start_time_vr(itrial)));
        binned_spikes_dark{itrial} = binned_spikes_trials{itrial}(:, 1:corridor_start_idx_npx(itrial)-1);
        binned_spikes_corridor{itrial} = binned_spikes_trials{itrial}(:, corridor_start_idx_npx(itrial):end);

        trial_position_corridor{itrial} = trial_position{itrial}(corridor_start_idx_vr(itrial):end);
        trial_position_dark{itrial} = trial_position{itrial}(1:corridor_start_idx_vr(itrial)-1);

        trial_licks_corridor{itrial} = trial_licks{itrial}(corridor_start_idx_vr(itrial):end);
        trial_licks_dark{itrial} = trial_licks{itrial}(1:corridor_start_idx_vr(itrial)-1);

        trial_reward_corridor{itrial} = trial_reward{itrial}(corridor_start_idx_vr(itrial):end);
        trial_reward_dark{itrial} = trial_reward{itrial}(1:corridor_start_idx_vr(itrial)-1);

        trial_times_corridor{itrial} = trial_times_zeroed{itrial}(corridor_start_idx_vr(itrial):end);
        trial_times_dark{itrial} = trial_times_zeroed{itrial}(1:corridor_start_idx_vr(itrial)-1);

        % Bin positions
        [~, ~, bin_idx] = histcounts(trial_position_corridor{itrial}, bin_edges);

        spatial_binned_licks{itrial} = nan(1, num_bins);
        spatial_binned_durations{itrial} = nan(1, num_bins);

        trial_times_corridor_zeroed = trial_times_corridor{itrial} - trial_times_corridor{itrial}(1);
        % Bin data in space
        for ibin = 1:num_bins
            idx_in_bin = (bin_idx == ibin);
            if any(idx_in_bin)
                

                % Compute total licks
                spatial_binned_licks{itrial}(ibin) = sum(trial_licks_corridor{itrial}(idx_in_bin));
                bin_times = trial_times_corridor_zeroed(idx_in_bin);
                spatial_binned_durations{itrial}(ibin) = (bin_times(end) - bin_times(1))/1000;

                npx_bin_start_idx = floor(bin_times(1))
                binned_spikes_corridor{itrial}
            end
        end


    end

    fr_dark_trials = cellfun(@(x) sum(x, 2)/(size(x, 2)/1000), binned_spikes_dark, 'UniformOutput', false);
    fr_dark_trials = cat(1, [fr_dark_trials{:}]);

    fr_corridor_trials = cellfun(@(x) sum(x, 2)/(size(x, 2)/1000), binned_spikes_corridor, 'UniformOutput', false);
    fr_corridor_trials = cat(1, [fr_corridor_trials{:}]);




    if plot_summary_fig

        nexttile
        plot(movmean(trialDurations_vr, mov_window_size))
        xline(duration_peaks)
        title(num2str(all_data(ianimal).mouseid))
        axis tight
        ylim([5, 40])
        ylabel('trial duration (s)')

        nexttile
        plot(movmean(trial_lick_no, mov_window_size))
        ylabel('lick #')
        axis tight
        xline(trial_licks_change)

        nexttile
        plot(movmean(trial_success, mov_window_size))
        ylabel('reward')
        axis tight
        ylim([-0.2, 1.2])
        xline(trial_success_change)

        nexttile
        shadedErrorBar(1:n_trials, movmean(trial_average_fr_dms, mov_window_size), movmean(trial_sem_fr_dms, mov_window_size))
        % plot(movmean(trial_average_fr_dms, mov_window_size))
        ylabel('DMS fr')
        axis tight

        nexttile
        shadedErrorBar(1:n_trials, movmean(trial_average_fr_acc, mov_window_size), movmean(trial_sem_fr_acc, mov_window_size))
        % plot(movmean(trial_average_fr_acc, mov_window_size))
        ylabel('ACC fr')
        axis tight

        nexttile
        shadedErrorBar(1:n_trials, mean(fr_corridor_trials), sem(fr_corridor_trials))
        axis tight
        xlabel('trial no')
        ylabel('firing rate')
        title('corridor')

        nexttile
        shadedErrorBar(1:n_trials, mean(fr_dark_trials), sem(fr_dark_trials))
        axis tight
        xlabel('trial no')
        ylabel('firing rate')
        title('dark')
    end

end

if plot_summary_fig
    xlabel(t, 'trial #')
end