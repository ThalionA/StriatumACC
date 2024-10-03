%% Load

if ~exist("all_data", 'var')
    load all_data.mat
end

n_animals = size(all_data, 2);
figure
t = tiledlayout(1, n_animals);

for ianimal = 1:n_animals
    
    % cut data per trial
    n_vr_datapoints = length(all_data(ianimal).corrected_vr_time);
    changeIdx_vr = [find(diff(all_data(ianimal).vr_trial) ~= 0), n_vr_datapoints];
    startIdx_vr = [1, changeIdx_vr(1:end-1) + 1];
    trialLengths_vr = changeIdx_vr - startIdx_vr + 1;

    trialTimes_vr = mat2cell(all_data(ianimal).corrected_vr_time, 1, trialLengths_vr);
    trialDurations_vr = cellfun(@(x) x(end)-x(1), trialTimes_vr)/1000;
    trial_licks = mat2cell(all_data(ianimal).corrected_licks, 1, trialLengths_vr);
    trial_position = mat2cell(all_data(ianimal).vr_position, 1, trialLengths_vr);
    trial_reward = mat2cell(all_data(ianimal).vr_reward, 1, trialLengths_vr);
    trial_world = mat2cell(all_data(ianimal).vr_world, 1, trialLengths_vr);


    nexttile
    findpeaks(movmean(trialDurations_vr, 10), 'MinPeakProminence', 5)
    title(num2str(all_data(ianimal).mouseid))
    ylim([0, 40])
    
end