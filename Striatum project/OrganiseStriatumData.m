all_mouse_ids = [523, 614, 727, 730, 823];
num_mice = numel(all_mouse_ids);

%% import the depth data
opts = delimitedTextImportOptions("NumVariables", 9);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["MouseID", "ACCStart", "ACCEnd", "StriatumStart", "StriatumEnd", "DMSStart", "DMSEnd", "DLSStart", "DLSEnd"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
NeuropixelsDepthData = readtable("/RawData/Neuropixels_Depth_Data.csv", opts);

%% process each mouse

for imouse = 1:num_mice

    filename = ['./RawData/' num2str(all_mouse_ids(imouse)) '_raw.mat'];
    load(filename);

    num_units = size(binned_spikes, 1);

    % Assign areas to units based on depths
    unit_areas = cell(1, num_units);

    depths = NeuropixelsDepthData(NeuropixelsDepthData.MouseID == all_mouse_ids(imouse), :);

    unit_areas(goodcluster2(:, 2) >= depths.DMSStart & goodcluster2(:, 2) <= depths.DMSEnd) = {'DMS'};
    unit_areas(goodcluster2(:, 2) >= depths.ACCStart & goodcluster2(:, 2) <= depths.ACCEnd) = {'ACC'};

    units_to_keep = cellfun(@(x) ~isempty(x), unit_areas);

    % Keep only aligned spike data from DMS and ACC
    npx_start_frame = ceil(VR_times_synched(1)*1000);
    npx_end_frame = floor(VR_times_synched(end)*1000);

    final_spikes = binned_spikes(units_to_keep, npx_start_frame:npx_end_frame);
    final_areas = unit_areas(units_to_keep);
    clear binned_spikes
    
    % Correct time
    corrected_vr_time = (VR_times_synched - VR_times_synched(1))*1000;
    npx_time = 0:1:size(final_spikes, 2)-1;

    % Process licks
    corrected_licks = process_licks(VR_data(8, :) >= 1, corrected_vr_time, 100);

    all_data(imouse).mouseid = all_mouse_ids(imouse);
    all_data(imouse).final_spikes = final_spikes;
    all_data(imouse).final_areas = final_areas;
    all_data(imouse).npx_time = npx_time;
    all_data(imouse).corrected_vr_time = corrected_vr_time;
    all_data(imouse).corrected_licks = corrected_licks';
    all_data(imouse).vr_position = VR_data(2, :);
    all_data(imouse).vr_world = VR_data(5, :);
    all_data(imouse).vr_reward = VR_data(6, :);
    all_data(imouse).vr_trial = VR_data(7, :);


    % Bin in space (2.5cm)


    % Bin in time (?)


    % Cut in trials


    % Smooth spikes
    

    % Lick-triggered average


    % Reward-triggered average

    % Simple processing for average firing rates across entire experiment
    average_firing_rates = sum(final_spikes, 2)/((corrected_vr_time(end)-corrected_vr_time(1))/1000);
    average_DMS_fr = average_firing_rates(strcmp(final_areas, 'DMS'));
    average_ACC_fr = average_firing_rates(strcmp(final_areas, 'ACC'));

    % Simple average lick rate
    average_lick_rate = sum(corrected_licks)/((corrected_vr_time(end)-corrected_vr_time(1))/1000);

    % figure
    % my_errorbar_plot({average_ACC_fr, average_DMS_fr})
    % xticklabels({'ACC', 'DMS'})
    % ylabel('average FR')
    % [~, pval] = ttest2(average_ACC_fr, average_DMS_fr);
    % sigstar([1, 2], pval)
    % save_to_svg(['avg_fr_' num2str(all_mouse_ids(imouse))])

    all_data(imouse).avg_fr_all = average_firing_rates;
    all_data(imouse).average_DMS_fr = average_DMS_fr;
    all_data(imouse).average_ACC_fr = average_ACC_fr;
    all_data(imouse).average_lick_rate = average_lick_rate;

end
