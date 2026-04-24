% OrganiseStriatumData.m
% Extracts and organises relevant task data for the single-session learning 
% Striatum-ACC dataset, now including V1 multi-probe integration.

clear all;
close all;

%% ================= Configuration & Imports =================
all_mouse_ids = [523, 614, 624, 727, 730, 731, 822, 823, 1105, 1106, 1201, 1206, 1212, 409, 418, 703];
num_mice = numel(all_mouse_ids);

% --- Import Depth Data (Probe 1) ---
opts = delimitedTextImportOptions("NumVariables", 9);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["MouseID", "ACCStart", "ACCEnd", "StriatumStart", "StriatumEnd", "DMSStart", "DMSEnd", "DLSStart", "DLSEnd"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
depth_file_path = "./RawData/Neuropixels_Depth_Data.csv";

if isfile(depth_file_path)
    NeuropixelsDepthData = readtable(depth_file_path, opts);
else
    error('Depth data file not found at: %s', depth_file_path);
end

% --- Import Depth Data (Probe 2 - V1) ---
v1_depth_file = "./RawData/Neuropixels_V1_Depth_Data.csv";
if isfile(v1_depth_file)
    opts_v1 = detectImportOptions(v1_depth_file);
    opts_v1.VariableNames = {'MouseID', 'V1Start', 'V1End'};
    V1DepthData = readtable(v1_depth_file, opts_v1);
else
    V1DepthData = table();
    warning('Neuropixels_V1_Depth_Data.csv not found. Continuing without V1 integration.');
end

% --- Preallocate Output Structure ---
all_data = struct('mouseid', cell(1, num_mice), ...
                  'final_spikes', [], 'final_areas', [], 'final_neurontypes', [], ...
                  'npx_time', [], 'corrected_vr_time', [], 'corrected_licks', [], ...
                  'vr_position', [], 'vr_world', [], 'vr_reward', [], 'vr_trial', [], ...
                  'avg_fr_all', [], 'average_DMS_fr', [], 'average_DLS_fr', [], ...
                  'average_ACC_fr', [], 'average_V1_fr', [], 'average_lick_rate', []);

%% ================= Processing Loop =================
for imouse = 1:num_mice
    curr_mouse = all_mouse_ids(imouse);
    fprintf('Processing animal %d/%d (ID: %d)...\n', imouse, num_mice, curr_mouse);
    
    % --- 1. Load Raw Data (Probe 1) ---
    raw_filename = ['./RawData/' num2str(curr_mouse) '_raw.mat'];
    
    if ~isfile(raw_filename)
        warning('Raw file not found for mouse %d. Skipping.', curr_mouse);
        continue;
    end
    
    RawDat = load(raw_filename, 'binned_spikes', 'goodcluster2', 'VR_times_synched', 'VR_data');
    num_units_p1 = size(RawDat.binned_spikes, 1);

    % --- 2. Load Raw Data (Probe 2 - V1) ---
    v1_filename = ['./RawData/' num2str(curr_mouse) '_V1_raw.mat'];
    has_v1 = isfile(v1_filename);
    if has_v1
        V1Dat = load(v1_filename, 'binned_spikes', 'goodcluster2');
    end
    
    % --- 3. Load Neuron Types ---
    nt_filename = ['./RawData/' num2str(curr_mouse) '_neurontype2025.mat'];
    raw_neurontype = [];
    if isfile(nt_filename)
        tmp_nt = load(nt_filename, 'neurontype');
        if isfield(tmp_nt, 'neurontype')
            raw_neurontype = tmp_nt.neurontype;
            if size(raw_neurontype, 1) ~= num_units_p1
                warning('Mismatch in unit count for neurontype file (Mouse %d). Filling with NaNs.', curr_mouse);
                raw_neurontype = nan(num_units_p1, 1);
            end
            if size(raw_neurontype, 2) < 5
                raw_neurontype(raw_neurontype(:,3)>=0.4 & raw_neurontype(:,4)<=40,5) = 1; %MSN
                raw_neurontype(raw_neurontype(:,3)<0.4 & raw_neurontype(:,2)<0.1,5) = 2;  %FSN
                raw_neurontype(raw_neurontype(:,3)>=0.4 & raw_neurontype(:,4)>40,5) = 3;  %TAN
                raw_neurontype(raw_neurontype(:,3)<0.4 & raw_neurontype(:,2)>=0.1,5) = 4; %UIN
            end
        else
            raw_neurontype = nan(num_units_p1, 1);
        end
    else
        raw_neurontype = nan(num_units_p1, 1);
    end
    
    % --- 4. Assign Areas (Probe 1) ---
    unit_areas_p1 = cell(num_units_p1, 1); 
    depths = NeuropixelsDepthData(NeuropixelsDepthData.MouseID == curr_mouse, :);
    
    if isempty(depths)
        warning('No depth data found for mouse %d in CSV.', curr_mouse);
        units_to_keep_p1 = false(num_units_p1, 1); 
    else
        unit_depths_p1 = RawDat.goodcluster2(:, 2);
        is_dms = unit_depths_p1 >= depths.DMSStart & unit_depths_p1 <= depths.DMSEnd;
        is_dls = unit_depths_p1 >= depths.DLSStart & unit_depths_p1 <= depths.DLSEnd;
        is_acc = unit_depths_p1 >= depths.ACCStart & unit_depths_p1 <= depths.ACCEnd;
        
        unit_areas_p1(is_dms) = {'DMS'};
        unit_areas_p1(is_dls) = {'DLS'};
        unit_areas_p1(is_acc) = {'ACC'};
        units_to_keep_p1 = ~cellfun(@isempty, unit_areas_p1);
    end

    % --- 5. Assign Areas (Probe 2 - V1) ---
    if has_v1
        unit_depths_p2 = V1Dat.goodcluster2(:, 2);
        unit_areas_p2 = cell(size(unit_depths_p2, 1), 1);
        
        if ~isempty(V1DepthData)
            v1_depths = V1DepthData(V1DepthData.MouseID == curr_mouse, :);
            if ~isempty(v1_depths) && ~isnan(v1_depths.V1Start(1))
                is_v1 = unit_depths_p2 >= v1_depths.V1Start(1) & unit_depths_p2 <= v1_depths.V1End(1);
            else
                is_v1 = false(size(unit_depths_p2));
            end
        else
            is_v1 = false(size(unit_depths_p2));
        end
        
        unit_areas_p2(is_v1) = {'V1'};
        units_to_keep_p2 = ~cellfun(@isempty, unit_areas_p2);
    end
    
    % --- 6. Slicing and Alignment ---
    npx_start_frame = ceil(RawDat.VR_times_synched(1)*1000);
    npx_end_frame = floor(RawDat.VR_times_synched(end)*1000);
    npx_start_frame = max(1, npx_start_frame);
    npx_end_frame = min(size(RawDat.binned_spikes, 2), npx_end_frame);
    
    % Slice Probe 1
    final_spikes_p1 = RawDat.binned_spikes(units_to_keep_p1, npx_start_frame:npx_end_frame);
    final_areas_p1 = unit_areas_p1(units_to_keep_p1);
    final_nt_p1 = raw_neurontype(units_to_keep_p1, :);

    % Slice Probe 2 and Merge
    if has_v1
        v1_end_frame = min(size(V1Dat.binned_spikes, 2), npx_end_frame);
        if v1_end_frame < npx_end_frame
            final_spikes_p2 = zeros(sum(units_to_keep_p2), npx_end_frame - npx_start_frame + 1);
            final_spikes_p2(:, 1:(v1_end_frame - npx_start_frame + 1)) = V1Dat.binned_spikes(units_to_keep_p2, npx_start_frame:v1_end_frame);
        else
            final_spikes_p2 = V1Dat.binned_spikes(units_to_keep_p2, npx_start_frame:npx_end_frame);
        end
        final_areas_p2 = unit_areas_p2(units_to_keep_p2);
        final_nt_p2 = nan(sum(units_to_keep_p2), size(final_nt_p1, 2)); 

        final_spikes = [final_spikes_p1; final_spikes_p2];
        final_areas = [final_areas_p1; final_areas_p2];
        final_neurontypes = [final_nt_p1; final_nt_p2];
    else
        final_spikes = final_spikes_p1;
        final_areas = final_areas_p1;
        final_neurontypes = final_nt_p1;
    end
    
    corrected_vr_time = (RawDat.VR_times_synched - RawDat.VR_times_synched(1))*1000;
    npx_time = 0:1:size(final_spikes, 2)-1;
    
    is_lick = RawDat.VR_data(8, :) >= 1; 
    corrected_licks = process_licks(is_lick, corrected_vr_time, 100);
    
    % --- 7. Populate Struct ---
    all_data(imouse).mouseid = curr_mouse;
    all_data(imouse).final_spikes = final_spikes;
    all_data(imouse).final_areas = final_areas;
    all_data(imouse).final_neurontypes = final_neurontypes;
    all_data(imouse).npx_time = npx_time;
    all_data(imouse).corrected_vr_time = corrected_vr_time;
    all_data(imouse).corrected_licks = corrected_licks';
    
    all_data(imouse).vr_position = RawDat.VR_data(2, :);
    all_data(imouse).vr_world = RawDat.VR_data(5, :);
    all_data(imouse).vr_reward = RawDat.VR_data(6, :);
    all_data(imouse).vr_trial = RawDat.VR_data(7, :);
    
    % --- 8. Stats Calculation ---
    duration_sec = (corrected_vr_time(end)-corrected_vr_time(1))/1000;
    average_firing_rates = sum(final_spikes, 2) / duration_sec;
    
    all_data(imouse).avg_fr_all = average_firing_rates;
    all_data(imouse).average_DMS_fr = average_firing_rates(strcmp(final_areas, 'DMS'));
    all_data(imouse).average_DLS_fr = average_firing_rates(strcmp(final_areas, 'DLS'));
    all_data(imouse).average_ACC_fr = average_firing_rates(strcmp(final_areas, 'ACC'));
    all_data(imouse).average_V1_fr  = average_firing_rates(strcmp(final_areas, 'V1')); % NEW
    
    all_data(imouse).average_lick_rate = sum(corrected_licks) / duration_sec;
end

empty_indices = cellfun(@isempty, {all_data.mouseid});
all_data(empty_indices) = [];
fprintf('Processing complete. %d mice successfully loaded.\n', length(all_data));

%% ================= Visualization =================
% Extract data for plotting
average_DMS_fr_all = {all_data.average_DMS_fr};
average_DLS_fr_all = {all_data.average_DLS_fr};
average_ACC_fr_all = {all_data.average_ACC_fr};
average_V1_fr_all  = {all_data.average_V1_fr}; % NEW: Extract V1

average_lick_rate_all = [all_data.average_lick_rate];

% Median per animal
% Safe median wrapper to return NaN for animals missing V1 data (empty arrays)
safe_median = @(x) nanmedian([x(:); NaN]); 
median_DMS_fr_animals = cellfun(safe_median, average_DMS_fr_all);
median_DLS_fr_animals = cellfun(safe_median, average_DLS_fr_all);
median_ACC_fr_animals = cellfun(safe_median, average_ACC_fr_all);
median_V1_fr_animals  = cellfun(safe_median, average_V1_fr_all); % NEW

% Plot 1: Median FR per animal
figure('Name', 'Median FR per Animal');
my_errorbar_plot([median_V1_fr_animals', median_DMS_fr_animals', median_DLS_fr_animals', median_ACC_fr_animals'], true);
xticklabels({'V1', 'DMS', 'DLS', 'ACC'});
ylabel('Median Firing Rate (Hz)');
title('Population Median Firing Rates');

% Plot 2: All neurons pooled
all_DMS_frs = cat(1, average_DMS_fr_all{:});
all_DLS_frs = cat(1, average_DLS_fr_all{:});
all_ACC_frs = cat(1, average_ACC_fr_all{:});
all_V1_frs  = cat(1, average_V1_fr_all{:}); % NEW

figure('Name', 'All Neurons FR');
my_errorbar_plot({all_V1_frs, all_DMS_frs, all_DLS_frs, all_ACC_frs});
xticklabels({'V1', 'DMS', 'DLS', 'ACC'});
ylabel('Average Firing Rate (Hz)');
title('Pooled Firing Rates');

% Plot 3: Correlations
figure('Name', 'FR vs Lick Rate');
% Expanded to 1x3 subplot to include V1
subplot(1, 3, 1);
scatter(median_V1_fr_animals, average_lick_rate_all, 'filled', 'MarkerEdgeColor', 'w');
lsline;
title('V1 Activity vs Licking');
xlabel('Median FR (Hz)'); ylabel('Avg Lick Rate (Hz)');

subplot(1, 3, 2);
scatter(median_ACC_fr_animals, average_lick_rate_all, 'filled', 'MarkerEdgeColor', 'w');
lsline;
title('ACC Activity vs Licking');
xlabel('Median FR (Hz)'); ylabel('Avg Lick Rate (Hz)');

subplot(1, 3, 3);
scatter(median_DMS_fr_animals, average_lick_rate_all, 'filled', 'MarkerEdgeColor', 'w');
lsline;
title('DMS Activity vs Licking');
xlabel('Median FR (Hz)');