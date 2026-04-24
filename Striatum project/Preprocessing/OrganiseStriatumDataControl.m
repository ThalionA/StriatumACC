%% ================= Configuration & Path Setup =================
% Control IDs: [407, 513, 515, 817, 1205]
all_mouse_ids = [407, 513, 515, 817, 1205];
num_mice = numel(all_mouse_ids);

% --- Import Depth Data for Controls ---
opts = delimitedTextImportOptions("NumVariables", 9);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["MouseID", "ACCStart", "ACCEnd", "StriatumStart", "StriatumEnd", "DMSStart", "DMSEnd", "DLSStart", "DLSEnd"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Path to your control depth CSV
depth_file_path = "./RawDataControl/Neuropixels_Depth_Data_control.csv";
if isfile(depth_file_path)
    NeuropixelsDepthData = readtable(depth_file_path, opts);
else
    error('Control depth data file not found at: %s', depth_file_path);
end

% --- Preallocate Output Structure ---
all_data = struct('mouseid', cell(1, num_mice), ...
                  'final_spikes', [], 'final_areas', [], 'final_neurontypes', [], ...
                  'npx_time', [], 'corrected_vr_time', [], 'corrected_licks', [], ...
                  'vr_position', [], 'vr_world', [], 'vr_reward', [], 'vr_trial', [], ...
                  'avg_fr_all', [], 'average_DMS_fr', [], 'average_DLS_fr', [], ...
                  'average_ACC_fr', [], 'average_lick_rate', []);

%% ================= Processing Loop =================
for imouse = 1:num_mice
    fprintf('Processing Control Animal %d/%d (ID: %d)...\n', imouse, num_mice, all_mouse_ids(imouse));
    
    % --- 1. Load Raw Data ---
    raw_filename = ['./RawDataControl/' num2str(all_mouse_ids(imouse)) '_raw.mat'];
    if ~isfile(raw_filename)
        warning('Raw file not found for mouse %d. Skipping.', all_mouse_ids(imouse));
        continue;
    end
    RawDat = load(raw_filename, 'binned_spikes', 'goodcluster2', 'VR_times_synched', 'VR_data');
    num_units = size(RawDat.binned_spikes, 1);
    
    % --- 2. Load Neuron Types (Matching Task Pipeline) ---
    % Assuming the neurontype files for controls follow the same naming convention
    nt_filename = ['./RawDataControl/' num2str(all_mouse_ids(imouse)) '_neurontype2025.mat'];
    raw_neurontype = [];
    if isfile(nt_filename)
        tmp_nt = load(nt_filename, 'neurontype');
        if isfield(tmp_nt, 'neurontype')
            raw_neurontype = tmp_nt.neurontype;
            % Ensure dimensions match
            if size(raw_neurontype, 1) ~= num_units
                warning('Mismatch in unit count for neurontype file (Mouse %d). Filling with NaNs.', all_mouse_ids(imouse));
                raw_neurontype = nan(num_units, 1);
            end

            if size(raw_neurontype, 2) < 5
                raw_neurontype(raw_neurontype(:,3)>=0.4 & raw_neurontype(:,4)<=40,5) = 1; %MSN
                raw_neurontype(raw_neurontype(:,3)<0.4 & raw_neurontype(:,2)<0.1,5) = 2;  %FSN
                raw_neurontype(raw_neurontype(:,3)>=0.4 & raw_neurontype(:,4)>40,5) = 3;  %TAN
                raw_neurontype(raw_neurontype(:,3)<0.4 & raw_neurontype(:,2)>=0.1,5) = 4; %UIN (Unidentified)
            end
        else
            warning('File found but variable ''neurontype'' missing for mouse %d. Filling with NaNs.', all_mouse_ids(imouse));
            raw_neurontype = nan(num_units, 1);
        end
    else
        % File missing: Fill with NaNs silently (or warn if you prefer)
        % fprintf('  No neurontype file found. Filling with NaNs.\n');
        raw_neurontype = nan(num_units, 1);
    end

    % --- 3. Assign Areas based on Depths ---
    unit_areas = cell(num_units, 1);
    depths = NeuropixelsDepthData(NeuropixelsDepthData.MouseID == all_mouse_ids(imouse), :);
    
    if isempty(depths)
        warning('No depth data for mouse %d. Skipping.', all_mouse_ids(imouse));
        units_to_keep = false(num_units, 1);
    else
        unit_depths = RawDat.goodcluster2(:, 2);
        is_dms = unit_depths >= depths.DMSStart & unit_depths <= depths.DMSEnd;
        is_dls = unit_depths >= depths.DLSStart & unit_depths <= depths.DLSEnd;
        is_acc = unit_depths >= depths.ACCStart & unit_depths <= depths.ACCEnd;
        
        unit_areas(is_dms) = {'DMS'};
        unit_areas(is_dls) = {'DLS'};
        unit_areas(is_acc) = {'ACC'};
        units_to_keep = ~cellfun(@isempty, unit_areas);
    end

    % --- 4. Slicing and Alignment ---
    % Convert seconds to ms (1kHz bins)
    npx_start_frame = ceil(RawDat.VR_times_synched(1)*1000);
    npx_end_frame = floor(RawDat.VR_times_synched(end)*1000);
    
    % Clamp frames to matrix bounds
    npx_start_frame = max(1, npx_start_frame);
    npx_end_frame = min(size(RawDat.binned_spikes, 2), npx_end_frame);
    
    % Filter spikes, areas, and types
    final_spikes = RawDat.binned_spikes(units_to_keep, npx_start_frame:npx_end_frame);
    final_areas = unit_areas(units_to_keep);
    final_neurontypes = raw_neurontype(units_to_keep, :);

    % Sync VR time to start at 0
    corrected_vr_time = (RawDat.VR_times_synched - RawDat.VR_times_synched(1))*1000;
    npx_time = 0:1:size(final_spikes, 2)-1;

    % Process Licks (Assumes process_licks function is in path)
    is_lick = RawDat.VR_data(8, :) >= 1; 
    corrected_licks = process_licks(is_lick, corrected_vr_time, 100);

    % --- 5. Populate Struct ---
    all_data(imouse).mouseid = all_mouse_ids(imouse);
    all_data(imouse).final_spikes = final_spikes;
    all_data(imouse).final_areas = final_areas;
    all_data(imouse).final_neurontypes = final_neurontypes;
    all_data(imouse).npx_time = npx_time;
    all_data(imouse).corrected_vr_time = corrected_vr_time;
    all_data(imouse).corrected_licks = corrected_licks';
    
    % VR Data Map
    all_data(imouse).vr_position = RawDat.VR_data(2, :);
    all_data(imouse).vr_world = RawDat.VR_data(5, :);
    all_data(imouse).vr_reward = RawDat.VR_data(6, :);
    all_data(imouse).vr_trial = RawDat.VR_data(7, :);

    % --- 6. Firing Rate Stats ---
    duration_sec = (corrected_vr_time(end)-corrected_vr_time(1))/1000;
    average_firing_rates = sum(final_spikes, 2) / duration_sec;
    
    all_data(imouse).avg_fr_all = average_firing_rates;
    all_data(imouse).average_DMS_fr = average_firing_rates(strcmp(final_areas, 'DMS'));
    all_data(imouse).average_DLS_fr = average_firing_rates(strcmp(final_areas, 'DLS'));
    all_data(imouse).average_ACC_fr = average_firing_rates(strcmp(final_areas, 'ACC'));
    all_data(imouse).average_lick_rate = sum(corrected_licks) / duration_sec;
end

% Cleanup
all_data(cellfun(@isempty, {all_data.mouseid})) = [];
save('all_data_control.mat', 'all_data', '-v7.3');
fprintf('Success: %d control mice processed.\n', length(all_data));