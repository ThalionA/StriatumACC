% OrganiseStriatumDataControlIncV1.m
% Control-cohort organiser WITH V1 multi-probe integration.
% Mirrors OrganiseStriatumDataIncV1.m (task) for the control mice.
%   Probe 1 (striatum / ACC) depths : RawDataControl/Neuropixels_Depth_Data_control.csv
%   Probe 2 (V1 / CA1 / DG / ...)   : RawDataControl/Neuropixels_V1_Depth_Data_control.csv
% Output: all_data_control.mat (V1-integrated; additive vs the striatum-only
% OrganiseStriatumDataControl.m -- DMS/DLS/ACC counts are unchanged, V1/CA1/DG
% units are appended for the mice that have a probe-2 recording).
%
% NOTE ON FILE NAMING: control probe-2 raw files are LOWERCASE
% '<id>_v1_raw.mat' (task uses uppercase '<id>_V1_raw.mat'). This is matched
% exactly below. Do NOT "align" the case to the task convention without
% renaming the files on disk -- case matters on the Linux cluster even though
% macOS/APFS is case-insensitive.

clear all;
close all;

%% ================= Configuration & Imports =================
all_mouse_ids = [407, 513, 515, 817, 1205];
num_mice = numel(all_mouse_ids);

% --- Import Depth Data (Probe 1 -- striatum / ACC) ---
opts = delimitedTextImportOptions("NumVariables", 9);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["MouseID", "ACCStart", "ACCEnd", "StriatumStart", "StriatumEnd", "DMSStart", "DMSEnd", "DLSStart", "DLSEnd"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
depth_file_path = "./RawDataControl/Neuropixels_Depth_Data_control.csv";
if isfile(depth_file_path)
    NeuropixelsDepthData = readtable(depth_file_path, opts);
else
    error('Control depth data file not found at: %s', depth_file_path);
end

% --- Import Depth Data (Probe 2 -- V1 / CA1 / DG / ...) ---
% Any number of <Area>Start / <Area>End column pairs; area names auto-detected.
v1_depth_file = "./RawDataControl/Neuropixels_V1_Depth_Data_control.csv";
if isfile(v1_depth_file)
    opts_v1 = detectImportOptions(v1_depth_file);
    ProbeBDepthData = readtable(v1_depth_file, opts_v1);
else
    ProbeBDepthData = table();
    warning('Neuropixels_V1_Depth_Data_control.csv not found. Continuing without V1/CA1/DG.');
end

% Identify probe-2 area names from column headers (e.g. V1Start -> 'V1').
probe_b_col_names = string(ProbeBDepthData.Properties.VariableNames);
probe_b_areas = strings(0);
for icol = 2:numel(probe_b_col_names)
    nm = probe_b_col_names(icol);
    if endsWith(nm, "Start")
        probe_b_areas(end + 1) = extractBefore(nm, strlength(nm) - 4); %#ok<SAGROW>
    end
end
if ~isempty(probe_b_col_names)
    mouse_col = probe_b_col_names(1);   % handles 'MouseID' / 'Mouse_ID' / 'MouseId'
else
    mouse_col = "MouseID";
end
fprintf('Probe-2 areas detected from CSV: %s\n', strjoin(probe_b_areas, ', '));

% --- Preallocate Output Structure ---
all_data = struct('mouseid', cell(1, num_mice), ...
    'final_spikes', [], 'final_areas', [], 'final_neurontypes', [], ...
    'npx_time', [], 'corrected_vr_time', [], 'corrected_licks', [], ...
    'vr_position', [], 'vr_world', [], 'vr_reward', [], 'vr_trial', [], ...
    'avg_fr_all', [], 'average_DMS_fr', [], 'average_DLS_fr', [], ...
    'average_ACC_fr', [], 'average_lick_rate', []);
% Per-probe-2 area mean-FR fields added dynamically.
for i_area = 1:numel(probe_b_areas)
    fname = ['average_' char(probe_b_areas(i_area)) '_fr'];
    [all_data.(fname)] = deal([]);
end

%% ================= Processing Loop =================
for imouse = 1:num_mice
    curr_mouse = all_mouse_ids(imouse);
    fprintf('Processing control animal %d/%d (ID: %d)...\n', imouse, num_mice, curr_mouse);

    % --- 1. Load Raw Data (Probe 1) ---
    raw_filename = ['./RawDataControl/' num2str(curr_mouse) '_raw.mat'];
    if ~isfile(raw_filename)
        warning('Raw file not found for mouse %d. Skipping.', curr_mouse);
        continue;
    end
    RawDat = load(raw_filename, 'binned_spikes', 'goodcluster2', 'VR_times_synched', 'VR_data');
    num_units_p1 = size(RawDat.binned_spikes, 1);

    % --- 2. Load Raw Data (Probe 2 -- V1 / CA1 / DG) ---
    % Control raws are lowercase '<id>_v1_raw.mat' (see header note).
    v1_filename = ['./RawDataControl/' num2str(curr_mouse) '_v1_raw.mat'];
    has_probe_b = isfile(v1_filename);
    if has_probe_b
        V1Dat = load(v1_filename, 'binned_spikes', 'goodcluster2');
    end

    % --- 3. Load Neuron Types (Probe 1) ---
    nt_filename = ['./RawDataControl/' num2str(curr_mouse) '_neurontype2025.mat'];
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

    % --- 4. Assign Areas (Probe 1: DMS -> DLS -> ACC precedence) ---
    depths = NeuropixelsDepthData(NeuropixelsDepthData.MouseID == curr_mouse, :);
    if isempty(depths)
        warning('No depth data found for mouse %d in CSV.', curr_mouse);
        unit_areas_p1 = repmat({''}, num_units_p1, 1);
        units_to_keep_p1 = false(num_units_p1, 1);
    else
        unit_depths_p1 = RawDat.goodcluster2(:, 2);
        p1_names  = {'DMS', 'DLS', 'ACC'};
        p1_starts = [depths.DMSStart, depths.DLSStart, depths.ACCStart];
        p1_ends   = [depths.DMSEnd,   depths.DLSEnd,   depths.ACCEnd];
        unit_areas_p1 = assign_areas_by_depth(unit_depths_p1, p1_names, p1_starts, p1_ends);
        units_to_keep_p1 = ~cellfun(@isempty, unit_areas_p1);
    end

    % --- 5. Assign Areas (Probe 2 -- auto-detected V1/CA1/DG/...) ---
    if has_probe_b
        unit_depths_p2 = V1Dat.goodcluster2(:, 2);
        row_mask = ProbeBDepthData.(char(mouse_col)) == curr_mouse;
        b_depths = ProbeBDepthData(row_mask, :);

        p2_names  = cellstr(probe_b_areas);
        p2_starts = nan(1, numel(probe_b_areas));
        p2_ends   = nan(1, numel(probe_b_areas));
        if ~isempty(b_depths)
            for i_area = 1:numel(probe_b_areas)
                area_nm = char(probe_b_areas(i_area));
                sc = [area_nm 'Start'];
                ec = [area_nm 'End'];
                if ismember(sc, b_depths.Properties.VariableNames) && ismember(ec, b_depths.Properties.VariableNames)
                    p2_starts(i_area) = b_depths.(sc)(1);
                    p2_ends(i_area)   = b_depths.(ec)(1);
                end
            end
        end
        unit_areas_p2 = assign_areas_by_depth(unit_depths_p2, p2_names, p2_starts, p2_ends);
        units_to_keep_p2 = ~cellfun(@isempty, unit_areas_p2);
    end

    % --- 6. Slicing and Alignment ---
    npx_start_frame = ceil(RawDat.VR_times_synched(1)*1000);
    npx_end_frame = floor(RawDat.VR_times_synched(end)*1000);
    npx_start_frame = max(1, npx_start_frame);
    npx_end_frame = min(size(RawDat.binned_spikes, 2), npx_end_frame);

    % Slice Probe 1
    final_spikes_p1 = RawDat.binned_spikes(units_to_keep_p1, npx_start_frame:npx_end_frame);
    final_areas_p1  = unit_areas_p1(units_to_keep_p1);
    final_nt_p1     = raw_neurontype(units_to_keep_p1, :);

    % Slice Probe 2 and merge
    if has_probe_b
        v1_end_frame = min(size(V1Dat.binned_spikes, 2), npx_end_frame);
        if v1_end_frame < npx_end_frame
            final_spikes_p2 = zeros(sum(units_to_keep_p2), npx_end_frame - npx_start_frame + 1);
            final_spikes_p2(:, 1:(v1_end_frame - npx_start_frame + 1)) = V1Dat.binned_spikes(units_to_keep_p2, npx_start_frame:v1_end_frame);
        else
            final_spikes_p2 = V1Dat.binned_spikes(units_to_keep_p2, npx_start_frame:npx_end_frame);
        end
        final_areas_p2 = unit_areas_p2(units_to_keep_p2);
        final_nt_p2    = nan(sum(units_to_keep_p2), size(final_nt_p1, 2));

        final_spikes = [final_spikes_p1; final_spikes_p2];
        final_areas  = [final_areas_p1; final_areas_p2];
        final_neurontypes = [final_nt_p1; final_nt_p2];
    else
        final_spikes = final_spikes_p1;
        final_areas  = final_areas_p1;
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

    % --- 8. Firing-rate stats ---
    duration_sec = (corrected_vr_time(end)-corrected_vr_time(1))/1000;
    average_firing_rates = sum(final_spikes, 2) / duration_sec;

    all_data(imouse).avg_fr_all = average_firing_rates;
    all_data(imouse).average_DMS_fr = average_firing_rates(strcmp(final_areas, 'DMS'));
    all_data(imouse).average_DLS_fr = average_firing_rates(strcmp(final_areas, 'DLS'));
    all_data(imouse).average_ACC_fr = average_firing_rates(strcmp(final_areas, 'ACC'));
    for i_area = 1:numel(probe_b_areas)
        a_nm = char(probe_b_areas(i_area));
        all_data(imouse).(['average_' a_nm '_fr']) = average_firing_rates(strcmp(final_areas, a_nm));
    end
    all_data(imouse).average_lick_rate = sum(corrected_licks) / duration_sec;
end

% Drop skipped mice (empty struct slots)
empty_indices = cellfun(@isempty, {all_data.mouseid});
all_data(empty_indices) = [];

save('all_data_control.mat', 'all_data', '-v7.3');
fprintf('Success: %d control mice processed (V1-integrated).\n', numel(all_data));
