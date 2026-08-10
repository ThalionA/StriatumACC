% OrganiseStriatumDataControl2.m
% Control2-cohort organiser. Moved out of legacy/ and fixed 2026-08-10:
%   - now SAVES all_data_control2.mat (the old version built the variable in
%     memory but never saved, so PreprocessStriatumControl2.m had nothing to load)
%   - corrected the depth-CSV read path (was an absolute "/RawDataControl2/..."
%     that never resolved)
%   - added sessions 1107_M1 and 1107_M2. Their raw files are '1107_M1_raw.mat' /
%     '1107_M2_raw.mat' and share the numeric stem 1107, so session IDs are now
%     STRINGS (matched against a string MouseID column in the depth CSV). mouseid
%     is a label only -- no downstream consumer indexes control2 by it.
%
% NOTE (control2 VR_data format): these recordings have a 10-row VR_data and the
% trial channel is row 10 (NOT row 7 as in the 8-row task format). Preserved from
% the original control2 loader.
%
% FOLLOW-UP (not done here): Zihao produced waveform-derived neuron types for
% 1103 / 1107_M1 / 1107_M2. This loader does not read neuron types yet (the old
% one never did, and summary_numbers.m consumes control2 final_neurontypes only
% behind an isfield guard). Add <id>_neurontype2025.mat loading once those files
% are synced into RawDataControl2/ if control2 neuron-type stats are wanted.

clear all;
close all;

% String session IDs -- must match the raw filename stems <id>_raw.mat AND the
% MouseID column of the depth CSV. 1107_M1/1107_M2 share numeric 1107, hence strings.
all_mouse_ids = {'316', '317', '1011', '1103', '1107_M1', '1107_M2'};
num_mice = numel(all_mouse_ids);

% --- Import depth data (MouseID as STRING to allow '1107_M1' etc.) ---
opts = delimitedTextImportOptions("NumVariables", 9);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["MouseID", "ACCStart", "ACCEnd", "StriatumStart", "StriatumEnd", "DMSStart", "DMSEnd", "DLSStart", "DLSEnd"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
depth_file_path = "./RawDataControl2/Neuropixels_Depth_Data_control2.csv";
if isfile(depth_file_path)
    NeuropixelsDepthData = readtable(depth_file_path, opts);
else
    error('Control2 depth data file not found at: %s', depth_file_path);
end

% --- Preallocate output ---
all_data = struct('mouseid', cell(1, num_mice), ...
    'final_spikes', [], 'final_areas', [], 'npx_time', [], ...
    'corrected_vr_time', [], 'corrected_licks', [], ...
    'vr_position', [], 'vr_world', [], 'vr_reward', [], 'vr_trial', [], ...
    'avg_fr_all', []);

%% ================= Processing Loop =================
for imouse = 1:num_mice
    curr_mouse = all_mouse_ids{imouse};
    fprintf('Processing control2 session %d/%d (ID: %s)...\n', imouse, num_mice, curr_mouse);

    % --- 1. Load raw data ---
    raw_filename = ['./RawDataControl2/' curr_mouse '_raw.mat'];
    if ~isfile(raw_filename)
        warning('Raw file not found for %s. Skipping.', curr_mouse);
        continue;
    end
    RawDat = load(raw_filename, 'binned_spikes', 'goodcluster2', 'VR_times_synched', 'VR_data');
    num_units = size(RawDat.binned_spikes, 1);

    % --- 2. Assign areas by depth (DMS -> DLS -> ACC precedence) ---
    depths = NeuropixelsDepthData(strcmp(NeuropixelsDepthData.MouseID, curr_mouse), :);
    if isempty(depths)
        warning('No depth data found for %s in CSV. Skipping.', curr_mouse);
        continue;
    end
    unit_depths = RawDat.goodcluster2(:, 2);
    unit_areas = assign_areas_by_depth(unit_depths, {'DMS', 'DLS', 'ACC'}, ...
        [depths.DMSStart, depths.DLSStart, depths.ACCStart], ...
        [depths.DMSEnd,   depths.DLSEnd,   depths.ACCEnd]);
    units_to_keep = ~cellfun(@isempty, unit_areas);

    % --- 3. Slice and align ---
    npx_start_frame = ceil(RawDat.VR_times_synched(1)*1000);
    npx_end_frame = floor(RawDat.VR_times_synched(end)*1000);
    npx_start_frame = max(1, npx_start_frame);
    npx_end_frame = min(size(RawDat.binned_spikes, 2), npx_end_frame);

    final_spikes = RawDat.binned_spikes(units_to_keep, npx_start_frame:npx_end_frame);
    final_areas  = unit_areas(units_to_keep);

    corrected_vr_time = (RawDat.VR_times_synched - RawDat.VR_times_synched(1))*1000;
    npx_time = 0:1:size(final_spikes, 2)-1;
    corrected_licks = process_licks(RawDat.VR_data(8, :) >= 1, corrected_vr_time, 100);

    % --- 4. Populate struct ---
    all_data(imouse).mouseid = curr_mouse;
    all_data(imouse).final_spikes = final_spikes;
    all_data(imouse).final_areas = final_areas;
    all_data(imouse).npx_time = npx_time;
    all_data(imouse).corrected_vr_time = corrected_vr_time;
    all_data(imouse).corrected_licks = corrected_licks';
    all_data(imouse).vr_position = RawDat.VR_data(2, :);
    all_data(imouse).vr_world = RawDat.VR_data(5, :);
    all_data(imouse).vr_reward = RawDat.VR_data(6, :);
    all_data(imouse).vr_trial = RawDat.VR_data(10, :);   % control2: trial is row 10

    duration_sec = (corrected_vr_time(end) - corrected_vr_time(1)) / 1000;
    all_data(imouse).avg_fr_all = sum(final_spikes, 2) / duration_sec;

    fprintf('  %s: kept %d of %d units\n', curr_mouse, sum(units_to_keep), num_units);
end

% Drop skipped sessions and save
all_data(cellfun(@isempty, {all_data.mouseid})) = [];
save('all_data_control2.mat', 'all_data', '-v7.3');
fprintf('Success: %d control2 sessions processed -> all_data_control2.mat\n', numel(all_data));
