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

% --- Import Depth Data (Probe 2 — V1, CA1, DG, etc.) ---
% The CSV may contain any number of <Area>Start / <Area>End column pairs.
% Area names are extracted from headers automatically, so adding a new
% area downstream just requires adding two columns to the CSV.
% (Generalised 2026-05-08; was V1-only.)
v1_depth_file = "./RawData/Neuropixels_V1_Depth_Data.csv";
if isfile(v1_depth_file)
    opts_v1 = detectImportOptions(v1_depth_file);
    ProbeBDepthData = readtable(v1_depth_file, opts_v1);
else
    ProbeBDepthData = table();
    warning('Neuropixels_V1_Depth_Data.csv not found. Continuing without V1/CA1/DG.');
end

% Identify probe-2 area names from column headers.
probe_b_col_names = string(ProbeBDepthData.Properties.VariableNames);
probe_b_areas = strings(0);
for icol = 2:numel(probe_b_col_names)
    nm = probe_b_col_names(icol);
    if endsWith(nm, "Start")
        probe_b_areas(end + 1) = extractBefore(nm, strlength(nm) - 4); %#ok<SAGROW>
    end
end
% Find which column is mouse ID (handles 'MouseID' / 'Mouse_ID' / 'MouseId' etc.)
mouse_col = probe_b_col_names(1);
fprintf('Probe-2 areas detected from CSV: %s\n', strjoin(probe_b_areas, ', '));

% --- Preallocate Output Structure ---
all_data = struct('mouseid', cell(1, num_mice), ...
    'final_spikes', [], 'final_areas', [], 'final_neurontypes', [], ...
    'npx_time', [], 'corrected_vr_time', [], 'corrected_licks', [], ...
    'vr_position', [], 'vr_world', [], 'vr_reward', [], 'vr_trial', [], ...
    'avg_fr_all', [], 'average_DMS_fr', [], 'average_DLS_fr', [], ...
    'average_ACC_fr', [], 'average_lick_rate', []);
% Per-probe-2 area mean FR fields are added dynamically below.
for i_area = 1:numel(probe_b_areas)
    fname = ['average_' char(probe_b_areas(i_area)) '_fr'];
    [all_data.(fname)] = deal([]);
end

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

    % --- 2. Load Raw Data (Probe 2 — V1 / CA1 / DG / ...) ---
    v1_filename = ['./RawData/' num2str(curr_mouse) '_V1_raw.mat'];
    has_probe_b = isfile(v1_filename);
    has_v1 = has_probe_b;   % legacy alias (keep so older code paths work)
    if has_probe_b
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

    % --- 5. Assign Areas (Probe 2 — V1 / CA1 / DG / etc.) ---
    % Iterates over whichever <Area>Start/<Area>End columns the CSV provides.
    if has_probe_b
        unit_depths_p2 = V1Dat.goodcluster2(:, 2);
        unit_areas_p2  = cell(size(unit_depths_p2, 1), 1);

        b_depths = table();
        if ~isempty(ProbeBDepthData)
            row_mask = ProbeBDepthData.(char(mouse_col)) == curr_mouse;
            b_depths = ProbeBDepthData(row_mask, :);
        end

        for i_area = 1:numel(probe_b_areas)
            area_nm  = char(probe_b_areas(i_area));
            start_col = [area_nm 'Start'];
            end_col   = [area_nm 'End'];

            if isempty(b_depths) || ~ismember(start_col, b_depths.Properties.VariableNames) ...
                    || ~ismember(end_col,   b_depths.Properties.VariableNames)
                continue;
            end

            sv = b_depths.(start_col)(1);
            ev = b_depths.(end_col)(1);
            if isnan(sv) || isnan(ev)
                continue;
            end

            in_area = unit_depths_p2 >= sv & unit_depths_p2 <= ev;
            unit_areas_p2(in_area) = {area_nm};
        end

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
    if has_probe_b
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
    % Per-probe-2 area (V1, CA1, DG, ...) — driven by what the CSV declared.
    for i_area = 1:numel(probe_b_areas)
        a_nm = char(probe_b_areas(i_area));
        all_data(imouse).(['average_' a_nm '_fr']) = ...
            average_firing_rates(strcmp(final_areas, a_nm));
    end

    all_data(imouse).average_lick_rate = sum(corrected_licks) / duration_sec;
end

empty_indices = cellfun(@isempty, {all_data.mouseid});
all_data(empty_indices) = [];
fprintf('Processing complete. %d mice successfully loaded.\n', length(all_data));

%% ================= Visualization =================
% Generalised 2026-05-08 to plot every probe-1 + probe-2 area present.
plot_area_names = ['DMS', 'DLS', 'ACC', cellstr(probe_b_areas)];
plot_area_names = plot_area_names(:)';

avg_fr_by_area = cell(1, numel(plot_area_names));
for i_area = 1:numel(plot_area_names)
    fname = ['average_' plot_area_names{i_area} '_fr'];
    if isfield(all_data, fname)
        avg_fr_by_area{i_area} = {all_data.(fname)};
    else
        avg_fr_by_area{i_area} = repmat({[]}, 1, numel(all_data));
    end
end

average_lick_rate_all = [all_data.average_lick_rate];

% Median per animal — NaN for animals missing that area's probe
safe_median = @(x) nanmedian([x(:); NaN]);
median_fr_per_animal = nan(numel(all_data), numel(plot_area_names));
for i_area = 1:numel(plot_area_names)
    median_fr_per_animal(:, i_area) = cellfun(safe_median, avg_fr_by_area{i_area});
end

% Plot 1: Median FR per animal
figure('Name', 'Median FR per Animal');
my_errorbar_plot(median_fr_per_animal, true);
xticklabels(plot_area_names);
ylabel('Median Firing Rate (Hz)');
title('Population Median Firing Rates');

% Plot 2: All neurons pooled
pooled_frs = cell(1, numel(plot_area_names));
for i_area = 1:numel(plot_area_names)
    pooled_frs{i_area} = cat(1, avg_fr_by_area{i_area}{:});
end
figure('Name', 'All Neurons FR');
my_errorbar_plot(pooled_frs);
xticklabels(plot_area_names);
ylabel('Average Firing Rate (Hz)');
title('Pooled Firing Rates');

% Plot 3: Correlations of per-animal median FR with lick rate, one subplot per area
figure('Name', 'FR vs Lick Rate');
n_areas = numel(plot_area_names);
n_cols = min(4, n_areas); n_rows = ceil(n_areas / n_cols);
for i_area = 1:n_areas
    subplot(n_rows, n_cols, i_area);
    scatter(median_fr_per_animal(:, i_area), average_lick_rate_all, ...
        'filled', 'MarkerEdgeColor', 'w');
    lsline;
    title(sprintf('%s Activity vs Licking', plot_area_names{i_area}));
    xlabel('Median FR (Hz)'); ylabel('Avg Lick Rate (Hz)');
end