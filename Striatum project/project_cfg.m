function cfg = project_cfg()
% PROJECT_CFG Single source of truth for project-wide constants.
%
%   cfg = project_cfg()
%
% Returns a struct with every magic number that used to be scattered
% across analysis scripts. Each entry-point script should start with:
%
%       cfg = project_cfg();
%       % then override any field for the local analysis, e.g.
%       cfg.ridge_lambda = 0.5;
%
% Created 2026-05-07.

    cfg = struct();

    % --- Spatial geometry ---
    cfg.bin_size_au   = 4;          % bins are 4 a.u. wide
    cfg.au_to_cm      = 1.25;       % 1 VR a.u. = 1.25 cm
    cfg.bin_size_cm   = cfg.bin_size_au * cfg.au_to_cm;  % 5 cm/bin
    cfg.corridor_au   = 200;        % full corridor length (a.u.)
    cfg.n_bins_full   = 50;         % corridor / bin_size_au
    cfg.max_bin       = 30;         % spatial truncation (RZ + ~5 bins)

    % Landmark bins (spatial bin index)
    
    cfg.visual_bin    = 20;
    cfg.reward_bin    = 25;
    cfg.target_rz_bin = 25;         % alias for callers that want it

    cfg.corridor_cm   = 250;        % full corridor length (cm)
    cfg.visual_zone_cm    = 80;
    cfg.reward_zone_cm    = 100;

    % --- Areas ---
    % CA1 and DG added 2026-05-08. To add a new area in future:
    %   1) Add a column to RawData/Neuropixels_V1_Depth_Data.csv
    %   2) Append the name to cfg.areas below
    %   3) Append the matching `is_<lower>` field to area_field_map
    %   4) Append a colour to cfg.area_colors
    %   5) (optional) Add cross-area pairs to cfg.area_pairs
    % All downstream code reads from these fields.
    cfg.areas         = {'DMS', 'DLS', 'ACC', 'V1', 'CA1', 'DG'};
    cfg.area_field_map = containers.Map( ...
        cfg.areas, ...
        {'is_dms', 'is_dls', 'is_acc', 'is_v1', 'is_ca1', 'is_dg'});
    cfg.area_colors = [0      0.4470 0.7410;   % DMS (blue)
                       0.4660 0.6740 0.1880;   % DLS (green)
                       0.8500 0.3250 0.0980;   % ACC (orange)
                       0.4940 0.1840 0.5560;   % V1  (purple)
                       0.8000 0.1000 0.2000;   % CA1 (crimson)
                       0.2000 0.7000 0.7000];  % DG  (teal)

    % All pairs across the 6 areas (n*(n-1)/2 = 15) — for CCA / MI / lag.
    cfg.area_pairs = {...
        'DMS', 'DLS'; 'DMS', 'ACC'; 'DLS', 'ACC'; ...
        'V1',  'DMS'; 'V1',  'DLS'; 'V1',  'ACC'; ...
        'CA1', 'DMS'; 'CA1', 'DLS'; 'CA1', 'ACC'; 'CA1', 'V1'; ...
        'DG',  'DMS'; 'DG',  'DLS'; 'DG',  'ACC'; 'DG',  'V1'; 'DG', 'CA1'};

    % --- Learning point definition (START-of-window convention) ---
    cfg.lp_z_threshold      = -2;
    cfg.lp_window           = 10;
    cfg.lp_min_consecutive  = 7;
    cfg.expert_starts_at    = 'lp';  % 'lp' or 'lp1' (MI v2 legacy)

    % --- Epoch definition ---
    cfg.trials_per_epoch    = 10;
    cfg.epoch_names         = {'Naive', 'Intermediate', 'Expert'};
    cfg.epoch_colors        = [0.298 0.447 0.690;
                               0.867 0.518 0.322;
                               0.333 0.776 0.333];

    % --- Behavioural derived signals ---
    % velocity (cm/s) = (bin_size_au * au_to_cm) / duration(s)
    %                 = bin_size_cm / duration(s)
    cfg.velocity_factor = cfg.bin_size_cm;

    % --- Inclusion thresholds ---
    cfg.fr_threshold_hz   = 0.02;   % unit-level firing-rate floor
    cfg.min_units         = 5;      % per-area minimum for plotting
    cfg.min_units_per_mouse = 3;    % per-mouse minimum for hierarchical

    % --- Decoder hyperparameters ---
    cfg.ridge_lambda      = 1.0;
    cfg.n_shuffles        = 25;     % bumped from 1 on 2026-05-07
    cfg.behav_targets     = {'lick_rate', 'velocity'};

    % --- Cell-type column index in final_neurontypes ---
    cfg.ntype_col         = 5;

    % --- Reproducibility ---
    cfg.seed              = 42;

    % --- Data paths (override per-script if your CWD differs) ---
    cfg.task_data_file     = 'processed_data/preprocessed_data.mat';
    cfg.control_data_file  = 'processed_data/preprocessed_data_control.mat';
    cfg.control2_data_file = 'processed_data/preprocessed_data_control2.mat';
end
