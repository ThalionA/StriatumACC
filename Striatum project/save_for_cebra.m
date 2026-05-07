% save_for_cebra.m
% Exports per-mouse data files for CEBRA analysis with multi-dimensional
% contrastive labels. Replaces the older Legacy/save_for_cebra.m which only
% exported lick_data and per-trial lick_errors.
%
% Each output file `cebra_mouse{N}data.mat` contains:
%   neural_data  [neurons x bins x trials]      spike-count firing rates
%   lick_rate    [trials  x bins]               per-bin lick rate
%   lick_errors  [1 x trials]                   per-trial z-scored lick error
%   position     [1 x bins]                     bin centres (0-indexed bin number)
%   velocity     [trials x bins]                cm/s per bin (bin_size / duration)
%   learning_point   scalar (NaN for non-learners)
%   change_point     scalar (NaN if undefined)
%   group_id     scalar (1=task, 2=control, 3=control2)
%   area_labels  {1 x neurons}  cell of strings ('DMS', 'DLS', 'ACC', 'V1')
%   neuron_types [neurons x 1] integer (1=MSN, 2=FSN, 3=TAN, 4=UIN, NaN=unknown)
%   mouse_id     scalar
%   bin_size_cm  scalar (1.25 cm per VR a.u., bin = bin_size * 1.25)
%
% Inputs: pulls from `preprocessed_data` in the workspace, plus `cfg` for
% bin geometry. If neither is in scope, attempts to load from
% `cfg.task_data_file`. Outputs go into ./cebra_data/.
%
% Created 2026-05-07 as part of the audit consolidation.

if ~exist('cfg', 'var') || isempty(cfg)
    cfg = struct();
end
if ~isfield(cfg, 'bin_size'),       cfg.bin_size = 4;       end
if ~isfield(cfg, 'au_to_cm'),       cfg.au_to_cm = 1.25;    end
if ~isfield(cfg, 'output_dir'),     cfg.output_dir = './cebra_data'; end
if ~isfield(cfg, 'group_id'),       cfg.group_id = 1;       end
if ~isfield(cfg, 'task_data_file'), cfg.task_data_file = 'preprocessed_data.mat'; end

if ~exist('preprocessed_data', 'var') || isempty(preprocessed_data)
    if isfile(cfg.task_data_file)
        fprintf('Loading %s ...\n', cfg.task_data_file);
        S = load(cfg.task_data_file, 'preprocessed_data');
        preprocessed_data = S.preprocessed_data;
    else
        error('preprocessed_data not in workspace and %s not found.', cfg.task_data_file);
    end
end

if ~isfolder(cfg.output_dir)
    mkdir(cfg.output_dir);
end

n_animals = numel(preprocessed_data);
fprintf('Exporting %d animals to %s\n', n_animals, cfg.output_dir);

% Find learning points up-front using the shared helper.
[lps, ~] = find_learning_points(preprocessed_data, struct( ...
    'lp_z_threshold', -2, 'lp_window', 10, 'lp_min_consecutive', 7));

for ianimal = 1:n_animals
    pd = preprocessed_data(ianimal);

    n_trials = pd.n_trials;
    if n_trials < 1
        warning('Skipping animal %d (no trials).', ianimal);
        continue;
    end

    % --- Neural data ---
    neural_data = pd.spatial_binned_fr_all(:, :, 1:n_trials);     % [N x B x T]
    [n_neurons, n_bins, ~] = size(neural_data);

    % --- Behavioural labels ---
    lick_rate = pd.spatial_binned_data.licks(1:n_trials, :);      % [T x B]
    lick_rate(lick_rate > quantile(lick_rate(:), 0.99)) = NaN;    % cap top 1%
    lick_errors = pd.zscored_lick_errors(1, 1:n_trials);          % [1 x T]

    durations = pd.spatial_binned_data.durations(1:n_trials, :);  % [T x B] in same time units as the rig
    velocity = (cfg.bin_size * cfg.au_to_cm) ./ durations;        % cm/s

    position = (0:n_bins-1);                                      % bin index

    % --- Anatomy / cell type ---
    if isfield(pd, 'is_dms') && isfield(pd, 'is_dls') && isfield(pd, 'is_acc')
        area_labels = repmat({'Unknown'}, 1, n_neurons);
        if numel(pd.is_dms) == n_neurons, area_labels(pd.is_dms) = {'DMS'}; end
        if numel(pd.is_dls) == n_neurons, area_labels(pd.is_dls) = {'DLS'}; end
        if numel(pd.is_acc) == n_neurons, area_labels(pd.is_acc) = {'ACC'}; end
        if isfield(pd, 'is_v1') && numel(pd.is_v1) == n_neurons
            area_labels(pd.is_v1) = {'V1'};
        end
    else
        area_labels = repmat({'Unknown'}, 1, n_neurons);
    end

    if isfield(pd, 'final_neurontypes') && ~isempty(pd.final_neurontypes)
        nt = pd.final_neurontypes;
        if size(nt, 2) >= 5
            neuron_types = nt(:, 5);
        elseif size(nt, 2) == 1
            neuron_types = nt(:, 1);
        else
            neuron_types = nan(n_neurons, 1);
        end
    else
        neuron_types = nan(n_neurons, 1);
    end

    % --- Per-mouse LP ---
    learning_point = lps(ianimal);  % NaN for non-learners
    if isfield(pd, 'change_point_mean') && ~isempty(pd.change_point_mean)
        change_point = pd.change_point_mean;
    else
        change_point = NaN;
    end

    if isfield(pd, 'mouseid') && ~isempty(pd.mouseid)
        mouse_id = pd.mouseid;
    else
        mouse_id = ianimal;
    end

    bin_size_cm = cfg.bin_size * cfg.au_to_cm;
    group_id = cfg.group_id;

    out_file = fullfile(cfg.output_dir, sprintf('cebra_mouse%d_data.mat', ianimal));
    save(out_file, 'neural_data', 'lick_rate', 'lick_errors', 'position', ...
                   'velocity', 'learning_point', 'change_point', 'group_id', ...
                   'area_labels', 'neuron_types', 'mouse_id', 'bin_size_cm', '-v7.3');
    fprintf('  Wrote %s : N=%d, B=%d, T=%d, LP=%s\n', ...
            out_file, n_neurons, n_bins, n_trials, num2str(learning_point));
end

fprintf('\nExport complete. Run cebra_analysis.py from this directory next.\n');


% Local LP helper removed 2026-05-07; uses shared find_learning_points.m
