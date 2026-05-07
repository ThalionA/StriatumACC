function [lps, avg_lp] = find_learning_points(data_struct, cfg)
% FIND_LEARNING_POINTS Per-animal learning point from z-scored lick errors.
%
%   [lps, avg_lp] = find_learning_points(data_struct, cfg)
%
% INPUTS
%   data_struct - struct array with field `zscored_lick_errors` (1 x n_trials).
%   cfg         - struct with fields:
%                   lp_z_threshold     (default -2)   z-score threshold
%                   lp_window          (default 10)   sliding window length
%                   lp_min_consecutive (default 7)    pass count required
%                 Any subset is fine; defaults fill in the rest.
%
% OUTPUTS
%   lps    - 1 x n_animals double, NaN for non-learners
%   avg_lp - scalar mean LP over learners (round); NaN if none
%
% Replaces the inlined LP-finding loop that lived in IntegratedAll_v1,
% StriatumTaskControl_IntegratedAnalysis, CCA_striatum_spatial_v2,
% Nonlinear_Epoch_Decoding, MutualInformationStriatum_v2, and the legacy
% learning_points_task script. Call this once at the top of an analysis.
%
% Created 2026-05-07.

    if nargin < 2 || isempty(cfg), cfg = struct(); end
    if ~isfield(cfg, 'lp_z_threshold'),     cfg.lp_z_threshold = -2;    end
    if ~isfield(cfg, 'lp_window'),          cfg.lp_window = 10;         end
    if ~isfield(cfg, 'lp_min_consecutive'), cfg.lp_min_consecutive = 7; end

    n_animals = numel(data_struct);
    lps = nan(1, n_animals);

    for i = 1:n_animals
        zerr = data_struct(i).zscored_lick_errors(:)';
        n_trials = numel(zerr);
        if n_trials < cfg.lp_window
            continue;
        end
        % Sliding count of trials passing threshold within each window
        passes = double(zerr <= cfg.lp_z_threshold);
        win_counts = movsum(passes, [0, cfg.lp_window - 1]);
        idx = find(win_counts >= cfg.lp_min_consecutive, 1, 'first');
        if ~isempty(idx)
            % LP = start of the first qualifying window. This matches
            % IntegratedAll_v1, processTaskData and the legacy
            % learning_points_task script. Callers wanting the "end of
            % qualifying window" convention (e.g. MutualInformationStriatum_v2)
            % can shift with lp_end = lp + cfg.lp_window - 1.
            lps(i) = idx;
        end
    end

    avg_lp = round(mean(lps, 'omitnan'));
end
