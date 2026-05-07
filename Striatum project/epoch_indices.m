function idx = epoch_indices(lp, n_trials, cfg)
% EPOCH_INDICES Naive / Intermediate / Expert trial indices around a learning point.
%
%   idx = epoch_indices(lp, n_trials, cfg)
%
% INPUTS
%   lp       - scalar learning point (NaN for non-learners)
%   n_trials - total trials available for the animal
%   cfg      - struct with optional fields:
%                trials_per_epoch (default 10)   epoch length
%                naive_start      (default 1)    first trial of Naive
%                expert_starts_at (default 'lp') 'lp'  -> Expert = lp:lp+w-1
%                                                'lp1' -> Expert = lp+1:lp+w
%                                                          (legacy MI v2 style)
% OUTPUTS
%   idx - 1x3 cell array {naive, intermediate, expert}; entries that don't
%         fit within [1, n_trials] are returned as [].
%
% Replaces six near-identical inline epoch-slicing snippets across
% IntegratedAll_v1, MutualInformationStriatum_v2, Nonlinear_Epoch_Decoding,
% CrossSpatialBinDecoding, CCA_striatum_spatial_v2, and SpatioTemporal*.
%
% Created 2026-05-07.

    if nargin < 3 || isempty(cfg), cfg = struct(); end
    if ~isfield(cfg, 'trials_per_epoch'), cfg.trials_per_epoch = 10; end
    if ~isfield(cfg, 'naive_start'),      cfg.naive_start = 1;       end
    if ~isfield(cfg, 'expert_starts_at'), cfg.expert_starts_at = 'lp'; end

    w = cfg.trials_per_epoch;
    idx = {[], [], []};

    % Naive: first w trials of the session
    if n_trials >= w
        idx{1} = cfg.naive_start : (cfg.naive_start + w - 1);
    end

    if isnan(lp), return; end

    % Intermediate: w trials immediately before LP
    pre_start = lp - w;
    pre_end   = lp - 1;
    if pre_start >= 1 && pre_end <= n_trials && pre_end >= pre_start
        idx{2} = pre_start : pre_end;
    end

    % Expert: w trials starting at LP (or at LP+1 in the MI-style convention)
    switch cfg.expert_starts_at
        case 'lp'
            post_start = lp;
        case 'lp1'
            post_start = lp + 1;
        otherwise
            error('Unknown expert_starts_at: %s', cfg.expert_starts_at);
    end
    post_end = post_start + w - 1;
    if post_start >= 1 && post_end <= n_trials
        idx{3} = post_start : post_end;
    end
end
