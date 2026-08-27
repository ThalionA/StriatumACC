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
%                naive_split      (default [])   split Naive after this many
%                                                trials, giving FOUR windows
% OUTPUTS
%   idx - 1x3 cell {naive, intermediate, expert}, or 1x4
%         {naive_early, naive_late, intermediate, expert} when naive_split is
%         set; entries that don't fit within [1, n_trials] are returned as [].
%
% The naive_split option exists because the neural analyses resolve the first
% few trials separately (SpatioTemporalActivityEvolution.m: epoch_trials =
% {1:3, 4:10, 11:20, 21:30}); setting naive_split = 3 gives the behavioural
% analyses the matching windows. Note the resulting windows are unequal
% (3, 7, w, w trials), so the first two carry more sampling noise.
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
    if ~isfield(cfg, 'naive_split'),      cfg.naive_split = [];      end

    w = cfg.trials_per_epoch;
    split = cfg.naive_split;
    if ~isempty(split)
        assert(split >= 1 && split < w, 'epoch_indices:badNaiveSplit', ...
            'naive_split must be in [1, trials_per_epoch-1]; got %g', split);
    end

    % Naive occupies the first w trials, optionally cut in two.
    n0 = cfg.naive_start;
    if isempty(split)
        idx = {[], [], []};
        if n_trials >= w
            idx{1} = n0 : (n0 + w - 1);
        end
        k = 1;                       % offset of the LP-relative windows
    else
        idx = {[], [], [], []};
        if n_trials >= split
            idx{1} = n0 : (n0 + split - 1);
        end
        if n_trials >= w
            idx{2} = (n0 + split) : (n0 + w - 1);
        end
        k = 2;
    end

    if isnan(lp), return; end

    % A learning point beyond the available trials means the LP is not in the
    % data at all (several callers pass a disengagement-truncated n_trials),
    % so no LP-relative window is defined. This guard matches the inline
    % blocks this helper replaces; without it the Intermediate window would be
    % returned for truncated sessions where the inline code returned []
    % (caught by test_epoch_indices' equivalence check, 2026-08-27).
    if lp > n_trials, return; end

    % Intermediate: w trials immediately before LP
    pre_start = lp - w;
    pre_end   = lp - 1;
    if pre_start >= 1 && pre_end <= n_trials && pre_end >= pre_start
        idx{k + 1} = pre_start : pre_end;
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
        idx{k + 2} = post_start : post_end;
    end
end
