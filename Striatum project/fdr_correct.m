function [p_adj, h, fdr_thresh] = fdr_correct(p, q, method)
% FDR_CORRECT Multiple-comparisons correction.
%
%   p_adj = fdr_correct(p, q)
%   [p_adj, h, fdr_thresh] = fdr_correct(p, q, method)
%
% INPUTS
%   p      vector or matrix of raw p-values; NaNs are propagated
%   q      false-discovery-rate threshold (default 0.05)
%   method 'bh'    Benjamini-Hochberg, assumes independent or PRDS tests (default)
%          'by'    Benjamini-Yekutieli, no dependence assumption (more conservative)
%          'holm'  Holm-Bonferroni (controls FWER, not FDR)
%
% OUTPUTS
%   p_adj      array same shape as p, containing adjusted p-values
%   h          logical array; true where p_adj <= q (significant)
%   fdr_thresh largest raw p-value that survives the BH cutoff
%
% Created 2026-05-07. Used by SpatioTemporalActivityEvolution.m for KS-test
% grids; recommended for any panel-level multiple-test situation.

    if nargin < 2 || isempty(q),      q = 0.05;     end
    if nargin < 3 || isempty(method), method = 'bh'; end

    sz_in = size(p);
    pv = p(:);
    valid = ~isnan(pv);
    pv_v  = pv(valid);
    n     = numel(pv_v);

    if n == 0
        p_adj = nan(sz_in);
        h = false(sz_in);
        fdr_thresh = NaN;
        return;
    end

    [pv_sorted, idx_sort] = sort(pv_v, 'ascend');
    rank = (1:n)';

    switch lower(method)
        case 'bh'
            adj_sorted = pv_sorted .* n ./ rank;
        case 'by'
            cm = sum(1 ./ rank);  % harmonic correction
            adj_sorted = pv_sorted .* n .* cm ./ rank;
        case 'holm'
            adj_sorted = pv_sorted .* (n - rank + 1);
        otherwise
            error('fdr_correct: unknown method "%s"', method);
    end

    % Enforce monotonicity (BH/BY can have non-monotone p_adj)
    adj_sorted = min(cummax(adj_sorted, 'reverse'), 1);

    % Unsort
    adj_unsorted = nan(n, 1);
    adj_unsorted(idx_sort) = adj_sorted;

    % Re-embed into the original shape with NaN preservation
    p_adj_v = nan(numel(pv), 1);
    p_adj_v(valid) = adj_unsorted;
    p_adj = reshape(p_adj_v, sz_in);

    h = (p_adj <= q);

    % Largest raw p that survives the BH cutoff
    surviving = pv_sorted(adj_sorted <= q);
    if isempty(surviving)
        fdr_thresh = 0;
    else
        fdr_thresh = max(surviving);
    end
end
