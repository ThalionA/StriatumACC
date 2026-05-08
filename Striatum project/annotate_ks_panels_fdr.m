function p_adj = annotate_ks_panels_fdr(ax, p_raw, ks_stat, ks_did, label, q)
% ANNOTATE_KS_PANELS_FDR Apply BH-FDR across a row of KS-test panels and
% draw text annotations on each axis using the corrected p-values.
%
%   p_adj = annotate_ks_panels_fdr(ax, p_raw, ks_stat, ks_did, label, q)
%
% INPUTS
%   ax        gobject handles, one per panel
%   p_raw     raw p-values (NaN if a panel didn't run a test)
%   ks_stat   KS statistics (D), same shape as p_raw
%   ks_did    logical vector, true for panels that ran a test
%   label     short string for the test, e.g. 'Naive, Exp' or 'Tr1, Tr21'
%   q         FDR threshold (default 0.05)
%
% OUTPUT
%   p_adj     BH-adjusted p-values (NaN where ks_did was false)
%
% Created 2026-05-07 to centralise the KS+FDR annotation pattern across
% the four sites in SpatioTemporalActivityEvolution.m.

    if nargin < 6 || isempty(q), q = 0.05; end

    n = numel(ax);
    p_adj = nan(1, n);
    if any(ks_did)
        p_adj(ks_did) = fdr_correct(p_raw(ks_did), q, 'bh');
    end
    for k = 1:n
        if ~ks_did(k), continue; end
        sig_star = '';
        if p_adj(k) < 0.05,  sig_star = '*';   end
        if p_adj(k) < 0.01,  sig_star = '**';  end
        if p_adj(k) < 0.001, sig_star = '***'; end
        ks_txt = sprintf('KS(%s):\nD=%.2f, p=%.1e\np_{FDR}=%.1e %s', ...
            label, ks_stat(k), p_raw(k), p_adj(k), sig_star);
        text(ax(k), 0.05, 0.95, ks_txt, 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'FontSize', 9, 'FontWeight', 'bold', ...
            'BackgroundColor', [1 1 1 0.7]);
    end
end
