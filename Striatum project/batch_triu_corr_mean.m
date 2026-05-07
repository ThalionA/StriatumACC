function r = batch_triu_corr_mean(mat3d)
% BATCH_TRIU_CORR_MEAN Vectorised per-cell mean of upper-triangle Pearson r.
%
%   r = batch_triu_corr_mean(mat3d)
%
% INPUT
%   mat3d  [n_cells x bins x w] tensor. For each cell, treats the [bins x w]
%          slice as `w` trial profiles of length `bins` and computes the
%          mean of all pairwise (off-diagonal) Pearson correlations across
%          the w columns.
%
% OUTPUT
%   r      [n_cells x 1]
%
% Drop-in vectorised replacement for the per-cell `calc_triu_corr` loop in
% IntegratedAll_v1's stability section. ~50-100x speedup on n_cells=O(1000).
%
% Created 2026-05-07.

    [n_cells, bins, w] = size(mat3d);
    if w < 2 || bins < 2 || n_cells == 0
        r = nan(n_cells, 1);
        return;
    end

    % Z-score across bins (dim 2) per cell per trial-column.
    mu_S = mean(mat3d, 2, 'omitnan');
    sd_S = std(mat3d, 0, 2, 'omitnan');
    sd_S(sd_S == 0) = 1;
    Z = (mat3d - mu_S) ./ sd_S;
    Z(isnan(Z)) = 0;        % treat fully-NaN cells as zero (yields r=NaN below)

    % Per-cell Z' * Z / (bins-1) via paged matrix multiply.
    A = permute(Z, [3, 2, 1]);              % [w, bins, n_cells]
    B = permute(Z, [2, 3, 1]);              % [bins, w, n_cells]
    R = pagemtimes(A, B) / (bins - 1);      % [w, w, n_cells]

    mask = triu(true(w), 1);                % off-diagonal upper triangle
    n_pairs = sum(mask(:));
    R_flat  = reshape(R, w * w, n_cells);   % [w*w, n_cells]
    valid   = R_flat(mask(:), :);           % [n_pairs, n_cells]
    r = mean(valid, 1, 'omitnan')';         % [n_cells, 1]

    % Cells that had zero variance everywhere will give NaN — match the
    % per-cell calc_triu_corr behaviour.
    bad = all(isnan(valid), 1);
    r(bad) = NaN;
end
