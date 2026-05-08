function [r_test, r_train, A, B] = held_out_canoncorr(X, Y, num_ccs, seed)
% HELD_OUT_CANONCORR Cross-validated canonical correlations.
%
%   [r_test, r_train] = held_out_canoncorr(X, Y, num_ccs)
%   [r_test, r_train, A, B] = held_out_canoncorr(X, Y, num_ccs, seed)
%
% Splits rows (samples) into two halves, fits canoncorr on the first half
% to learn the canonical projections (A, B), then projects the held-out
% half through (A, B) and computes the canonical correlations on the
% projected test data. The result is unbiased by construction — in
% contrast to in-sample canoncorr, which is upward-biased because the
% projections are optimised against the data they're scored on.
%
% INPUTS
%   X        n_samples x p1
%   Y        n_samples x p2
%   num_ccs  number of canonical correlations to return (default min(p1,p2))
%   seed     RNG seed for the train/test split (default 0)
%
% OUTPUTS
%   r_test   1 x num_ccs  held-out canonical correlations
%   r_train  1 x num_ccs  in-sample (training) canonical correlations
%                          for direct comparison
%   A        p1 x num_ccs train-fold canonical projections for X
%   B        p2 x num_ccs train-fold canonical projections for Y
%
% Created 2026-05-07 for CCA_striatum_spatial_v2.

    if nargin < 3 || isempty(num_ccs), num_ccs = min(size(X, 2), size(Y, 2)); end
    if nargin < 4 || isempty(seed),    seed = 0; end

    n = size(X, 1);
    if n < 2 * (max(size(X, 2), size(Y, 2)) + 2)
        % Not enough samples for a meaningful split — return NaNs.
        r_test  = nan(1, num_ccs);
        r_train = nan(1, num_ccs);
        A = nan(size(X, 2), num_ccs);
        B = nan(size(Y, 2), num_ccs);
        return;
    end

    % --- Random 50/50 split ---
    rng_state = rng();        % preserve caller RNG
    rng(seed, 'twister');
    perm = randperm(n);
    rng(rng_state);

    half = floor(n / 2);
    train_idx = perm(1:half);
    test_idx  = perm(half + 1:end);

    X_train = X(train_idx, :);  Y_train = Y(train_idx, :);
    X_test  = X(test_idx, :);   Y_test  = Y(test_idx, :);

    % --- Fit canoncorr on train fold ---
    try
        [A, B, r_train_full] = canoncorr(X_train, Y_train);
    catch ME
        warning('held_out_canoncorr:CanoncorrFailed', '%s', ME.message);
        r_test  = nan(1, num_ccs);
        r_train = nan(1, num_ccs);
        A = nan(size(X, 2), num_ccs);
        B = nan(size(Y, 2), num_ccs);
        return;
    end

    % --- Project held-out half through train projections, correlate ---
    d = min(size(A, 2), size(B, 2));
    n_keep = min(d, num_ccs);

    A = A(:, 1:n_keep);
    B = B(:, 1:n_keep);

    U_test = X_test * A;
    V_test = Y_test * B;

    r_test  = nan(1, num_ccs);
    r_train = nan(1, num_ccs);
    for k = 1:n_keep
        if std(U_test(:, k)) > 0 && std(V_test(:, k)) > 0
            R = corrcoef(U_test(:, k), V_test(:, k));
            r_test(k) = R(1, 2);
        end
        r_train(k) = r_train_full(k);
    end
end
