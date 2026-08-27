function test_epoch_indices()
%TEST_EPOCH_INDICES Tests for the shared epoch-slicing helper.
%
% Includes an explicit equivalence check against the inline formula that was
% duplicated at six sites in IntegratedAll_v1.m, since that equivalence is
% what licenses replacing them with calls to this helper.
%
%   /Applications/MATLAB_R2026a.app/bin/matlab -batch "test_epoch_indices"

fprintf('test_epoch_indices\n');
w = 10;

%% --- default: three windows, Expert starting AT the LP --------------------
idx = epoch_indices(41, 300, struct('trials_per_epoch', w));
assert(isequal(idx{1}, 1:10),  'naive');
assert(isequal(idx{2}, 31:40), 'intermediate');
assert(isequal(idx{3}, 41:50), 'expert starts at lp');
assert(numel(idx) == 3, 'default must return 3 windows');
fprintf('  three-window default ............ ok\n');

%% --- legacy MI convention: Expert starts at LP+1 -------------------------
idx = epoch_indices(41, 300, struct('trials_per_epoch', w, 'expert_starts_at', 'lp1'));
assert(isequal(idx{3}, 42:51), 'lp1 convention');
fprintf('  expert_starts_at = lp1 .......... ok\n');

%% --- non-learner: Naive only --------------------------------------------
idx = epoch_indices(NaN, 300, struct('trials_per_epoch', w));
assert(isequal(idx{1}, 1:10) && isempty(idx{2}) && isempty(idx{3}), 'NaN lp');
fprintf('  NaN learning point .............. ok\n');

%% --- windows that do not fit come back empty -----------------------------
idx = epoch_indices(41, 45, struct('trials_per_epoch', w));   % expert needs 50
assert(isempty(idx{3}), 'expert must not overrun n_trials');
idx = epoch_indices(5, 300, struct('trials_per_epoch', w));   % lp too early
assert(isempty(idx{2}), 'intermediate must not start before trial 1');
fprintf('  out-of-range windows are empty .. ok\n');

%% --- EQUIVALENCE with the six inline blocks ------------------------------
% The inline formula, verbatim from IntegratedAll_v1.m (pre-migration):
%   if n_tr >= w,                                  e{1} = 1:w;
%   if ~isnan(lp) && lp > w && lp <= n_tr,         e{2} = (lp-w):(lp-1);
%   if ~isnan(lp) && (lp+w-1) <= n_tr,             e{3} = lp:(lp+w-1);
n_checked = 0;
for n_tr = [9 10 11 25 50 100 300]
    for lp = [NaN 1 5 10 11 14 22 41 84 291 300]
        e = {[], [], []};
        if n_tr >= w, e{1} = 1:w; end
        if ~isnan(lp) && lp > w && lp <= n_tr, e{2} = (lp - w):(lp - 1); end
        if ~isnan(lp) && (lp + w - 1) <= n_tr, e{3} = lp:(lp + w - 1); end
        got = epoch_indices(lp, n_tr, struct('trials_per_epoch', w));
        for k = 1:3
            assert(isequal(got{k}(:)', e{k}(:)'), ...
                'mismatch at lp=%g n_tr=%d window %d: [%s] vs inline [%s]', ...
                lp, n_tr, k, num2str(got{k}), num2str(e{k}));
        end
        n_checked = n_checked + 1;
    end
end
fprintf('  equivalence with inline formula .. ok (%d combinations)\n', n_checked);

%% --- NEW: naive_split gives four windows ---------------------------------
idx = epoch_indices(41, 300, struct('trials_per_epoch', w, 'naive_split', 3));
assert(numel(idx) == 4, 'naive_split must return 4 windows');
assert(isequal(idx{1}, 1:3),   'trials 1-3');
assert(isequal(idx{2}, 4:10),  'trials 4-10');
assert(isequal(idx{3}, 31:40), 'intermediate unchanged');
assert(isequal(idx{4}, 41:50), 'expert unchanged');
fprintf('  naive_split = 3 -> four windows .. ok\n');

% The split must partition the original Naive window exactly.
base = epoch_indices(41, 300, struct('trials_per_epoch', w));
split = epoch_indices(41, 300, struct('trials_per_epoch', w, 'naive_split', 3));
assert(isequal([split{1}, split{2}], base{1}), 'split must partition Naive');
assert(isequal(split{3}, base{2}) && isequal(split{4}, base{3}), ...
       'LP-relative windows must be untouched by the split');
fprintf('  split partitions Naive exactly ... ok\n');

% A split wider than the epoch, or <=0, is a usage error.
threw = false;
try, epoch_indices(41, 300, struct('trials_per_epoch', w, 'naive_split', 10)); ...
catch, threw = true; end
assert(threw, 'naive_split >= trials_per_epoch must error');
fprintf('  invalid naive_split rejected .... ok\n');

fprintf('ALL TESTS PASSED\n');
end
