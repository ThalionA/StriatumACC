function test_classify_neuron_types()
%TEST_CLASSIFY_NEURON_TYPES Unit tests for area-aware cell-type assignment.
%
% Run headless:
%   /Applications/MATLAB_R2026a.app/bin/matlab -batch "test_classify_neuron_types"

fprintf('test_classify_neuron_types\n');

% Feature columns, matching the *_neurontype2025.mat layout:
%   1 = mean firing rate (Hz)
%   2 = proportion of long ISIs
%   3 = peak-to-trough time (ms)   <- the waveform-width axis
%   4 = post-spike ISI metric
mk = @(fr, prop, pt, isi) [fr, prop, pt, isi];

%% --- Striatal criteria are unchanged (legacy behaviour) ------------------
feats = [mk(2, 0.05, 0.60, 10);    % broad, low isi  -> MSN
         mk(8, 0.05, 0.25, 10);    % narrow, low prop-> FSN
         mk(4, 0.05, 0.60, 60);    % broad, high isi -> TAN
         mk(4, 0.20, 0.25, 10)];   % narrow, high prop-> UIN
areas = {'DMS'; 'DLS'; 'DMS'; 'DLS'};
codes = classify_neuron_types(feats, areas);
assert(isequal(codes(:)', [1 2 3 4]), 'striatal codes changed: %s', mat2str(codes(:)'));
fprintf('  striatal MSN/FSN/TAN/UIN ....... ok\n');

%% --- Cortex & hippocampus: binary FS / RS --------------------------------
% Everything that is not fast-spiking must come back regular-spiking, in
% every non-striatal area — no UIN, no NaN gaps.
nonstriatal = {'ACC', 'V1', 'CA1', 'DG'};
for k = 1:numel(nonstriatal)
    a = nonstriatal(k);
    assert(classify_neuron_types(mk(8, 0.05, 0.25, 10), a) == 2, ...
        '%s narrow waveform should be FS', a{1});
    assert(classify_neuron_types(mk(2, 0.05, 0.60, 10), a) == 5, ...
        '%s broad waveform should be RS', a{1});
    % Feature combinations that would be TAN or UIN in striatum are still
    % just RS / FS outside it — width is the only axis that applies.
    assert(classify_neuron_types(mk(4, 0.05, 0.60, 60), a) == 5, ...
        '%s TAN-like features should be RS', a{1});
    assert(classify_neuron_types(mk(4, 0.20, 0.25, 10), a) == 2, ...
        '%s UIN-like features should be FS', a{1});
end
fprintf('  non-striatal FS/RS in ACC/V1/CA1/DG ... ok\n');

%% --- The threshold is exactly 0.4 ms, FS strictly below ------------------
assert(classify_neuron_types(mk(5, 0.05, 0.3999, 10), {'V1'}) == 2, 'just-below should be FS');
assert(classify_neuron_types(mk(5, 0.05, 0.4000, 10), {'V1'}) == 5, 'exactly 0.4 should be RS');
fprintf('  0.4 ms boundary ................ ok\n');

%% --- No non-striatal unit is ever left unclassified ----------------------
rng(0);
n = 500;
feats = [10*rand(n,1), rand(n,1), 0.1 + 0.9*rand(n,1), 80*rand(n,1)];
areas = repmat({'V1'}, n, 1);
codes = classify_neuron_types(feats, areas);
assert(~any(isnan(codes)), 'non-striatal units left as NaN');
assert(all(ismember(codes, [2 5])), 'non-striatal produced codes outside {FS,RS}');
expect_fs = sum(feats(:,3) < 0.4);
assert(sum(codes == 2) == expect_fs, 'FS count does not match the width rule');
fprintf('  every non-striatal unit classified (%d FS / %d RS) ... ok\n', ...
        sum(codes == 2), sum(codes == 5));

%% --- Missing features stay missing ---------------------------------------
codes = classify_neuron_types([NaN NaN NaN NaN; mk(2,0.05,0.6,10)], {'V1'; 'V1'});
assert(isnan(codes(1)) && codes(2) == 5, 'NaN features must stay NaN');
fprintf('  NaN features preserved ......... ok\n');

%% --- Unknown area falls back to non-striatal (safe default) --------------
assert(classify_neuron_types(mk(2, 0.05, 0.6, 10), {'RHP'}) == 5, ...
    'unknown area should use the non-striatal rule');
fprintf('  unknown area -> non-striatal ... ok\n');

fprintf('ALL TESTS PASSED\n');
end
