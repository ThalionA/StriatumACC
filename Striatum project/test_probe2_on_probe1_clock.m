function test_probe2_on_probe1_clock()
%TEST_PROBE2_ON_PROBE1_CLOCK Probe-2 spikes resampled onto probe 1's ms grid.
%
% Each probe's bundle carries its own VR_times_synched: the time of every VR
% frame on THAT probe's sample clock. In the control cohort the two clocks drift
% apart (817: 116 -> 159 ms over the session), so cropping probe 2 with probe 1's
% frame indices misaligns it by that amount. The helper maps each probe-1
% millisecond to probe 2's clock through the shared VR frames.
%
%   /Applications/MATLAB_R2026a.app/bin/matlab -batch "test_probe2_on_probe1_clock"

fprintf('test_probe2_on_probe1_clock\n');
n_units = 3;
n_bins = 20000;
spikes = zeros(n_units, n_bins);
spikes(:, 1:n_bins) = repmat(1:n_bins, n_units, 1);   % value == bin index
vr1 = (1:0.5:15)';                                     % seconds, probe-1 clock
frames = 1000:14000;

%% --- identical clocks: plain slice ----------------------------------------
out = probe2_on_probe1_clock(spikes, vr1, vr1, frames);
assert(isequal(out, spikes(:, frames)), 'identical clocks must reduce to a slice');
fprintf('  identical clocks ................ ok\n');

%% --- constant offset: probe 2 runs 50 ms ahead ----------------------------
out = probe2_on_probe1_clock(spikes, vr1, vr1 + 0.050, frames);
assert(isequal(out(1, :), frames + 50), 'constant offset');
fprintf('  constant 50 ms offset ........... ok\n');

%% --- drifting offset: 0 ms at the start, 40 ms at the end -----------------
drift = 0.040 * (vr1 - vr1(1)) / (vr1(end) - vr1(1));
out = probe2_on_probe1_clock(spikes, vr1, vr1 + drift, frames);
expected = frames + 1000 * interp1(vr1, drift, frames / 1000, 'linear', 'extrap');
% nearest bin: within half a bin, give or take floating point at x.5
assert(max(abs(out(1, :) - expected)) <= 0.5 + 1e-6, 'drifting offset');
assert(out(1, end) - frames(end) >= 37, 'the end must carry ~40 ms of drift');
fprintf('  drifting offset ................. ok\n');

%% --- beyond the probe-2 recording: zeros, as the old crop padded ----------
out = probe2_on_probe1_clock(spikes(:, 1:12000), vr1, vr1, frames);
assert(all(out(:, frames > 12000) == 0, 'all'), 'past the end must be zero');
assert(isequal(out(1, frames <= 12000), frames(frames <= 12000)), 'inside is unchanged');
fprintf('  past the recording end .......... ok\n');

fprintf('test_probe2_on_probe1_clock: all passed\n');
end
