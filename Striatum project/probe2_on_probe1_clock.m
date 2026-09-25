function out = probe2_on_probe1_clock(spikes_p2, vr_times_p1, vr_times_p2, frames)
%PROBE2_ON_PROBE1_CLOCK Probe-2 spike bins resampled onto probe 1's 1 ms grid.
%
%   out = probe2_on_probe1_clock(spikes_p2, vr_times_p1, vr_times_p2, frames)
%
%   spikes_p2    units x bins, 1 ms bins on probe 2's own clock
%   vr_times_p1  VR frame times (s) on probe 1's clock (probe-1 bundle)
%   vr_times_p2  the SAME VR frames' times (s) on probe 2's clock (probe-2 bundle)
%   frames       probe-1 bin indices to produce (e.g. npx_start_frame:npx_end_frame)
%
% Each bundle's VR_times_synched was synced to its own probe. In the task
% cohort the two agree exactly; in the control cohort they drift apart (513:
% 6 -> 48 ms, 817: 116 -> 159 ms over the session), so slicing probe 2 with
% probe 1's frame indices misaligns it by that drifting amount. Measured
% 2026-09-25 (lfp/scripts/audit_probe2_clock.py): with the probe-2 bundle's own
% times all three controls show the same sharp V1 onset transient; with probe
% 1's times 817's is 120 ms late and smeared.
%
% Each probe-1 millisecond is mapped to probe 2's clock by interpolating
% between the shared VR frames, and the nearest probe-2 bin is taken. The
% drift is ~5 ppm, so this duplicates or skips one bin every few minutes.
% Bins beyond the probe-2 recording are zero, as the previous crop padded.
% With identical clocks the result is exactly spikes_p2(:, frames).

t2_ms = 1000 * interp1(vr_times_p1(:), vr_times_p2(:), frames(:)' / 1000, ...
                       'linear', 'extrap');
idx2 = round(t2_ms);
valid = idx2 >= 1 & idx2 <= size(spikes_p2, 2);
out = zeros(size(spikes_p2, 1), numel(frames), 'like', spikes_p2);
out(:, valid) = spikes_p2(:, idx2(valid));
end
