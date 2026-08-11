function type_codes = classify_neuron_types(nt_features, area_labels, pt_threshold_ms)
%CLASSIFY_NEURON_TYPES Assign cell-type codes using area-appropriate criteria.
%
%   TYPE_CODES = CLASSIFY_NEURON_TYPES(NT_FEATURES, AREA_LABELS) returns one
%   integer code per unit. NT_FEATURES is [n_units x >=4], matching the
%   layout of RawData/<mouse>_neurontype2025.mat:
%       col 1  mean firing rate (Hz)
%       col 2  proportion of long ISIs
%       col 3  peak-to-trough time (ms)   <- the waveform-width axis
%       col 4  post-spike ISI metric
%   AREA_LABELS is a cellstr (or char) of the same length.
%
%   Striatal areas (DMS, DLS) keep the legacy four-way criteria:
%       1 MSN   broad waveform, low ISI metric
%       2 FSN   narrow waveform, low long-ISI proportion
%       3 TAN   broad waveform, high ISI metric
%       4 UIN   narrow waveform, high long-ISI proportion
%
%   Cortex and hippocampus (ACC, V1, CA1, DG, and any other area) get a
%   BINARY split on waveform width — anything that is not fast-spiking is
%   regular-spiking, so no unit is left unclassified:
%       2 FS    peak-to-trough < PT_THRESHOLD_MS (default 0.4)
%       5 RS    everything else
%
%   Rationale (2026-08-11): before this function, the striatal four-way rule
%   was applied to ACC as though cortical units were MSN/TAN/UIN, while
%   probe-2 units (V1/CA1/DG) were never classified at all — the organiser
%   hardcoded NaN and never loaded <mouse>_v1_neurontype2025.mat, which has
%   existed all along. Every "RS" figure panel was therefore empty and V1
%   showed zero units of BOTH types.
%
%   Units with missing features stay NaN.
%
%   See also: test_classify_neuron_types, OrganiseStriatumDataIncV1.

if nargin < 3 || isempty(pt_threshold_ms)
    pt_threshold_ms = 0.4;         % standard narrow/broad waveform boundary
end
if ischar(area_labels) || isstring(area_labels)
    area_labels = cellstr(area_labels);
end

n_units = size(nt_features, 1);
assert(numel(area_labels) == n_units, ...
    'classify_neuron_types:sizeMismatch', ...
    '%d feature rows but %d area labels', n_units, numel(area_labels));

type_codes = nan(n_units, 1);
if n_units == 0
    return
end

% col 1 (mean firing rate) is not used by either rule; the layout is
% documented above so callers keep the column meanings straight.
prop = nt_features(:, 2);
pt   = nt_features(:, 3);         % peak-to-trough time (ms)
isi  = nt_features(:, 4);

is_striatal = ismember(upper(strtrim(area_labels(:))), {'DMS', 'DLS'});
narrow      = pt < pt_threshold_ms;
have_feats  = isfinite(pt);

% --- Striatum: legacy four-way rule (unchanged) --------------------------
s = is_striatal & have_feats;
type_codes(s & ~narrow & isi <= 40)  = 1;   % MSN
type_codes(s &  narrow & prop <  0.1) = 2;  % FSN
type_codes(s & ~narrow & isi >  40)  = 3;   % TAN
type_codes(s &  narrow & prop >= 0.1) = 4;  % UIN

% --- Cortex / hippocampus: fast-spiking vs everything else ---------------
c = ~is_striatal & have_feats;
type_codes(c &  narrow) = 2;                % FS
type_codes(c & ~narrow) = 5;                % RS
end
