function v = is_v1_safe(s)
% IS_V1_SAFE  Backward-compat wrapper around is_area_safe.
%
%   v = is_v1_safe(s)   is equivalent to   v = is_area_safe(s, 'V1')
%
% Kept so existing callers don't need updating. New code should call
% is_area_safe(s, area_name) directly.

    v = is_area_safe(s, 'V1');
end
