function v = is_area_safe(s, area)
% IS_AREA_SAFE Defensive accessor for any optional `is_<area>` field.
%
%   v = is_area_safe(s, 'V1')
%   v = is_area_safe(s, 'CA1')
%   v = is_area_safe(s, 'DG')
%
% Returns s.(['is_' lower(area)]) as a logical column vector if present
% and non-empty; otherwise an all-false column matching size(s.is_dms).
% Used so V1/CA1/DG-aware code keeps working on control mice and on
% datasets that predate those columns being saved.
%
% Created 2026-05-08 (generalised from is_v1_safe).

    if nargin < 2 || isempty(area)
        error('is_area_safe:NoArea', 'Pass the area name as the second argument.');
    end
    fname = ['is_' lower(area)];
    if isfield(s, fname) && ~isempty(s.(fname))
        v = logical(s.(fname)(:));
    else
        v = false(size(s.is_dms(:)));
    end
end
