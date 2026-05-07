function v = is_v1_safe(s)
% IS_V1_SAFE Defensive accessor for the optional `is_v1` field.
%
%   v = is_v1_safe(s)
%
% Returns s.is_v1 as a logical column vector if present and non-empty,
% otherwise an all-false column matching size(s.is_dms). Used so that
% V1-aware code keeps working on control mice (no V1 probe) and on
% datasets that predate the V1 column being saved.
%
% Created 2026-05-07.

    if isfield(s, 'is_v1') && ~isempty(s.is_v1)
        v = logical(s.is_v1(:));
    else
        v = false(size(s.is_dms(:)));
    end
end
