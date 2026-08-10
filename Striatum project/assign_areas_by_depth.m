function labels = assign_areas_by_depth(unit_depths, area_names, starts, ends)
%ASSIGN_AREAS_BY_DEPTH  Label each unit by the depth band it falls into.
%   labels = ASSIGN_AREAS_BY_DEPTH(unit_depths, area_names, starts, ends)
%
%   unit_depths : N-by-1 numeric depths (um).
%   area_names  : 1-by-K cellstr/string of area labels.
%   starts,ends : 1-by-K numeric inclusive band edges (um). A NaN start OR
%                 end leaves that area unassigned (e.g. an empty CSV cell,
%                 which readtable imports as NaN).
%   labels      : N-by-1 cell array of char labels; '' where a unit matched
%                 no band.
%
%   Bands are applied in order, so a later area OVERWRITES an earlier one on
%   overlap. This reproduces the DMS -> DLS -> ACC precedence the Organise*
%   scripts use (DLS wins over DMS, ACC wins over both); pass the areas in
%   that order to preserve it.
%
%   See also OrganiseStriatumDataControlIncV1, OrganiseStriatumDataIncV1.

    unit_depths = unit_depths(:);
    area_names  = cellstr(area_names);
    n = numel(unit_depths);
    labels = repmat({''}, n, 1);

    for k = 1:numel(area_names)
        s = starts(k);
        e = ends(k);
        if isnan(s) || isnan(e)
            continue;   % unspecified band (empty CSV cell) -> assign nothing
        end
        in_band = unit_depths >= s & unit_depths <= e;
        labels(in_band) = area_names(k);
    end
end
