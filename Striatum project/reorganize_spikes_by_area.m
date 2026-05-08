function all_data = reorganize_spikes_by_area(all_data, ianimal, area_order)
% REORGANIZE_SPIKES_BY_AREA  Reorder a mouse's neurons by area.
%
%   all_data = reorganize_spikes_by_area(all_data, ianimal)
%   all_data = reorganize_spikes_by_area(all_data, ianimal, {'DMS','DLS','ACC','V1','CA1','DG'})
%
% Reorders final_spikes, final_areas and final_neurontypes so that all units
% from area_order{1} come first, then area_order{2}, etc. Units with an area
% label not present in area_order are appended at the end in their existing
% order.
%
% If `area_order` is omitted, the order is taken from project_cfg().areas
% (which currently lists DMS, DLS, ACC, V1, CA1, DG).
%
% Generalised 2026-05-08; previously V1-aware but hardcoded.

    if nargin < 3 || isempty(area_order)
        cfg = project_cfg();
        area_order = cfg.areas;
    end

    curr_areas  = all_data(ianimal).final_areas;
    curr_spikes = all_data(ianimal).final_spikes;
    curr_types  = all_data(ianimal).final_neurontypes;

    n_units = numel(curr_areas);
    new_order = [];

    % Append each area's units in the order requested
    for i = 1:numel(area_order)
        idx = find(strcmp(curr_areas, area_order{i}));
        new_order = [new_order; idx(:)]; %#ok<AGROW>
    end

    % Append any leftover units (areas not in `area_order`) so we don't drop them
    in_order = false(n_units, 1);
    in_order(new_order) = true;
    new_order = [new_order; find(~in_order)];

    % --- Apply reordering ---
    all_data(ianimal).final_spikes = curr_spikes(new_order, :);

    reordered_areas = curr_areas(new_order);
    if isrow(curr_areas)
        all_data(ianimal).final_areas = reordered_areas(:)';
    else
        all_data(ianimal).final_areas = reordered_areas(:);
    end

    if isvector(curr_types)
        ct = curr_types(:);
        all_data(ianimal).final_neurontypes = ct(new_order);
    else
        all_data(ianimal).final_neurontypes = curr_types(new_order, :);
    end
end
