function filtered_data = filterDataByArea(data_raw, areas_to_include, area_field_map)
% Filters raw data structure to include only specified brain areas.

    if isempty(areas_to_include) || isempty(data_raw)
        filtered_data = data_raw; % Return original if no filtering needed or data empty
        return;
    end

    % Validate area names
    valid_areas = isKey(area_field_map, areas_to_include);
    if ~all(valid_areas)
        invalid_str = strjoin(areas_to_include(~valid_areas), ', ');
        error('Invalid area names provided: %s. Valid names are: %s', invalid_str, strjoin(keys(area_field_map), ', '));
    end

    area_fields = values(area_field_map, areas_to_include); % Get field names like 'is_dms'

    filtered_data = data_raw; % Initialize with the input structure

    for i = 1:length(filtered_data)
        keep_mask = false(size(filtered_data(i).(area_fields{1}))); % Initialize mask based on first area field size

        % Build the combined mask (OR across selected areas)
        for k = 1:length(area_fields)
            if isfield(filtered_data(i), area_fields{k})
                keep_mask = keep_mask | filtered_data(i).(area_fields{k});
            else
                warning('Animal %d missing area field: %s. Assuming false for this area.', i, area_fields{k});
            end
        end

        % Apply the mask to relevant neuron-dimensioned fields
        % Add other fields here if necessary!
        if isfield(filtered_data(i), 'spatial_binned_fr_all')
            filtered_data(i).spatial_binned_fr_all = filtered_data(i).spatial_binned_fr_all(keep_mask, :, :);
            if isfield(filtered_data(i), 'final_neurontypes')
                filtered_data(i).final_neurontypes = filtered_data(i).final_neurontypes(keep_mask, :);
            end
        end

        % Apply mask to the area flag fields themselves
        all_area_fields = values(area_field_map);
        for k = 1:length(all_area_fields)
             if isfield(filtered_data(i), all_area_fields{k})
                 filtered_data(i).(all_area_fields{k}) = filtered_data(i).(all_area_fields{k})(keep_mask);
             end
        end
         % Add other neuron-indexed fields if they exist, e.g., unit IDs, quality metrics
         % if isfield(filtered_data(i), 'unit_ids')
         %    filtered_data(i).unit_ids = filtered_data(i).unit_ids(keep_mask);
         % end
    end

    % Remove animals that have no units left after filtering
    units_per_animal = arrayfun(@(x) size(x.spatial_binned_fr_all, 1), filtered_data);
    filtered_data = filtered_data(units_per_animal > 0);

    if isempty(filtered_data) && ~isempty(data_raw)
        warning('filterDataByArea: No animals remaining after filtering for areas: %s', strjoin(areas_to_include, ', '));
    end
end