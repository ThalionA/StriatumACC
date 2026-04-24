function all_data = reorganize_spikes_by_area(all_data, ianimal)
    % Reorganizes final_spikes, final_areas, and neurontypes so that 
    % DMS units come first, then DLS, then ACC, then V1 units.
    % Robust to input orientation (row vs column vectors).

    % Extract current data for readability and safety
    curr_areas = all_data(ianimal).final_areas;
    curr_spikes = all_data(ianimal).final_spikes;
    curr_types = all_data(ianimal).final_neurontypes;

    % Create logical masks (strcmp works regardless of orientation)
    is_dms = strcmp(curr_areas, 'DMS');
    is_dls = strcmp(curr_areas, 'DLS');
    is_acc = strcmp(curr_areas, 'ACC');
    is_v1  = strcmp(curr_areas, 'V1'); % NEW: Find V1

    % --- 1. Reorder Spikes (Always Vertical Concatenation) ---
    all_data(ianimal).final_spikes = [curr_spikes(is_dms, :); ...
                                      curr_spikes(is_dls, :); ...
                                      curr_spikes(is_acc, :); ...
                                      curr_spikes(is_v1, :)]; % NEW: Keep V1 spikes

    % --- 2. Reorder Areas (Force Horizontal/Row Vector) ---
    dms_a = curr_areas(is_dms);
    dls_a = curr_areas(is_dls);
    acc_a = curr_areas(is_acc);
    v1_a  = curr_areas(is_v1); % NEW
    
    % Force to 1xN (Row) and concatenate
    all_data(ianimal).final_areas = [dms_a(:)', dls_a(:)', acc_a(:)', v1_a(:)']; % NEW: Keep V1 areas

    % --- 3. Reorder Neuron Types (Force Vertical/Column Vector) ---
    dms_t = curr_types(is_dms, :);
    dls_t = curr_types(is_dls, :);
    acc_t = curr_types(is_acc, :);
    v1_t  = curr_types(is_v1, :); % NEW
    
    if isvector(curr_types)
         all_data(ianimal).final_neurontypes = [dms_t(:); dls_t(:); acc_t(:); v1_t(:)]; % NEW
    else
         all_data(ianimal).final_neurontypes = [dms_t; dls_t; acc_t; v1_t]; % NEW
    end
end