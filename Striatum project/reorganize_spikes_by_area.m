function all_data = reorganize_spikes_by_area(all_data, ianimal)
    % Reorganizes final_spikes, final_areas, and neurontypes so that 
    % DMS units come first, then DLS, then ACC units.
    % Robust to input orientation (row vs column vectors).

    % Extract current data for readability and safety
    curr_areas = all_data(ianimal).final_areas;
    curr_spikes = all_data(ianimal).final_spikes;
    curr_types = all_data(ianimal).final_neurontypes;

    % Create logical masks (strcmp works regardless of orientation)
    is_dms = strcmp(curr_areas, 'DMS');
    is_dls = strcmp(curr_areas, 'DLS');
    is_acc = strcmp(curr_areas, 'ACC');

    % --- 1. Reorder Spikes (Always Vertical Concatenation) ---
    % Assumes spikes are [Neurons x Time]. 
    % We do not reshape here to preserve time dimensions.
    all_data(ianimal).final_spikes = [curr_spikes(is_dms, :); ...
                                      curr_spikes(is_dls, :); ...
                                      curr_spikes(is_acc, :)];

    % --- 2. Reorder Areas (Force Horizontal/Row Vector) ---
    % We force the subsets to be row vectors (:)' before concatenating.
    % This prevents 'horzcat' errors if the input was a column vector.
    
    dms_a = curr_areas(is_dms);
    dls_a = curr_areas(is_dls);
    acc_a = curr_areas(is_acc);
    
    % Force to 1xN (Row) and concatenate
    all_data(ianimal).final_areas = [dms_a(:)', dls_a(:)', acc_a(:)'];

    % --- 3. Reorder Neuron Types (Force Vertical/Column Vector) ---
    % We force the subsets to be column vectors (:).
    % This prevents dimension mismatch if types were stored as rows.
    
    dms_t = curr_types(is_dms, :);
    dls_t = curr_types(is_dls, :);
    acc_t = curr_types(is_acc, :);
    
    % Force to Nx1 (Column) or NxM (if types are matrices) and concatenate vertically
    % Note: If types is strictly a vector, (:) ensures column. 
    % If it has columns (e.g. multiple type flags), we keep (:,:)
    if isvector(curr_types)
         all_data(ianimal).final_neurontypes = [dms_t(:); dls_t(:); acc_t(:)];
    else
         all_data(ianimal).final_neurontypes = [dms_t; dls_t; acc_t];
    end
end