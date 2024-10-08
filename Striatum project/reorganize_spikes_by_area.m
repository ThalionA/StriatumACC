function all_data = reorganize_spikes_by_area(all_data, ianimal)
    % Reorganizes final_spikes and final_areas so that DMS units come first, then ACC units.

    is_dms = strcmp(all_data(ianimal).final_areas, 'DMS');
    is_acc = strcmp(all_data(ianimal).final_areas, 'ACC');

    % Reorder spikes
    temp_final_spikes = [all_data(ianimal).final_spikes(is_dms, :); all_data(ianimal).final_spikes(is_acc, :)];
    all_data(ianimal).final_spikes = temp_final_spikes;

    % Reorder areas
    all_data(ianimal).final_areas = [all_data(ianimal).final_areas(is_dms), all_data(ianimal).final_areas(is_acc)];
end