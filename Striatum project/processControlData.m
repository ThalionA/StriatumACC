function [control_data_processed, control_indices_info] = processControlData(control_data_raw, ref_point, cfg)
% Processes control data: filters based on trial availability relative to a reference point.

    fprintf('Processing Control Mice...\n');
    n_animals_raw = length(control_data_raw);
    control_data_processed = []; % Initialize empty struct array
    control_indices_info = struct(); % To store index definitions

    if n_animals_raw == 0
        fprintf('  No raw control data provided.\n');
        return;
    end
     if isnan(ref_point) || isempty(ref_point)
        error('processControlData: Invalid reference point (NaN or empty) provided for control alignment.');
    end

    % --- Define trial indices based on reference point and config windows ---
    try
        control_indices_early = cfg.control_epoch_windows{1};
        control_indices_pre_ref = (ref_point + cfg.control_epoch_windows{2}(1)) : (ref_point + cfg.control_epoch_windows{2}(2));
        control_indices_post_ref = (ref_point + cfg.control_epoch_windows{3}(1)) : (ref_point + cfg.control_epoch_windows{3}(2));
    catch ME
        error('Error defining control indices from cfg.control_epoch_windows and ref_point=%d. Check config format. Original error: %s', ref_point, ME.message);
    end

    % --- Check indices validity (must be positive) ---
    if any(control_indices_pre_ref <= 0) || any(control_indices_post_ref <= 0)
        warning('Calculated control indices based on ref_point %d are not all positive (Pre: %d:%d, Post: %d:%d). Clamping to 1.', ...
              ref_point, min(control_indices_pre_ref), max(control_indices_pre_ref), min(control_indices_post_ref), max(control_indices_post_ref));
        control_indices_pre_ref(control_indices_pre_ref <= 0) = 1;
        control_indices_post_ref(control_indices_post_ref <= 0) = 1;
        % Consider adding an error option here instead of just warning/clamping
    end

    control_indices_all_needed = unique([control_indices_early, control_indices_pre_ref, control_indices_post_ref]);
    max_trial_needed_control = max(control_indices_all_needed);

    % Store the index definitions
    control_indices_info.early = control_indices_early;
    control_indices_info.pre_ref = control_indices_pre_ref;
    control_indices_info.post_ref = control_indices_post_ref;
    control_indices_info.max_trial_needed = max_trial_needed_control;

    fprintf('  Control trials required: Early [%s], Pre-Ref [%s], Post-Ref [%s]. Max trial needed: %d\n', ...
        mat2str(control_indices_early), mat2str(control_indices_pre_ref), mat2str(control_indices_post_ref), max_trial_needed_control);


    % --- Filter control mice based on trial availability ---
    valid_control_mask = false(1, n_animals_raw); % Keep track of which controls are used
    for i_control = 1:n_animals_raw
        % Ensure the neural data field exists and get trial count
        if isfield(control_data_raw(i_control), 'spatial_binned_fr_all')
            num_trials_control = size(control_data_raw(i_control).spatial_binned_fr_all, 3);
            if num_trials_control >= max_trial_needed_control
                valid_control_mask(i_control) = true;
            else
                 fprintf('  Excluding control mouse %d: Has %d trials, needs at least %d trials.\n', ...
                         i_control, num_trials_control, max_trial_needed_control);
            end
        else
             fprintf('  Excluding control mouse %d: Missing "spatial_binned_fr_all" field.\n', i_control);
        end
    end

    control_data_processed = control_data_raw(valid_control_mask);
    n_animals_processed = sum(valid_control_mask);
    fprintf('  Found %d/%d control mice with sufficient trials.\n', n_animals_processed, n_animals_raw);

end