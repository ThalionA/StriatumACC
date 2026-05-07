function [task_data_processed, learning_points, avg_learning_point, aligned_lick_errors] = processTaskData(task_data_raw, cfg)
% Processes task data: calculates learning points, filters animals, aligns lick errors.

    fprintf('Processing Task Mice...\n');
    n_animals_raw = length(task_data_raw);
    task_data_processed = [];
    learning_points = {};
    avg_learning_point = [];
    aligned_lick_errors = []; % Initialize

    if n_animals_raw == 0
        fprintf('  No raw task data provided.\n');
        return;
    end

    % --- Calculate learning point for ALL task mice first ---
    zscored_lick_errors_all = {task_data_raw(:).zscored_lick_errors};

    % Check for required fields
    if ~isfield(task_data_raw, 'zscored_lick_errors')
        warning('processTaskData: Field "zscored_lick_errors" not found in task data. Cannot calculate learning points.');
        task_data_processed = task_data_raw; % Return unfiltered data maybe? Or empty?
        return;
    end

    % Calculate Learning Points using moving sum
    thresh = cfg.task_lp_zscore_threshold;
    win = cfg.task_lp_window_length; % e.g., 10 for current + next 9
    min_consecutive = cfg.task_lp_min_consecutive; % e.g., 8

    learning_point_all = cellfun(@(x) ...
        find(movsum(x <= thresh, [0, win-1], 'omitnan') >= min_consecutive, 1, 'first'), ...
        zscored_lick_errors_all, 'UniformOutput', false);

    has_learning_point = ~cellfun(@isempty, learning_point_all);

    % --- Filter task data ---
    task_data_processed = task_data_raw(has_learning_point);
    learning_points = learning_point_all(has_learning_point); % Keep only corresponding learning points
    zscored_lick_errors_filt = zscored_lick_errors_all(has_learning_point);

    n_animals_processed = length(task_data_processed);
    fprintf('  Found %d/%d task mice with a defined learning point.\n', n_animals_processed, n_animals_raw);

    if n_animals_processed == 0
        return; % No animals left, can't calculate average LP or align errors
    end

    % --- Calculate Average Learning Point ---
    avg_learning_point = round(mean(cell2mat(learning_points)));
    fprintf('  Average learning point from task mice: Trial %d\n', avg_learning_point);

    % --- Align Lick Errors ---
    n_trials_aligned = numel(cfg.control_epoch_windows{1}) + ...
                       abs(diff(cfg.control_epoch_windows{2})) + 1 + ... % Size of pre-window
                       abs(diff(cfg.control_epoch_windows{3})); % Size of post-window
    if n_trials_aligned ~= 30
         warning('Aligned lick error calculation assumes 30 trials (10 early, 10 pre, 10 post). Check cfg.control_epoch_windows.');
         % Adjust logic if the 30 trial assumption is relaxed
    end

    aligned_lick_errors = nan(n_animals_processed, 30); % Assuming 30 trials: 10 early, 10 pre-LP, 10 post-LP

    for ianimal = 1:n_animals_processed
        lp = learning_points{ianimal};
        errors = zscored_lick_errors_filt{ianimal};
        n_trials_mouse = numel(errors);

        % Define indices based on cfg.control_epoch_windows and individual LP
        idx_early = cfg.control_epoch_windows{1}; % e.g., 1:10
        idx_pre   = (lp + cfg.control_epoch_windows{2}(1)) : (lp + cfg.control_epoch_windows{2}(2)); % e.g., lp-10 : lp-1
        idx_post  = (lp + cfg.control_epoch_windows{3}(1)) : (lp + cfg.control_epoch_windows{3}(2)); % e.g., lp+1 : lp+10

        % Check if indices are valid for this animal
        valid_early = all(idx_early >= 1) && all(idx_early <= n_trials_mouse);
        valid_pre = all(idx_pre >= 1) && all(idx_pre <= n_trials_mouse);
        valid_post = all(idx_post >= 1) && all(idx_post <= n_trials_mouse);

        if valid_early && valid_pre && valid_post
            aligned_lick_errors(ianimal, 1:10)   = errors(idx_early);
            aligned_lick_errors(ianimal, 11:20) = errors(idx_pre);
            aligned_lick_errors(ianimal, 21:30) = errors(idx_post);
        else
            fprintf('  Warning: Task mouse %d (LP=%d, Trials=%d) lacks sufficient trials for lick error alignment. Leaving row as NaN.\n', ianimal, lp, n_trials_mouse);
        end
    end
    fprintf('  Aligned lick errors for task mice.\n');

end