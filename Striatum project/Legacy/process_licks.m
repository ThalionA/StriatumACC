function filtered_licks = process_licks(licks, times, min_gap_ms)
    % remove_close_licks removes licks that are within min_gap_ms of each other.
    %
    % Inputs:
    %   licks      - Binary vector (0 or 1) indicating licks.
    %   times      - Vector of timestamps corresponding to each lick (in ms).
    %   min_gap_ms - Minimum gap between licks in milliseconds (default: 100 ms).
    %
    % Outputs:
    %   filtered_licks  - Binary vector with licks spaced at least min_gap_ms apart.
    %   filtered_times  - Corresponding timestamps of the filtered licks.

    if nargin < 3
        min_gap_ms = 100; % Default minimum gap
    end

    % Ensure inputs are column vectors for consistency
    licks = licks(:);
    times = times(:);

    % Step 1: Find indices where licks occur
    lick_indices = find(licks == 1);

    if isempty(lick_indices)
        % No licks to process
        filtered_licks = licks;
        return;
    end

    % Step 2: Extract times of the licks
    lick_times = times(lick_indices);

    % Step 3: Initialize a logical mask to keep licks
    keep = false(size(lick_times));
    keep(1) = true; % Always keep the first lick

    % Compute the time differences between consecutive licks

    % Cumulative sum of time differences
    % Reset the cumulative sum whenever a lick is kept
    % This helps in determining whether the current lick is at least min_gap_ms away from the last kept lick
    last_kept = lick_times(1);
    for ilick = 2:length(lick_times)
        if (lick_times(ilick) - last_kept) >= min_gap_ms
            keep(ilick) = true;
            last_kept = lick_times(ilick);
        else
            keep(ilick) = false;
        end
    end

    % Step 4: Create filtered vectors
    filtered_licks = zeros(size(licks));
    filtered_licks(lick_indices(keep)) = 1;

end
