n_animals = numel(task_data);


zscored_lick_errors = {task_data(:).zscored_lick_errors};

learning_point = cellfun(@(x) find(movsum(x <= -2, [0,9]) >= 7, 1), zscored_lick_errors, 'UniformOutput', false);

change_points = {task_data(:).change_point_mean};


figure
t = tiledlayout('flow', 'TileSpacing', 'compact');
mov_window_size = 5;

for ianimal = 1:n_animals
    nexttile
    lick_errors = zscored_lick_errors{ianimal};
    trials = min([change_points{ianimal}, size(lick_errors, 2)]);

    shadedErrorBar(1:trials, movmean(lick_errors(1:trials), mov_window_size, 'omitmissing'), ...
        movstd(lick_errors(1:trials), mov_window_size, [], 2, 'omitmissing')/sqrt(mov_window_size))
    yline(-2, 'r')
    if ~isempty(learning_point{ianimal})
        xline(learning_point{ianimal})
        proportion_precise = sum(lick_errors(learning_point{ianimal}:trials) <= -2, 'omitmissing')/(size(lick_errors, 2));
    else
        proportion_precise = sum(lick_errors(1:trials) <= -2, 'omitmissing')/size(lick_errors, 2);
    end


    annotation_text = sprintf('Precise: %.2f%%', proportion_precise*100);
    % Add text annotation at a desired (x,y) coordinate, e.g. near the top-right:
    
    % legend(annotation_text, 'FontSize', 12)

    xlabel('Trial')
    ylabel('Z-scored Lick Error')
    axis tight
end

