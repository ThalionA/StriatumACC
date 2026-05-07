function plot_striatum_pca(data, num_components, change_point_mean, dark_data)

if nargin < 2
    num_components = 3;
end

if isnan(change_point_mean)
    change_point_mean = size(data, 3);
end

[~, num_bins, n_trials] = size(data);

spatial_binned_fr_reshaped = data(:, :);

[coeff, score, ~, ~, explained, ~] = pca(spatial_binned_fr_reshaped', "NumComponents", num_components, "Centered", true);
cumsum_explained = cumsum(explained);
figure
plot(cumsum_explained, 'LineWidth', 1)
ylabel('explained variance (%)')
xlabel('component #')
hold on
idx_90 = find(cumsum_explained >= 90, 1);
% Add a vertical dotted line from (idx_90, 0) to (idx_90, cumsum_explained(idx_90))
plot([idx_90, idx_90], [0, cumsum_explained(idx_90)], '--', 'Color', [0.6 0.6 0.6]);
% Add a horizontal dotted line from (0, 90) to (idx_90, 90)
plot([0, idx_90], [90, 90], '--', 'Color', [0.6 0.6 0.6]);
axis tight

score_reshaped = reshape(score, [num_bins, n_trials, num_components]);

mean_score_early = squeeze(mean(score_reshaped(:, 1:3, :), 2));
mean_score_engaged = squeeze(mean(score_reshaped(:, 4:change_point_mean-10, :), 2));
mean_score_expert = squeeze(mean(score_reshaped(:, change_point_mean-9:change_point_mean, :), 2));
if change_point_mean < size(data, 3)
    mean_score_disengaged = squeeze(mean(score_reshaped(:, change_point_mean+1:change_point_mean + 50, :), 2));
end

saturation = linspace(0.1, 1, num_bins)';  % Saturation from 0.1 to 1

% Define base hues for each condition (H values in HSV)
H_early = 0.6667;    % Blue
H_engaged = 0.0833;  % Yellow
H_expert = 0.333;    % Green
H_disengaged = 0;    % Red

% Create HSV color arrays for each condition
HSV_early = [H_early * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_early = hsv2rgb(HSV_early);

HSV_engaged = [H_engaged * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_engaged = hsv2rgb(HSV_engaged);

HSV_expert = [H_expert * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_expert = hsv2rgb(HSV_expert);

HSV_disengaged = [H_disengaged * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_disengaged = hsv2rgb(HSV_disengaged);


figure
hold on

if num_components == 3

    if nargin > 3
        subplot(1, 2, 1);
    end

    % Plot early condition
    x = mean_score_early(:,1);
    y = mean_score_early(:,2);
    z = mean_score_early(:,3);
    for ii = 1:length(x)-1
        line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
            'Color', colors_early(ii,:), 'LineWidth', 2);
    end

    % Plot engaged condition
    x = mean_score_engaged(:,1);
    y = mean_score_engaged(:,2);
    z = mean_score_engaged(:,3);
    for ii = 1:length(x)-1
        line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
            'Color', colors_engaged(ii,:), 'LineWidth', 2);
    end

    % Plot engaged condition
    x = mean_score_expert(:,1);
    y = mean_score_expert(:,2);
    z = mean_score_expert(:,3);
    for ii = 1:length(x)-1
        line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
            'Color', colors_expert(ii,:), 'LineWidth', 2);
    end

    if change_point_mean < size(data, 3)
        % Plot disengaged condition
        x = mean_score_disengaged(:,1);
        y = mean_score_disengaged(:,2);
        z = mean_score_disengaged(:,3);
        for ii = 1:length(x)-1
            line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
                'Color', colors_disengaged(ii,:), 'LineWidth', 2);
        end

        %
    end
    hold off
    rotate3d on
    grid on
    view(-25, 45)
    xlabel('PC1')
    ylabel('PC2')
    zlabel('PC3')

    if nargin > 3
        subplot(1, 2, 2);

        hold on
        dark_nbins = size(dark_data, 2);
        dark_data_resh = dark_data(:, :);

        dark_scores = dark_data_resh'*coeff;
        dark_scores_resh = reshape(dark_scores, [dark_nbins, n_trials, num_components]);

        mean_dark_score_early = squeeze(mean(dark_scores_resh(:, 1:3, :), 2));
        mean_dark_score_engaged = squeeze(mean(dark_scores_resh(:, 4:change_point_mean-10, :), 2));
        mean_dark_score_expert = squeeze(mean(dark_scores_resh(:, change_point_mean-9:change_point_mean, :), 2));
        if change_point_mean < size(data, 3)
            mean_dark_score_disengaged = squeeze(mean(dark_scores_resh(:, change_point_mean+1:change_point_mean+50, :), 2));
        end
        
        % Plot early condition
        x = mean_dark_score_early(:,1);
        y = mean_dark_score_early(:,2);
        z = mean_dark_score_early(:,3);
        for ii = 1:length(x)-1
            line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
                'Color', colors_early(ii,:), 'LineWidth', 2);
        end

        % Plot engaged condition
        x = mean_dark_score_engaged(:,1);
        y = mean_dark_score_engaged(:,2);
        z = mean_dark_score_engaged(:,3);
        for ii = 1:length(x)-1
            line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
                'Color', colors_engaged(ii,:), 'LineWidth', 2);
        end

        % Plot engaged condition
        x = mean_dark_score_expert(:,1);
        y = mean_dark_score_expert(:,2);
        z = mean_dark_score_expert(:,3);
        for ii = 1:length(x)-1
            line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
                'Color', colors_expert(ii,:), 'LineWidth', 2);
        end

        if change_point_mean < size(data, 3)
            % Plot disengaged condition
            x = mean_dark_score_disengaged(:,1);
            y = mean_dark_score_disengaged(:,2);
            z = mean_dark_score_disengaged(:,3);
            for ii = 1:length(x)-1
                line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
                    'Color', colors_disengaged(ii,:), 'LineWidth', 2);
            end
        end
        hold off
        rotate3d on
        grid on
        view(-25, 45)

        xlabel('PC1')
        ylabel('PC2')
        zlabel('PC3')

        linkaxes
    end

elseif num_components == 2
    % Plot early condition
    x = mean_score_early(:,1);
    y = mean_score_early(:,2);
    for ii = 1:length(x)-1
        line([x(ii), x(ii+1)], [y(ii), y(ii+1)], ...
            'Color', colors_early(ii,:), 'LineWidth', 2);
    end

    % Plot engaged condition
    x = mean_score_engaged(:,1);
    y = mean_score_engaged(:,2);
    for ii = 1:length(x)-1
        line([x(ii), x(ii+1)], [y(ii), y(ii+1)], ...
            'Color', colors_engaged(ii,:), 'LineWidth', 2);
    end

    % Plot disengaged condition
    x = mean_score_disengaged(:,1);
    y = mean_score_disengaged(:,2);
    for ii = 1:length(x)-1
        line([x(ii), x(ii+1)], [y(ii), y(ii+1)], ...
            'Color', colors_disengaged(ii,:), 'LineWidth', 2);
    end
    hold off
end

title(sprintf('explained variance = %.2f%', cumsum_explained(num_components)))