function plot_striatum_pca_new(stim_data, num_components, number_of_trials, lick_errors, dark_data)

if nargin < 2
    num_components = 3;
end

stim_data = stim_data(:, :, 1:number_of_trials);
dark_data = dark_data(:, :, 1:number_of_trials);
lick_errors = lick_errors(1:number_of_trials);

[~, num_bins, n_trials] = size(stim_data);

first_idx = false(1, n_trials);
first_idx(1:3) = true;

precise_idx = lick_errors <= -2;
precise_idx(1:3) = false;

random_idx = lick_errors > -2;
random_idx(1:3) = false;

stim_data_reshaped = stim_data(:, :);

[coeff_stim, score_stim, ~, ~, explained_stim, ~] = pca(stim_data_reshaped', "NumComponents", num_components, "Centered", true);
cumsum_explained = cumsum(explained_stim);
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

score_reshaped_stim = reshape(score_stim, [num_bins, n_trials, num_components]);

mean_score_early = squeeze(mean(score_reshaped_stim(:, first_idx, :), 2));
mean_score_precise = squeeze(mean(score_reshaped_stim(:, precise_idx, :), 2));
mean_score_random = squeeze(mean(score_reshaped_stim(:, random_idx, :), 2));

saturation = linspace(0.1, 1, num_bins)';  % Saturation from 0.1 to 1

% Define base hues for each condition (H values in HSV)
H_early = 0.6667;    % Blue
H_precise = 0.333;    % Green
H_random = 0;    % Red

% Create HSV color arrays for each condition
HSV_early = [H_early * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_early = hsv2rgb(HSV_early);

HSV_precise = [H_precise * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_precise = hsv2rgb(HSV_precise);

HSV_random = [H_random * ones(num_bins,1), saturation, ones(num_bins,1)];
colors_random = hsv2rgb(HSV_random);


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

    % Plot precise condition
    x = mean_score_precise(:,1);
    y = mean_score_precise(:,2);
    z = mean_score_precise(:,3);
    for ii = 1:length(x)-1
        line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
            'Color', colors_precise(ii,:), 'LineWidth', 2);
    end

    % Plot random condition
    x = mean_score_random(:,1);
    y = mean_score_random(:,2);
    z = mean_score_random(:,3);
    for ii = 1:length(x)-1
        line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
            'Color', colors_random(ii,:), 'LineWidth', 2);
    end

    hold off
    rotate3d on
    grid on
    view(-25, 45)
    xlabel('PC1')
    ylabel('PC2')
    zlabel('PC3')


        subplot(1, 2, 2);

        hold on
        dark_nbins = size(dark_data, 2);
        dark_data_resh = dark_data(:, :);

        dark_scores = dark_data_resh'*coeff_stim;
        dark_scores_resh = reshape(dark_scores, [dark_nbins, n_trials, num_components]);

        mean_dark_score_early = squeeze(mean(dark_scores_resh(:, first_idx, :), 2));
        mean_dark_score_precise = squeeze(mean(dark_scores_resh(:, precise_idx, :), 2));
        mean_dark_score_random = squeeze(mean(dark_scores_resh(:, random_idx, :), 2));
        
        % Plot early condition
        x = mean_dark_score_early(:,1);
        y = mean_dark_score_early(:,2);
        z = mean_dark_score_early(:,3);
        for ii = 1:length(x)-1
            line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
                'Color', colors_early(ii,:), 'LineWidth', 2);
        end

        % Plot precise condition
        x = mean_dark_score_precise(:,1);
        y = mean_dark_score_precise(:,2);
        z = mean_dark_score_precise(:,3);
        for ii = 1:length(x)-1
            line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
                'Color', colors_precise(ii,:), 'LineWidth', 2);
        end

        % Plot random condition
        x = mean_dark_score_random(:,1);
        y = mean_dark_score_random(:,2);
        z = mean_dark_score_random(:,3);
        for ii = 1:length(x)-1
            line([x(ii), x(ii+1)], [y(ii), y(ii+1)], [z(ii), z(ii+1)], ...
                'Color', colors_random(ii,:), 'LineWidth', 2);
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

title(sprintf('explained variance = %.2f%', cumsum_explained(num_components)))