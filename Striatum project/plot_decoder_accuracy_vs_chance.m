function plot_decoder_accuracy_vs_chance(decoder_performance, neuron_count, n_pos_bins)
    % Check if n_pos_bins is provided; if not, set a default value
    if nargin < 3
        n_pos_bins = 50;  % Adjust this value based on your data
    end

    n_animals = length(decoder_performance);

    % Initialize variables to store accuracies
    all_accuracies = [];

    % Collect accuracies for each animal at the specified neuron count
    for ianimal = 1:n_animals
        % Get neuron counts for this animal
        animal_neuron_counts = decoder_performance(ianimal).neuron_counts;
        overall_accuracy = decoder_performance(ianimal).overall_accuracy;  % size: n_counts x n_bootstraps

        % Find the index for the specified neuron count
        idx = find(animal_neuron_counts == neuron_count);
        if isempty(idx)
            warning('Animal %d does not have data for neuron count %d. Skipping this animal.', ianimal, neuron_count);
            continue;
        end

        % Extract accuracies for this neuron count
        accuracies = overall_accuracy(idx, :);  % 1 x n_bootstraps

        % Append to the list of all accuracies
        all_accuracies = [all_accuracies; accuracies];
    end

    % Check if any data was collected
    if isempty(all_accuracies)
        error('No data available for neuron count %d across all animals.', neuron_count);
    end

    % Compute mean and SEM across all animals and bootstraps
    mean_acc = mean(all_accuracies(:), 'omitnan');
    sem_acc = std(all_accuracies(:), [], 'omitnan') / sqrt(numel(all_accuracies));

    % Plot the accuracy
    figure;
    hold on;

    % Bar plot of the decoder accuracy
    bar(1, mean_acc, 'FaceColor', 'b');
    errorbar(1, mean_acc, sem_acc, 'k', 'LineWidth', 1.5);

    % Plot chance level
    chance_level = 1 / n_pos_bins;  % Assuming n_pos_bins is known
    yline(chance_level, 'r--', 'LineWidth', 1);

    % Customize the plot
    xlim([0.5, 1.5]);
    xticks(1);
    xticklabels({sprintf('%d Neurons', neuron_count)});
    ylabel('Decoder Accuracy');
    title(sprintf('Decoder Accuracy at Neuron Count %d', neuron_count));
    legend({'Decoder Accuracy', 'Chance Level'}, 'Location', 'best');
    hold off;
end