function plot_decoder_accuracy_vs_chance(decoder_performance, neuron_count, n_pos_bins)
    % Check if n_pos_bins is provided; if not, set a default value
    if nargin < 3
        n_pos_bins = 50;  % Adjust this value based on your data
    end

    n_animals = length(decoder_performance);

    % Initialize variables to store accuracies
    all_accuracies = [];

    invalid_animals = false(1, n_animals);
    % Collect accuracies for each animal at the specified neuron count
    for ianimal = 1:n_animals
        % Get neuron counts for this animal
        animal_neuron_counts = decoder_performance(ianimal).neuron_counts;
        overall_accuracy = decoder_performance(ianimal).overall_accuracy;  % size: n_counts x n_bootstraps

        % Find the index for the specified neuron count
        idx = find(animal_neuron_counts == neuron_count);
        if isempty(idx)
            warning('Animal %d does not have data for neuron count %d. Skipping this animal.', ianimal, neuron_count);
            invalid_animals(ianimal) = true;
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

    valid_animals = ~invalid_animals;

    % Plot the accuracy
    figure
    my_errorbar_plot(all_accuracies')

    % Plot chance level
    chance_level = 1 / n_pos_bins;  % Assuming n_pos_bins is known
    yline(chance_level, 'r--', 'LineWidth', 1);

    % Customize the plot
    % xlim([0.5, 1.5]);
    % xticks(1);
    xticklabels(strsplit(sprintf('animal%d ', find(valid_animals))));
    ylabel('Decoder Accuracy');
    
end