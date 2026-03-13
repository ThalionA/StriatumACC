% File: plotTSNEembedding.m
function plotTSNEembedding(model, labels_valid, cfg)
% Performs t-SNE on neuron factors and plots 2D embedding colored by area and mouse ID.

    fprintf('Performing t-SNE on neuron factors...\n');

    if isempty(model) || isempty(model.U{1})
        warning('plotTSNEembedding: Invalid model provided.');
        return;
    end
    if ~isfield(labels_valid, 'area_labels') || ~isfield(labels_valid, 'mouse_labels') || ...
        isempty(labels_valid.area_labels) || isempty(labels_valid.mouse_labels)
        warning('plotTSNEembedding: Valid area or mouse labels missing.');
        return;
    end
    if numel(labels_valid.area_labels) ~= size(model.U{1}, 1)
         warning('plotTSNEembedding: Label dimension mismatch with neuron factors.');
        return;
    end

    neuron_factors = model.U{1}; % N_valid_neurons x nFactors

    % Check if tsne function is available (Statistics Toolbox)
    if isempty(which('tsne'))
        warning('tsne function not found (requires Statistics and Machine Learning Toolbox). Skipping t-SNE plots.');
        return;
    end

    % Run t-SNE (adjust parameters as needed, e.g., Perplexity, Distance metric)
    fprintf('  Running t-SNE (perplexity 30)... This may take a moment.\n');
    try
        % Consider standardizing features? tsne often works okay without it.
        % neuron_factors_std = zscore(neuron_factors); % Optional
        perplexity_val = min(50, floor(size(neuron_factors, 1)/3) - 1); % Adjust perplexity if needed
        if perplexity_val < 5; perplexity_val = 5; end % Min perplexity
         fprintf('    Using perplexity: %d\n', perplexity_val);

        embedding_2d = tsne(neuron_factors, 'NumDimensions', 2, 'Perplexity', perplexity_val, 'Distance', 'cosine');
        fprintf('  t-SNE finished.\n');
    catch ME_tsne
        warning('Error during t-SNE execution: %s', "%s", ME_tsne.message);
        return; % Stop if t-SNE fails
    end

    % --- Plot t-SNE embedding colored by Brain Area ---
    figure;
    unique_areas = unique(labels_valid.area_labels);
    unique_areas = unique_areas(~strcmp(unique_areas, 'Unknown')); % Exclude 'Unknown'
    area_colors_present = cell2mat(values(cfg.plot.colors.area_map, unique_areas));
    area_labels_cat = categorical(labels_valid.area_labels);

    gscatter(embedding_2d(:,1), embedding_2d(:,2), area_labels_cat, area_colors_present, '.', 12);
    legend(unique_areas, 'Location', 'best'); % Show legend for areas
    xlabel('t-SNE Dimension 1');
    ylabel('t-SNE Dimension 2');
    title('t-SNE Embedding of Neuron Factors (Colored by Area)');
    axis tight; box off; grid on;

    % --- Plot t-SNE embedding colored by Mouse ID ---
    figure;
    mouse_labels_cat = categorical(labels_valid.mouse_labels);
    unique_mice = unique(labels_valid.mouse_labels);
    n_mice = numel(unique_mice);
    mouse_colors = parula(n_mice); % Use parula or another map for mice

    gscatter(embedding_2d(:,1), embedding_2d(:,2), mouse_labels_cat, mouse_colors, '.', 12);
    % Legend might be too crowded for many mice - consider omitting or customizing
    if n_mice <= 15 % Only show legend if not too many mice
         legend(arrayfun(@(x) sprintf('Mouse %d',x), unique_mice, 'UniformOutput', false),'Location', 'bestoutside');
    end
    xlabel('t-SNE Dimension 1');
    ylabel('t-SNE Dimension 2');
    title('t-SNE Embedding of Neuron Factors (Colored by Mouse ID)');
    axis tight; box off; grid on;

end