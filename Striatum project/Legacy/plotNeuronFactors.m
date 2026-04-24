% File: plotNeuronFactors.m
function plotNeuronFactors(model, labels, coloring_scheme, nFactors, nAnimalsTotal)
% Plots neuron factors (Mode 1) colored by mouse ID or potentially another scheme.

    fprintf('Plotting neuron factors colored by %s...\n', coloring_scheme);

    if isempty(model) || isempty(model.U{1}) || isempty(labels)
        warning('plotNeuronFactors: Invalid model or labels provided.');
        return;
    end

    neuron_loadings = model.U{1};
    nNeurons = size(neuron_loadings, 1);

    if numel(labels) ~= nNeurons
        warning('plotNeuronFactors: Mismatch between number of neurons (%d) and labels (%d).', nNeurons, numel(labels));
        return;
    end

    figure;
    t = tiledlayout('flow', 'TileSpacing', 'compact');
    title(t, sprintf('Neuron Factors (Mode 1) - Colored by %s', upper(coloring_scheme)));
    xlabel(t, 'Neuron Index (Concatenated)');
    ylabel(t, 'Loading');

    % Determine colors
    if strcmpi(coloring_scheme, 'mouse')
        % Use parula colormap based on total number of animals in the original combined set
        colors_to_use = parula(nAnimalsTotal);
        color_data = labels; % Assuming labels are mouse indices 1:nAnimalsTotal
        cblabel = 'Mouse ID';
        clim_vals = [1 nAnimalsTotal];
    else
        % Add other coloring schemes here if needed (e.g., 'area', 'group')
        warning('plotNeuronFactors: Unsupported coloring_scheme "%s". Using default.', coloring_scheme);
        colors_to_use = 'blue'; % Default single color
        color_data = 1; % Assign same color index to all
        cblabel = '';
        clim_vals = []; % No colorbar needed
    end

    for r = 1:nFactors
        nexttile;
        scatter(1:nNeurons, neuron_loadings(:,r), 10, color_data, 'filled');
        title(sprintf('Factor %d', r));
        axis tight;
        box off;

        % Apply colormap and colorbar only if multiple colors are used
        if ~ischar(colors_to_use) % Check if it's a colormap matrix
            colormap(gca, colors_to_use);
            if ~isempty(clim_vals) && clim_vals(1) ~= clim_vals(2)
                 clim(clim_vals);
                 if r == nFactors % Add colorbar to last or middle plot
                    cb = colorbar;
                    ylabel(cb, cblabel);
                    cb.Layout.Tile = 'east'; % Try to position nicely (might need adjustment)
                 end
            end
        end
    end
end