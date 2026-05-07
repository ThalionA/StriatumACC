function plotSpatialFactors(model, zone_params, nFactors)
% plotSpatialFactors  Plot Mode-2 (spatial) factors with multiple landmark patches.
%
% REQUIRED FIELDS in zone_params
%   .bin_size            – scalar, AU per spatial bin
%   .corridor_end_au     – scalar, last AU position in the corridor
%   .visual_zones_au     – [N×2] matrix, each row = [startAU endAU]
%   .reward_zone_au      – [1×2] vector, [startAU endAU]
%
% EXAMPLE:
%   zone_params.visual_zones_au = [80 90; 105 115];   % two blue landmarks
%   zone_params.reward_zone_au  = [125 135];          % red reward zone
%

% ---------- sanity checks -------------------------------------------------
if isempty(model) || numel(model.U) < 2
    disp('Skipping plot: model is empty or malformed'); return; end

if ~isfield(zone_params,'visual_zones_au') || isempty(zone_params.visual_zones_au)
    error('zone_params.visual_zones_au must be an N×2 array');
end
if ~isfield(zone_params,'reward_zone_au') || numel(zone_params.reward_zone_au)~=2
    error('zone_params.reward_zone_au must be a 1×2 vector');
end

% ---------- derive bin centres -------------------------------------------
bin_edges      = 0:zone_params.bin_size:zone_params.corridor_end_au;
if bin_edges(end) < zone_params.corridor_end_au
    bin_edges(end+1) = bin_edges(end)+zone_params.bin_size;
end
bin_centres    = bin_edges(1:end-1) + diff(bin_edges)/2;      % [1×nbins]
nbins_expected = size(model.U{2},1);
if numel(bin_centres) ~= nbins_expected
    warning('Bin mismatch – using 1:nbins for x-axis'); 
    bin_centres = 1:nbins_expected;
end

% convenient inline to convert AU → [idxStart idxEnd]
au2idx = @(au_pair) [ ...
        find(bin_centres >= au_pair(1), 1,'first'), ...
        find(bin_centres < au_pair(2), 1,'last')];

visual_idx  = arrayfun(@(r) au2idx(zone_params.visual_zones_au(r,:)), ...
                       (1:size(zone_params.visual_zones_au,1)).', ...
                       'UniformOutput', false);
reward_idx  = au2idx(zone_params.reward_zone_au);

% ---------- plotting ------------------------------------------------------
figure;
t = tiledlayout('flow','TileSpacing','compact');
title(t,'Spatial factors (Mode 2)'); xlabel(t,'Spatial bin'); ylabel(t,'Loading');

Y = model.U{2};
ymin = min(Y(:)); ymax = max(Y(:));
if ymin==ymax, ymin=0; ymax=1; end
patchY = [ymin ymin ymax ymax];

for f = 1:nFactors
    nexttile; hold on;
    plot(Y(:,f),'k','LineWidth',2);           % factor trace
    ylim([ymin ymax]);

    % ––– blue patches for *every* visual zone –––
    for v = 1:numel(visual_idx)
        idx = visual_idx{v};
        if ~isempty(idx) && all(~isnan(idx))
            patch(idx([1 2 2 1])+[-.5 .5 .5 -.5], patchY, ...
                  [0 0.447 0.741],'EdgeColor','none','FaceAlpha',0.15);
        end
    end

    % ––– red patch for reward zone –––
    if ~isempty(reward_idx) && all(~isnan(reward_idx))
        patch(reward_idx([1 2 2 1])+[-.5 .5 .5 -.5], patchY, ...
              [0.850 0.325 0.098],'EdgeColor','none','FaceAlpha',0.15);
    end

    uistack(findobj(gca,'Type','line'),'top'); % keep trace above patches
    title(sprintf('Factor %d',f)); box off; hold off;
end
end