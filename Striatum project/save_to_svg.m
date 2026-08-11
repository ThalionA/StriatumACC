function save_to_svg(name, fig)
%SAVE_TO_SVG Save a figure as an svg+png pair into figures/.
%   save_to_svg(NAME) saves the current figure; save_to_svg(NAME, FIG) saves
%   FIG. Writes figures/NAME.svg and figures/NAME.png (PNG longest side
%   capped at 1600 px per the project figure rule).
%
%   Repo-resident since 2026-08-11: IntegratedAll_v1.m has called a
%   save_to_svg that existed only on a personal MATLAB path (the old
%   root-level Behavioral_*.svg came from that shadow copy). This version
%   routes everything into figures/ so pipeline outputs land in one place.

if nargin < 2 || isempty(fig)
    fig = gcf;
end
out_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
% Vector first.
print(fig, fullfile(out_dir, [name '.svg']), '-dsvg', '-vector');
% PNG longest side <= 1600 px: pick dpi from the on-screen figure size.
pos = get(fig, 'Position');                    % [x y w h] in pixels
longest = max(pos(3:4));
dpi = floor(min(150, 1600 / max(longest, 1) * 96));
print(fig, fullfile(out_dir, [name '.png']), '-dpng', sprintf('-r%d', max(dpi, 60)));
end
