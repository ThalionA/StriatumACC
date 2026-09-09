function n = save_all_open_figures(prefix)
%SAVE_ALL_OPEN_FIGURES Save every open figure as an svg+png pair.
%   N = save_all_open_figures(PREFIX) sweeps all open figures (ordered by
%   figure number) and saves each via save_to_svg as
%   figures/PREFIX_<NN>_<name>.{svg,png}, where <name> is the figure's Name
%   property (sanitised) or 'fig' when unnamed. Returns the number saved.
%
%   Intended as the last line of the pipeline scripts (Run_TCA_pipeline,
%   ensemble_analysis, SpatioTemporalActivityEvolution) so headless -batch
%   runs leave every figure on disk instead of dying with the process
%   (added 2026-08-11 so runs never need repeating just to see a figure).

figs = findobj('Type', 'figure');
if isempty(figs)
    n = 0;
    return
end
figs = figs(isgraphics(figs, 'figure'));          % drop stale/deleted handles
[~, order] = sort([figs.Number]);
figs = figs(order);
n = 0;
skipped = {};
for k = 1:numel(figs)
    if ~isgraphics(figs(k), 'figure')
        continue
    end
    % Already written by save_to_svg under its own descriptive name: saving it
    % again here would put the same pixels on disk twice under two names, which
    % is what this sweep used to do for every figure of three pipelines.
    % The sweep remains the ONLY save path for Run_TCA_pipeline and
    % ensemble_analysis, which name no figures, so it is not removed.
    already = getappdata(figs(k), 'SaveToSvgNames');
    if ~isempty(already)
        skipped{end+1} = already{1}; %#ok<AGROW>
        continue
    end
    raw = get(figs(k), 'Name');
    if isempty(raw)
        raw = 'fig';
    end
    clean = regexprep(lower(strtrim(raw)), '[^a-z0-9]+', '_');
    clean = regexprep(clean, '^_+|_+$', '');
    if isempty(clean)
        clean = 'fig';
    end
    % One bad figure must never cost the whole sweep (a stale handle killed
    % the 2026-08-11 spatiotemporal run at its final line).
    try
        save_to_svg(sprintf('%s_%02d_%s', prefix, k, clean), figs(k));
        n = n + 1;
    catch err
        warning('save_all_open_figures:printFailed', ...
                'Figure %d ("%s") not saved: %s', k, clean, err.message);
    end
end
if ~isempty(skipped)
    fprintf(['Skipped %d figure(s) already saved by save_to_svg (e.g. %s) — ' ...
             'they are on disk under their own names, not under a %s_NN_ one.\n'], ...
            numel(skipped), skipped{1}, prefix);
end
fprintf('Saved %d of %d figures to figures/ with prefix "%s_" (svg+png pairs).\n', ...
        n, numel(figs), prefix);
end
