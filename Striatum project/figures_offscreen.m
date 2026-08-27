function guard = figures_offscreen()
%FIGURES_OFFSCREEN Build figures off-screen so they don't steal focus.
%
%   fig_guard = figures_offscreen();          % top of a plotting script
%   ...
%   clear fig_guard                           % bottom: restores visibility
%
% New figures are created with Visible='off', so they neither pop to the front
% nor stack up while a long script runs. They render and save exactly as
% before — print/saveas, save_to_svg and save_all_open_figures all operate on
% invisible figures unchanged.
%
% Visibility is restored when GUARD is cleared. Scripts here clear it at the
% end; because every plotting script also starts with `clearvars`, an aborted
% run's guard is released the next time you run one. To see a suppressed
% figure from the last run: set(f,'Visible','on') on its handle, or re-run
% with MATLAB_SHOW_FIGURES=1.
%
% Opt out entirely:  setenv('MATLAB_SHOW_FIGURES','1')
%
% Created 2026-08-27.

    if strcmp(getenv('MATLAB_SHOW_FIGURES'), '1')
        guard = onCleanup(@() []);       % no-op guard: figures display as usual
        return
    end
    prev = get(groot, 'DefaultFigureVisible');
    set(groot, 'DefaultFigureVisible', 'off');
    guard = onCleanup(@() set(groot, 'DefaultFigureVisible', prev));
end
