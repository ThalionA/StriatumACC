function test_figures_offscreen()
%TEST_FIGURES_OFFSCREEN Guard suppresses display and always restores it.
%
% Restoration is tested on the two paths that actually occur: the guard going
% out of scope when its owning function returns, and an error unwinding it.
% (`clear` inside a function is not a reliable way to release an onCleanup, so
% it is not used here — scripts release it by ending or by the next
% `clearvars`.)
fprintf('test_figures_offscreen\n');
before = get(groot, 'DefaultFigureVisible');

% --- suppression + still saveable, guard released on function return -------
saved = use_guard_once();
assert(saved, 'invisible figure must still print to svg');
assert(strcmp(char(get(groot, 'DefaultFigureVisible')), char(before)), ...
       'must restore when the guard goes out of scope');
fprintf('  suppress / save / restore ....... ok\n');

% --- restores even when the caller errors ---------------------------------
try
    erroring_guard();
catch
end
assert(strcmp(char(get(groot, 'DefaultFigureVisible')), char(before)), ...
       'must restore after an error');
fprintf('  restores after an error ......... ok\n');

% --- opt-out ---------------------------------------------------------------
setenv('MATLAB_SHOW_FIGURES', '1');
still_visible = check_optout();
setenv('MATLAB_SHOW_FIGURES', '');
assert(still_visible, 'MATLAB_SHOW_FIGURES=1 must not suppress');
fprintf('  MATLAB_SHOW_FIGURES opt-out .... ok\n');

fprintf('ALL TESTS PASSED\n');
end

function ok = use_guard_once()
    g = figures_offscreen(); %#ok<NASGU>
    assert(strcmp(char(get(groot, 'DefaultFigureVisible')), 'off'), 'should suppress');
    f = figure;
    assert(strcmp(f.Visible, 'off'), 'new figure must be hidden');
    plot(1:10);
    tmp = [tempname '.svg'];
    print(f, tmp, '-dsvg');
    d = dir(tmp);
    ok = ~isempty(d) && d.bytes > 0;
    delete(tmp); close(f);
end

function erroring_guard()
    g = figures_offscreen(); %#ok<NASGU>
    error('test:boom', 'boom');
end

function ok = check_optout()
    g = figures_offscreen(); %#ok<NASGU>
    ok = ~strcmp(char(get(groot, 'DefaultFigureVisible')), 'off');
end
