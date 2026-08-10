function test_assign_areas_by_depth()
%TEST_ASSIGN_AREAS_BY_DEPTH  Synthetic ground-truth checks for the
%   depth->area labeller shared by the Organise*IncV1 scripts.
%
%   Run from the project root:  test_assign_areas_by_depth
%
%   These cases encode the exact semantics the depth-boundary audit relied on:
%   inclusive bands, gaps between bands, DLS-overwrites-DMS precedence, the
%   NaN-edge (empty CSV cell) skip, and the specific 731 correction.

    % --- 1. Basic banding + gaps ---
    depths = [50; 150; 250; 450; 650; 850];
    names  = {'DMS', 'DLS', 'ACC'};
    starts = [500,   0,     2000];
    ends   = [800,   300,   2500];
    labels = assign_areas_by_depth(depths, names, starts, ends);
    assert(strcmp(labels{1}, 'DLS'), '50um should be DLS (0-300)');
    assert(strcmp(labels{2}, 'DLS'), '150um should be DLS');
    assert(strcmp(labels{3}, 'DLS'), '250um should be DLS');
    assert(isempty(labels{4}),       '450um is in a gap (300<d<500) -> no label');
    assert(strcmp(labels{5}, 'DMS'), '650um should be DMS (500-800)');
    assert(isempty(labels{6}),       '850um is above DMS, below ACC -> no label');

    % --- 2. Overlap precedence: later area overwrites earlier ---
    lab = assign_areas_by_depth([100; 100], {'DMS', 'DLS'}, [0, 0], [200, 200]);
    assert(strcmp(lab{1}, 'DLS'), 'DLS is applied after DMS -> must win on overlap');
    assert(strcmp(lab{2}, 'DLS'));

    % --- 3. NaN band assigns nothing (empty CSV cell -> NaN) ---
    lab = assign_areas_by_depth([100; 100], {'DLS'}, NaN, NaN);
    assert(isempty(lab{1}), 'NaN edge -> area assigns nothing');
    assert(isempty(lab{2}));

    % --- 4. Inclusive boundaries ---
    lab = assign_areas_by_depth([500; 800], {'DMS'}, 500, 800);
    assert(strcmp(lab{1}, 'DMS'), 'lower edge inclusive');
    assert(strcmp(lab{2}, 'DMS'), 'upper edge inclusive');

    % --- 5. Reproduces the 731 fix (DMS 500-800, no DLS) ---
    %   A unit at 250um (the OLD wrong DMS 0-300 band) must now be unlabelled;
    %   a unit at 650um must be DMS.
    lab = assign_areas_by_depth([250; 650], {'DMS', 'DLS', 'ACC'}, ...
                                [500, NaN, 2300], [800, NaN, 3000]);
    assert(isempty(lab{1}), '250um no longer DMS after the 731 fix');
    assert(strcmp(lab{2}, 'DMS'), '650um is DMS after the 731 fix');

    fprintf('test_assign_areas_by_depth: ALL PASSED\n');
end
