%% === Per-animal means & scatters, keeping trials up to disengagement ===================
n_animals_task = numel(task_data);

% Preallocate (NaNs are fine if an area is missing for an animal)
dms_fr_animal_good = nan(n_animals_task,1);
dls_fr_animal_good = nan(n_animals_task,1);
acc_fr_animal_good = nan(n_animals_task,1);

dms_fr_animal_bad  = nan(n_animals_task,1);
dls_fr_animal_bad  = nan(n_animals_task,1);
acc_fr_animal_bad  = nan(n_animals_task,1);

figure
tiledlayout('flow');

for ianimal = 1:n_animals_task
    td = task_data(ianimal);

    is_dms = td.is_dms;
    is_dls = td.is_dls;
    is_acc = td.is_acc;

    lick_errors_full = td.zscored_lick_errors(:);                 % [trials x 1]
    fr_full = td.z_spatial_binned_fr_all;                           % [neurons x bins x trials]
    ntrials = size(fr_full, 3);

    % ---- Keep only trials up to disengagement (change_point_mean). If NaN -> keep all.
    cpm = NaN;
    if isfield(td, 'change_point_mean'), cpm = td.change_point_mean; end
    if ~isnan(cpm) && isfinite(cpm) && cpm >= 1
        K = min(ntrials, round(cpm));  % include trials 1..K
    else
        K = ntrials;
    end
    keep_idx = 1:K;

    lick_errors_animal = lick_errors_full(keep_idx);
    fr_keep = fr_full(:, :, keep_idx);

    % ---- Good / Bad masks (match your thresholds)
    good_mask = (lick_errors_animal <= -2);
    bad_mask  = (lick_errors_animal >= 0);

    % ---- Per-animal area means for good/bad (avg over neurons & bins & selected trials)
    if any(is_dms)
        dms_fr_animal_good(ianimal) = mean(fr_keep(is_dms, :, good_mask), 'all', 'omitnan');
        dms_fr_animal_bad(ianimal)  = mean(fr_keep(is_dms, :, bad_mask),  'all', 'omitnan');
    end
    if any(is_dls)
        dls_fr_animal_good(ianimal) = mean(fr_keep(is_dls, :, good_mask), 'all', 'omitnan');
        dls_fr_animal_bad(ianimal)  = mean(fr_keep(is_dls, :, bad_mask),  'all', 'omitnan');
    end
    if any(is_acc)
        acc_fr_animal_good(ianimal) = mean(fr_keep(is_acc, :, good_mask), 'all', 'omitnan');
        acc_fr_animal_bad(ianimal)  = mean(fr_keep(is_acc, :, bad_mask),  'all', 'omitnan');
    end

    % ---- Per-animal scatters: per-trial mean FR (avg neurons & bins) vs lick error
    nexttile; hold on;
    legH = []; legN = {};

    if any(is_dms)
        dms_trial_mean = squeeze(mean(fr_keep(is_dms, :, :), [1, 2], 'omitnan')); % [K x 1]
        h1 = scatter(dms_trial_mean, lick_errors_animal, 'filled', ...
                     'MarkerEdgeColor','w', 'MarkerFaceAlpha',0.75);
        lsline; legH(end+1) = h1; legN{end+1} = 'DMS';
    end

    if any(is_acc)
        acc_trial_mean = squeeze(mean(fr_keep(is_acc, :, :), [1, 2], 'omitnan'));
        h2 = scatter(acc_trial_mean, lick_errors_animal, 'filled', ...
                     'MarkerEdgeColor','w', 'MarkerFaceAlpha',0.75);
        lsline; legH(end+1) = h2; legN{end+1} = 'ACC';
    end

    ylim([-5, 5])
    title(sprintf('animal %d (<= trial %d)', ianimal, K))
    xlabel('Per-trial mean FR'); ylabel('z-scored lick error');
    if ~isempty(legH), legend(legH, legN, 'Location','best'); end
    hold off
end

%% === Group errorbar plot ==========
figure
my_errorbar_plot([dms_fr_animal_good, dms_fr_animal_bad, ...
                  dls_fr_animal_good, dls_fr_animal_bad, ...
                  acc_fr_animal_good, acc_fr_animal_bad]);

%% === Heatmap: joint DMS × ACC activity vs performance (lick error) ====================
fprintf('--- Building DMS×ACC heatmap vs lick error (with disengagement filtering) ---\n');

n_animals = numel(task_data);
all_dms = []; all_acc = []; all_lick = [];

for iA = 1:n_animals
    td = task_data(iA);
    if ~isfield(td,'spatial_binned_fr_all') || isempty(td.spatial_binned_fr_all)
        continue
    end
    is_dms = td.is_dms;
    is_acc = td.is_acc;
    if ~any(is_dms) || ~any(is_acc)
        % need both areas for a joint map
        continue
    end

    fr_full   = td.z_spatial_binned_fr_all;          % [neurons x bins x trials]
    lick_full = td.zscored_lick_errors(:);         % [trials x 1]
    ntrials   = size(fr_full, 3);

    % Keep only trials up to disengagement (if defined)
    cpm = NaN;
    if isfield(td,'change_point_mean'), cpm = td.change_point_mean; end
    if ~isnan(cpm) && isfinite(cpm) && cpm >= 1
        K = min(ntrials, round(cpm));
    else
        K = ntrials;
    end
    keep_idx = 1:K;

    % Per-trial mean FR for DMS & ACC (avg over neurons & bins)
    dms_trial = squeeze(mean(fr_full(is_dms, :, keep_idx), [1,2], 'omitmissing'));
    acc_trial = squeeze(mean(fr_full(is_acc, :, keep_idx), [1,2], 'omitmissing'));
    lick_keep = lick_full(keep_idx);

    valid = ~isnan(dms_trial) & ~isnan(acc_trial) & ~isnan(lick_keep);
    all_dms  = [all_dms;  dms_trial(valid)];
    all_acc  = [all_acc;  acc_trial(valid)];
    all_lick = [all_lick; lick_keep(valid)];
end

if isempty(all_lick)
    warning('No valid trials across animals for joint DMS×ACC heatmap.'); 
else
    % Bin edges (include extrema with a tiny epsilon to avoid edge drops)
    nb = 10;  % bins per axis
    epsx = eps(max(abs(all_dms))+1);
    epsy = eps(max(abs(all_acc))+1);
    xedges = linspace(min(all_dms), max(all_dms)+epsx, nb+1);
    yedges = linspace(min(all_acc), max(all_acc)+epsy, nb+1);

    % Bin indices
    [~,~,binX] = histcounts(all_dms, xedges);
    [~,~,binY] = histcounts(all_acc, yedges);

    % Keep valid binned points
    v = binX > 0 & binY > 0 & ~isnan(all_lick);
    sub = [binX(v), binY(v)];

    % Mean lick error per bin (+ counts)
    Msum = accumarray(sub, all_lick(v), [nb nb], @nansum, 0);
    Mcnt = accumarray(sub, 1,           [nb nb], @nansum, 0);
    Mavg = Msum ./ max(Mcnt, 1);             % safe divide
    Mavg(Mcnt==0) = NaN;                     % mark empty bins

    % Plot heatmap
    figure('Name','DMS × ACC joint activity vs lick error','Position',[160 160 760 560]);
    hImg = imagesc(xedges, yedges, Mavg'); axis xy;
    colormap("abyss"); cb = colorbar;
    cb.Label.String = 'Mean z-scored lick error';
    xlabel('DMS mean FR (per trial)'); ylabel('ACC mean FR (per trial)');
    title('Joint DMS × ACC activity vs performance (lower = better)'); box on;

    % Lightly fade sparse bins
    min_count = 2;  % transparency threshold
    alphaMat = ones(size(Mcnt'));
    alphaMat(Mcnt' < min_count) = 0.5;
    set(hImg, 'AlphaData', alphaMat);    

    fprintf('Heatmap built over %d trials across animals (nb=%d, min_count=%d).\n', numel(all_lick), nb, min_count);
end

figure
surf(Mavg)

%% === Per-animal DMS × ACC heatmaps vs performance (tiledlayout) =======================
fprintf('--- Building per-animal DMS×ACC heatmaps (with disengagement filtering) ---\n');

n_animals = numel(task_data);

% ---- First pass: gather global ranges for consistent binning & color scale
all_dms = []; all_acc = []; all_lick = [];
has_both = false(n_animals,1);  % mark animals that have both DMS & ACC and valid trials

for iA = 1:n_animals
    td = task_data(iA);
    if ~isfield(td,'spatial_binned_fr_all') || isempty(td.spatial_binned_fr_all)
        continue
    end
    is_dms = td.is_dms; is_acc = td.is_acc;
    if ~any(is_dms) || ~any(is_acc), continue; end

    fr_full   = td.z_spatial_binned_fr_all;    % [neurons x bins x trials]
    lick_full = td.zscored_lick_errors(:);   % [trials x 1]
    ntrials   = size(fr_full, 3);

    % Keep only trials up to disengagement (if defined)
    cpm = NaN;
    if isfield(td,'change_point_mean'), cpm = td.change_point_mean; end
    if ~isnan(cpm) && isfinite(cpm) && cpm >= 1
        K = min(ntrials, round(cpm));
    else
        K = ntrials;
    end
    keep_idx = 1:K;

    dms_trial = squeeze(mean(fr_full(is_dms, :, keep_idx), [1,2], 'omitmissing'));
    acc_trial = squeeze(mean(fr_full(is_acc, :, keep_idx), [1,2], 'omitmissing'));
    lick_keep = lick_full(keep_idx);

    valid = ~isnan(dms_trial) & ~isnan(acc_trial) & ~isnan(lick_keep);
    if any(valid)
        has_both(iA) = true;
        all_dms  = [all_dms;  dms_trial(valid)];
        all_acc  = [all_acc;  acc_trial(valid)];
        all_lick = [all_lick; lick_keep(valid)];
    end
end

if isempty(all_lick)
    warning('No valid trials across animals for per-animal DMS×ACC heatmaps.');
else
    % ---- Global binning and color limits (consistent across animals)
    nb = 15;  % bins per axis (match your earlier block if desired)
    epsx = eps(max(abs(all_dms))+1);
    epsy = eps(max(abs(all_acc))+1);
    xedges = linspace(min(all_dms), max(all_dms)+epsx, nb+1);
    yedges = linspace(min(all_acc), max(all_acc)+epsy, nb+1);

    % Color limits from pooled lick-error values (robust to outliers if you prefer percentiles)
    % clim = [min(all_lick), max(all_lick)];
    % e.g., robust alternative:
    clim = [prctile(all_lick,1), prctile(all_lick,99)];

    % ---- Figure + layout
    figure('Name','Per-animal DMS × ACC heatmaps vs lick error','Position',[120 120 1100 700]);
    t = tiledlayout('flow','TileSpacing','compact','Padding','compact');
    title(t, 'Per-animal joint DMS × ACC activity vs performance (lower lick error = better)');
    xlabel(t, 'DMS mean FR (per trial)');
    ylabel(t, 'ACC mean FR (per trial)');

    n_plotted = 0;
    min_count = 2; % fade bins with < min_count samples

    for iA = 1:n_animals
        if ~has_both(iA), continue; end

        td = task_data(iA);
        is_dms = td.is_dms; is_acc = td.is_acc;

        fr_full   = td.z_spatial_binned_fr_all;
        lick_full = td.zscored_lick_errors(:);
        ntrials   = size(fr_full, 3);

        % Disengagement filtering
        cpm = NaN;
        if isfield(td,'change_point_mean'), cpm = td.change_point_mean; end
        if ~isnan(cpm) && isfinite(cpm) && cpm >= 1
            K = min(ntrials, round(cpm));
        else
            K = ntrials;
        end
        keep_idx = 1:K;

        % Per-trial means
        dms_trial = squeeze(mean(fr_full(is_dms, :, keep_idx), [1,2], 'omitmissing'));
        acc_trial = squeeze(mean(fr_full(is_acc, :, keep_idx), [1,2], 'omitmissing'));
        lick_keep = lick_full(keep_idx);

        valid = ~isnan(dms_trial) & ~isnan(acc_trial) & ~isnan(lick_keep);
        if ~any(valid), continue; end

        % Bin this animal into the global edges
        [~,~,binX] = histcounts(dms_trial(valid), xedges);
        [~,~,binY] = histcounts(acc_trial(valid), yedges);
        vv = binX>0 & binY>0;
        if ~any(vv), continue; end

        sub  = [binX(vv), binY(vv)];
        vals = lick_keep(valid); vals = vals(vv);

        Msum = accumarray(sub, vals, [nb nb], @nansum, 0);
        Mcnt = accumarray(sub, 1,    [nb nb], @nansum, 0);
        Mavg = Msum ./ max(Mcnt, 1);
        Mavg(Mcnt==0) = NaN;

        % Plot tile
        ax = nexttile; 
        hImg = imagesc(xedges, yedges, Mavg'); axis xy;
        colormap(ax, "hot"); caxis(ax, clim);
        title(sprintf('Animal %d (<= trial %d), n=%d', iA, K, sum(valid)));
        xlim([xedges(1), xedges(end)]); ylim([yedges(1), yedges(end)]);
        box on;

        % Fade sparse bins
        alphaMat = ones(size(Mcnt'));
        alphaMat(Mcnt' < min_count) = 0.5;
        set(hImg, 'AlphaData', alphaMat);

        n_plotted = n_plotted + 1;
    end

    % One shared colorbar
    cb = colorbar('eastoutside');
    cb.Label.String = 'Mean z-scored lick error';
    fprintf('Plotted %d animal heatmaps (nb=%d, min_count=%d).\n', n_plotted, nb, min_count);
end