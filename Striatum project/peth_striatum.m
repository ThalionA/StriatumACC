%% PETH ANALYSIS SCRIPT
% This script aligns neural spiking data to behavioral events (licks, rewards),
% normalizes the data using z-scoring, and generates comprehensive
% visualizations including sorted rastermaps and mean PETH plots with SEM.

% Let's assume 'task_data' is already loaded in the workspace.
% load('your_task_data.mat');

%% PARAMETERS 
binWidth      = 0.001;          % seconds per spike bin (1 ms)
periWindow    = [-1 1];         % seconds around event  (-1 s … +1 s)
baselineWin   = [-1 -0.5];      % seconds, for z-scoring
responseWin   = [0.05 0.5];     % seconds, for sorting trials

% --- Convert windows from seconds to bin indices ---
winBins       = round(periWindow / binWidth);
baselineBins  = round(baselineWin / binWidth);
responseBins  = round(responseWin / binWidth);
timeAxis      = (winBins(1):winBins(2)) * binWidth;

% --- Create logical masks for indexing time windows ---
baselineLogic = timeAxis >= baselineWin(1) & timeAxis <= baselineWin(2);
responseLogic = timeAxis >= responseWin(1) & timeAxis <= responseWin(2);

%% ALIGNMENT & AGGREGATION
ianimal = 3; % Example animal

% --- First, loop through all trials to find the total number of events ---
% This allows us to pre-allocate memory for efficiency.
totalLicks = 0;
totalRewards = 0;
numTrials = numel(task_data(ianimal).corridorData.trial_times);
for itr = 1:numTrials
    totalLicks = totalLicks + sum(task_data(ianimal).corridorData.trial_licks{itr});
    totalRewards = totalRewards + sum(task_data(ianimal).corridorData.trial_reward{itr} > 0);
end

% --- Pre-allocate matrices to hold all events from all trials ---
nUnits = size(task_data(ianimal).corridorData.binned_spikes{1}, 1);
nTimeBins = numel(timeAxis);

allLickSpikes   = nan(nUnits, nTimeBins, totalLicks);
allRewardSpikes = nan(nUnits, nTimeBins, totalRewards);

% --- Loop through trials again to extract and fill the matrices ---
lickCounter = 0;
rewardCounter = 0;
for itr = 1:numTrials
    % Shorthand handles
    behTime   = task_data(ianimal).corridorData.trial_times{itr};    % msec
    licks     = task_data(ianimal).corridorData.trial_licks{itr};    % logical
    reward    = task_data(ianimal).corridorData.trial_reward{itr};   % logical
    spikesBin = task_data(ianimal).corridorData.binned_spikes{itr};  % units × bins
    
    % Behavioural timestamps (relative to trial start)
    lickTimes   = behTime(logical(licks));
    rewTime     = behTime(find(reward, 1, 'first'));
    
    % Convert to bin indices (MATLAB 1-based)
    lickIdx   = round(lickTimes) + 1;
    rewIdx    = round(rewTime) + 1;
    
    % Build peri-event matrix for licks in this trial
    if ~isempty(lickIdx)
        periLickTrial = buildPeriMatrix(spikesBin, lickIdx, winBins);
        nLicksInTrial = size(periLickTrial, 3);
        allLickSpikes(:, :, lickCounter + (1:nLicksInTrial)) = periLickTrial;
        lickCounter = lickCounter + nLicksInTrial;
    end
    
    % Build peri-event matrix for reward in this trial
    if ~isempty(rewIdx)
        periRewardTrial = buildPeriMatrix(spikesBin, rewIdx, winBins);
        allRewardSpikes(:, :, rewardCounter + 1) = periRewardTrial;
        rewardCounter = rewardCounter + 1;
    end
end

% Convert spike counts to firing rate (spikes/sec)
firingRateLick   = allLickSpikes / binWidth;
firingRateReward = allRewardSpikes / binWidth;

%% Z-SCORE NORMALIZATION
% Z-scoring expresses the activity in terms of standard deviations from a
% baseline mean, making responses comparable across neurons.

% --- Licks ---
% Calculate mean and std during the baseline period for each unit
meanBaselineLick = mean(firingRateLick(:, baselineLogic, :), 2, 'omitnan');
stdBaselineLick  = std(firingRateLick(:, baselineLogic, :), 0, 2, 'omitnan');
% Avoid division by zero for silent neurons
stdBaselineLick(stdBaselineLick == 0) = 1; 
% Z-score calculation (using broadcasting)
zLick = (firingRateLick - meanBaselineLick) ./ stdBaselineLick;

% --- Rewards ---
meanBaselineRew = mean(firingRateReward(:, baselineLogic, :), 2, 'omitnan');
stdBaselineRew  = std(firingRateReward(:, baselineLogic, :), 0, 2, 'omitnan');
stdBaselineRew(stdBaselineRew == 0) = 1;
zReward = (firingRateReward - meanBaselineRew) ./ stdBaselineRew;

%% VISUALISE (Unit by Unit)
for u = 1:nUnits
    % --- Prepare data for this unit ---
    unitLickData   = squeeze(zLick(u, :, :));   % time x events
    unitRewardData = squeeze(zReward(u, :, :)); % time x events
    
    % --- Sort trials for plotting ---
    % Sorting by the mean activity in the response window can reveal patterns.
    [~, lickSortIdx] = sort(mean(unitLickData(responseLogic, :), 1, 'omitnan'));
    [~, rewardSortIdx] = sort(mean(unitRewardData(responseLogic, :), 1, 'omitnan'));
    
    sortedLickData = unitLickData(:, lickSortIdx);
    sortedRewardData = unitRewardData(:, rewardSortIdx);
    
    % --- Calculate Mean and SEM for PETH plots ---
    meanLick   = mean(unitLickData, 2, 'omitnan');
    semLick    = std(unitLickData, 0, 2, 'omitnan') / sqrt(size(unitLickData, 2));
    
    meanReward = mean(unitRewardData, 2, 'omitnan');
    semReward  = std(unitRewardData, 0, 2, 'omitnan') / sqrt(size(unitRewardData, 2));
    
    % --- Create the Figure ---
    figure('Position', [100, 100, 1200, 800], 'NumberTitle', 'off', 'Name', sprintf('Animal %d - Unit %d', ianimal, u));
    
    % 1. Lick-aligned Rastermap (Heatmap)
    ax1 = subplot(2, 2, 1);
    imagesc(timeAxis, 1:size(sortedLickData, 2), sortedLickData');
    title('Lick-Aligned (Sorted)');
    ylabel('Lick Event #');
    xline(0, '--k', 'LineWidth', 1.5); % Mark event time
    colormap(ax1, bluewhitered);
    
    % 2. Reward-aligned Rastermap (Heatmap)
    ax2 = subplot(2, 2, 2);
    imagesc(timeAxis, 1:size(sortedRewardData, 2), sortedRewardData');
    title('Reward-Aligned (Sorted)');
    ylabel('Trial #');
    xline(0, '--k', 'LineWidth', 1.5);
    colormap(ax2, bluewhitered);
    hcb = colorbar;
    hcb.Label.String = 'Z-Scored Firing Rate';
    
    % 3. Lick-aligned Mean PETH
    subplot(2, 2, 3);
    hold on;
    % Use 'boundedline' function if available, otherwise plot patch manually
    fill([timeAxis, fliplr(timeAxis)], [meanLick' - semLick', fliplr(meanLick' + semLick')], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(timeAxis, meanLick, 'r', 'LineWidth', 2);
    hold off;
    box off; grid on;
    xline(0, '--k', 'LineWidth', 1.5);
    xlabel('Time from Lick (s)');
    ylabel('Z-Scored Firing Rate');
    xlim(periWindow);
    
    % 4. Reward-aligned Mean PETH
    subplot(2, 2, 4);
    hold on;
    fill([timeAxis, fliplr(timeAxis)], [meanReward' - semReward', fliplr(meanReward' + semReward')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(timeAxis, meanReward, 'b', 'LineWidth', 2);
    hold off;
    box off; grid on;
    xline(0, '--k', 'LineWidth', 1.5);
    xlabel('Time from Reward (s)');
    ylabel('Z-Scored Firing Rate');
    xlim(periWindow);
    
    % Add a main title to the figure
    sgtitle(sprintf('Animal %d – Unit %d', ianimal, u));
    
    fprintf('Displaying Unit %d of %d. Press any key to continue...\n', u, nUnits);
    pause; % Wait for user to press a key
    if ishandle(gcf); close(gcf); end % Close the figure before showing the next
end

%% HELPER FUNCTION
function peri = buildPeriMatrix(spk, evtIdx, winBins)
% Extracts peri-event spike matrices.
% spk      : units × timeBins
% evtIdx   : vector of event indices (1-based)
% winBins  : [neg, pos] peri-window in BINS, e.g. [-1000 1000]
    win     = winBins(1):winBins(2);
    nUnits  = size(spk,1);
    nEvt    = numel(evtIdx);
    nBins   = numel(win);
    peri    = nan(nUnits, nBins, nEvt);
    for e = 1:nEvt
        % Calculate indices for the window around the current event
        idx = evtIdx(e) + win;
        % Find which indices are valid (i.e., within the trial bounds)
        good_idx = idx > 0 & idx <= size(spk, 2);
        
        % Place the valid spike data into the correct columns of the peri-event matrix
        if any(good_idx)
            peri(:, good_idx, e) = spk(:, idx(good_idx));
        end
    end
end

% A good diverging colormap function (if you don't have one)
function c = bluewhitered(m)
    if nargin < 1, m = size(get(gcf,'colormap'),1); end
    b = [0 0 1];
    w = [1 1 1];
    r = [1 0 0];
    % Interpolate from blue to white, then white to red
    n = floor(m/2);
    c1 = interp1([1 n], [b; w], 1:n);
    c2 = interp1([1 m-n+1], [w; r], 1:m-n+1);
    c = [c1; c2];
end
