%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ALIGN STRIATAL SPIKES TO TEMPORAL EVENTS AND VISUALISE FIRING RATE
% ------------------------------------------------------------------
% * Aligns spikes to lick, reward, acceleration, & deceleration events
% * Converts to firing rate (50 ms bins, Gaussian-smoothed)
% * Plots heat-maps of rate for every trial and unit
% * Plots trial-averaged PETH with SEM, split by engagement state
% * OPTIMISED for memory efficiency to prevent crashes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PARAMETERS you may want to tweak
binWidth      = 0.001;        % seconds per spike bin  (1 ms)
periWindow    = [-1 1];       % seconds around event
rateBin       = 0.050;        % seconds per rate bin   (50 ms)
smoothSigma   = 0.025;        % Gaussian SD for smoothing firing rate (25 ms)

% --- Kinematic event detection parameters ---
speedSmoothSigma    = 0.100;      % Gaussian SD for smoothing speed (100 ms)
accelThresh         = 5;          % Acceleration threshold (cm/s^2)
decelThresh         = -5;         % Deceleration threshold (cm/s^2)

% --- Derived parameters ---
winBins       = round(periWindow/binWidth);
timeAxisFull  = (winBins(1):winBins(2))*binWidth;
rateSamples   = round(rateBin/binWidth);
smoothBins    = round(smoothSigma/binWidth);
rateIdx       = 1:rateSamples:numel(timeAxisFull);
rateTimeAxis  = timeAxisFull(rateIdx);

% --- Data selection ---
ianimal       = 2;
numTrials     = numel(task_data(ianimal).corridorData.trial_times);
changePoint   = task_data(ianimal).change_point_mean; % Trial index for disengagement

%% DATA PROCESSING (MEMORY-EFFICIENT)
nUnits    = size(task_data(ianimal).corridorData.binned_spikes{1}, 1);
nRateBins = numel(rateIdx);

% Pre-allocate final output matrices. This is key for efficiency.
rateLickMat   = nan(nUnits, nRateBins, numTrials);
rateRewardMat = nan(nUnits, nRateBins, numTrials);
rateAccelMat  = nan(nUnits, nRateBins, numTrials);
rateDecelMat  = nan(nUnits, nRateBins, numTrials);

% --- Process data trial-by-trial to conserve memory ---
for itr = 1:numTrials
    % shorthand handles
    behTime   = task_data(ianimal).corridorData.trial_times{itr};   % ms
    position  = task_data(ianimal).corridorData.trial_position{itr};% position data
    licks     = task_data(ianimal).corridorData.trial_licks{itr};   % logical
    reward    = task_data(ianimal).corridorData.trial_reward{itr};  % logical
    spikesBin = task_data(ianimal).corridorData.binned_spikes{itr}; % units × bins

    % --- 1. ALIGN: Find event indices for this trial ---
    lickTimes = behTime(logical(licks));
    rewTime   = behTime(find(reward,1,'first'));
    lickIdx = round(lickTimes) + 1;
    rewIdx  = round(rewTime)  + 1;
    
    accelIdx = [];
    decelIdx = [];
    if numel(behTime) > 1
        speedSmoothBins = round(speedSmoothSigma / binWidth);
        rawSpeed = [0; diff(position(:)) ./ (diff(behTime(:))/1000)]; 
        smoothedSpeed = smoothdata(rawSpeed, 'gaussian', speedSmoothBins*6);
        acceleration = [0; diff(smoothedSpeed) / binWidth];
        accelIdx = find(acceleration > accelThresh);
        decelIdx = find(acceleration < decelThresh);
    end
    
    % --- 2. PROCESS: Build peri-event matrix, average, smooth, and bin ---
    % This is done for each event type, one at a time. The large intermediate
    % 'peri_...' matrix is discarded after each step, saving memory.
    
    % Licks
    peri_lick = buildPeriMatrix_vec(spikesBin, lickIdx, winBins);
    meanLick = mean(peri_lick, 3, 'omitnan');
    meanLickSm = smoothdata(meanLick, 2, 'gaussian', 6*smoothBins+1, 'omitnan');
    rateLickMat(:,:,itr) = meanLickSm(:, rateIdx) / rateBin;

    % Reward
    peri_reward = buildPeriMatrix_vec(spikesBin, rewIdx, winBins);
    meanReward = mean(peri_reward, 3, 'omitnan');
    meanRewardSm = smoothdata(meanReward, 2, 'gaussian', 6*smoothBins+1, 'omitnan');
    rateRewardMat(:,:,itr) = meanRewardSm(:, rateIdx) / rateBin;
    
    % Acceleration
    peri_accel = buildPeriMatrix_vec(spikesBin, accelIdx, winBins);
    meanAccel = mean(peri_accel, 3, 'omitnan');
    meanAccelSm = smoothdata(meanAccel, 2, 'gaussian', 6*smoothBins+1, 'omitnan');
    rateAccelMat(:,:,itr) = meanAccelSm(:, rateIdx) / rateBin;

    % Deceleration
    peri_decel = buildPeriMatrix_vec(spikesBin, decelIdx, winBins);
    meanDecel = mean(peri_decel, 3, 'omitnan');
    meanDecelSm = smoothdata(meanDecel, 2, 'gaussian', 6*smoothBins+1, 'omitnan');
    rateDecelMat(:,:,itr) = meanDecelSm(:, rateIdx) / rateBin;
end

%% VISUALISATION ----------------------------------------------------------
hasChangePoint = ~isempty(changePoint) && ~isnan(changePoint) && changePoint > 1 && changePoint < numTrials;

for u = 1:nUnits
    figure('Position', [50 50 1800 800]);
        
    % --- Heatmaps (Top Row) ---
    subplot(2,4,1); imagesc(rateTimeAxis, 1:numTrials, squeeze(rateLickMat(u,:,:))');
    title('Lick'); ylabel('Trial');
    subplot(2,4,2); imagesc(rateTimeAxis, 1:numTrials, squeeze(rateRewardMat(u,:,:))');
    title('Reward');
    subplot(2,4,3); imagesc(rateTimeAxis, 1:numTrials, squeeze(rateAccelMat(u,:,:))');
    title('Acceleration');
    subplot(2,4,4); imagesc(rateTimeAxis, 1:numTrials, squeeze(rateDecelMat(u,:,:))');
    title('Deceleration');
    
    % Apply common formatting to all heatmaps
    for i = 1:4
        subplot(2,4,i);
        colormap(gca, 'turbo');
        xline(0, '--k');
        if hasChangePoint; yline(changePoint-0.5, '--w', 'LineWidth', 1.5); end
    end
    cb = colorbar; cb.Label.String = 'Firing Rate (spikes/s)';
    cb.Position = [0.93 0.58 0.015 0.34];

    % --- PETH Plots (Bottom Row) ---
    pethAxes = [subplot(2,4,5), subplot(2,4,6), subplot(2,4,7), subplot(2,4,8)];
    allUnitData = {squeeze(rateLickMat(u,:,:))', squeeze(rateRewardMat(u,:,:))', ...
                   squeeze(rateAccelMat(u,:,:))', squeeze(rateDecelMat(u,:,:))'};
    xlabels = {'Time from Lick (s)', 'Time from Reward (s)', ...
               'Time from Accel. (s)', 'Time from Decel. (s)'};
    
    for i = 1:4
        axes(pethAxes(i));
        unitData = allUnitData{i};
        
        if hasChangePoint
            dataBefore = unitData(1:changePoint-1, :);
            meanBefore = mean(dataBefore, 1, 'omitnan');
            semBefore  = std(dataBefore, 0, 1, 'omitnan') / sqrt(sum(~isnan(dataBefore(:,1))));
            
            dataAfter = unitData(changePoint:end, :);
            meanAfter = mean(dataAfter, 1, 'omitnan');
            semAfter  = std(dataAfter, 0, 1, 'omitnan') / sqrt(sum(~isnan(dataAfter(:,1))));
            
            hold on;
            h1 = shadedErrorBar(rateTimeAxis, meanBefore, semBefore, 'lineprops', {'-g', 'LineWidth', 1.5});
            h2 = shadedErrorBar(rateTimeAxis, meanAfter, semAfter, 'lineprops', {'-m', 'LineWidth', 1.5});
            hold off;
            if i == 1; legend([h1.mainLine, h2.mainLine], {'Engaged', 'Disengaged'}, 'Location', 'northwest', 'Box', 'off'); end
        else
            meanAll = mean(unitData, 1, 'omitnan');
            semAll  = std(unitData, 0, 1, 'omitnan') / sqrt(sum(~isnan(unitData(:,1))));
            hold on;
            h = shadedErrorBar(rateTimeAxis, meanAll, semAll, 'lineprops', {'-g', 'LineWidth', 1.5});
            hold off;
            if i == 1; legend(h.mainLine, {'Engaged'}, 'Location', 'northwest', 'Box', 'off'); end
        end
        
        box off; xlim(periWindow); xline(0, '--k');
        xlabel(xlabels{i});
        if i == 1; ylabel('Avg. Firing Rate (spikes/s)'); end
    end

    % --- Add overall title ---
    sgtitle(sprintf('Animal %d – Unit %d (Rate: 50 ms bins, Smooth: σ = 25 ms)', ianimal, u));
    
    fprintf('Displaying Unit %d of %d. Press any key to continue...\n', u, nUnits);
    pause; % hit any key to step through units
    if ishandle(gcf); close(gcf); end
end

%% HELPER FUNCTION --------------------------------------------------------
function peri = buildPeriMatrix_vec(spk, evtIdx, winBins)
% Build peri-event spike block using a more vectorized approach.
% This version avoids looping over every event, which is faster.
% spk     : units × timeBins
% evtIdx  : vector of 1-based event indices (can be empty or NaN)
% winBins : [neg pos] peri window in BINS (e.g. [-1000 1000])
    win   = winBins(1):winBins(2);
    nUnit = size(spk,1);
    nEvt  = numel(evtIdx);
    nBins = numel(win);
    
    % If there are no events, return an empty matrix with the correct dimensions
    if nEvt == 0
        peri = nan(nUnit, nBins, 0, 'like', spk);
        return;
    end
    
    % Create a matrix of all time indices to be extracted.
    % Each row corresponds to an event, each column to a time bin in the window.
    % Resulting size: [nEvt, nBins]
    idx_matrix = evtIdx(:) + win; % Using broadcasting
    
    % Create a mask for indices that are within the bounds of the spike data
    good_mask = idx_matrix > 0 & idx_matrix <= size(spk, 2);
    
    % Pre-allocate the final output matrix with NaNs
    peri = nan(nUnit, nBins, nEvt, 'like', spk);
    
    % Loop over units (typically far fewer than events, so this is efficient)
    for u = 1:nUnit
        spk_unit = spk(u,:);
        % Create a temporary matrix to hold extracted data for this unit
        temp_peri_unit = nan(nEvt, nBins);
        % Use the good_mask to index both the source (spk_unit) and destination (temp_peri_unit).
        % This extracts all valid spike bins for all events for the current unit at once.
        temp_peri_unit(good_mask) = spk_unit(idx_matrix(good_mask));
        % Assign the transposed result to the final output matrix
        peri(u,:,:) = temp_peri_unit'; % Transpose to get [nBins, nEvt]
    end
end
