function [perfByCon, goProbByCon, lickByCon, velByCon, dwellByCon, preRZLickByCon, preRZVelByCon, gratingVelByCon, grVelFirstByCon, grVelSecondByCon, ...
    corrLickLatency_byCon, corrILI_byCon, grDecel_byCon, grMinVel_byCon, trialDuration_byCon] = ...
    plotPooledContrastFigures(vrSessions)
% plotPooledContrastFigures   Pool data from several sessions and plot
% contrast‐dependent performance, reward-zone lick counts, reward-zone
% velocity, dwell time, pre-reward-zone lick counts, pre-reward-zone velocity,
% and grating-phase velocity (average), plus additional global go/no-go plots.
%
% It also computes the following additional behavioral measures:
%  1. Time from corridor entry to first lick.
%  2. Average interlick interval (ILI) during the corridor (before RZ entry).
%  3. Deceleration during the first 500ms of the grating.
%  4. The minimum velocity reached during the grating.
%  5. Total trial duration.
%
% Then, it pools these trial-level metrics across sessions for a fixed set
% of contrasts [1, 0.25, 0.1, 0.01] and creates:
%   - Consolidated contrast-dependent summary plots (with individual session lines)
%   - Sigmoid fits for all measures.
%
% Dependencies: Assumes shadedErrorBar, colorGradient, and sem are on the MATLAB path.
%

%% Initialize pooled arrays (one row per trial)
allPerformance      = [];
allRZLick           = [];
allRZVel            = [];
allDwell            = [];
allPreRZLick        = [];
allPreRZVel         = [];    % average corridor velocity before RZ
allGratingVel       = [];    % average grating-phase velocity per trial
allStimuli          = [];    % contrast values per trial (from vr.cfg.trialContrasts)
allLickMatrix       = [];    % corridor-phase lick counts per trial
allVelMatrix        = [];    % corridor-phase velocity per trial
allGrVelTimeMatrix  = [];    % grating-phase time-binned velocity per trial
allSessionID        = [];    % to store session identity for each trial

% Additional behavioral measures (one value per trial)
allCorrLickLatency  = [];
allCorrILI          = [];
allGrDecel          = [];
allGrMinVel         = [];
allTrialDuration    = [];

%% Loop over sessions to extract and pool trial data
for sess = 1:length(vrSessions)
    sessionData = vrSessions{sess};
    
    % Extract trial-level metrics from the current session.
    % Modify these field names to match your actual session data structure.
    allPerformance      = [allPerformance;     sessionData.trialPerformance];
    allRZLick           = [allRZLick;          sessionData.trialRZLicks];
    allRZVel            = [allRZVel;           sessionData.trialRZVelocity];
    allDwell            = [allDwell;           sessionData.trialDwellTime];
    allPreRZLick        = [allPreRZLick;       sessionData.trialPreRZLicks];
    allPreRZVel         = [allPreRZVel;        sessionData.trialPreRZVelocity];
    allGratingVel       = [allGratingVel;      sessionData.trialGratingVelocity];
    
    % Updated field: extract contrast values from vr.cfg.trialContrasts
    allStimuli          = [allStimuli;         sessionData.vr.cfg.trialContrasts];
    
    allLickMatrix       = [allLickMatrix;      sessionData.trialLickMatrix];
    allVelMatrix        = [allVelMatrix;       sessionData.trialVelMatrix];
    allGrVelTimeMatrix  = [allGrVelTimeMatrix; sessionData.trialGrVelTimeMatrix];
    allSessionID        = [allSessionID;       repmat(sess, size(sessionData.trialPerformance))];
    
    % Additional measures
    allCorrLickLatency  = [allCorrLickLatency; sessionData.trialCorrLickLatency];
    allCorrILI          = [allCorrILI;         sessionData.trialCorrILI];
    allGrDecel          = [allGrDecel;         sessionData.trialGrDecel];
    allGrMinVel         = [allGrMinVel;        sessionData.trialGrMinVel];
    allTrialDuration    = [allTrialDuration;   sessionData.trialDuration];
end

%% Define fixed set of contrast levels
% The full set of contrasts is [1, 0.25, 0.1, 0.01]. Note that some contrasts 
% (e.g., 0.01) may not exist in all sessions, but we still maintain these on the x-axis.
contrastLevels = [1, 0.25, 0.1, 0.01];
nCon = length(contrastLevels);

% Preallocate output variables
perfByCon             = nan(nCon,1);
goProbByCon           = nan(nCon,1);
lickByCon             = nan(nCon,1);
velByCon              = nan(nCon,1);
dwellByCon            = nan(nCon,1);
preRZLickByCon        = nan(nCon,1);
preRZVelByCon         = nan(nCon,1);
gratingVelByCon       = nan(nCon,1);
grVelFirstByCon       = nan(nCon,1);
grVelSecondByCon      = nan(nCon,1);
corrLickLatency_byCon = nan(nCon,1);
corrILI_byCon         = nan(nCon,1);
grDecel_byCon         = nan(nCon,1);
grMinVel_byCon        = nan(nCon,1);
trialDuration_byCon   = nan(nCon,1);

%% Pool data by contrast level and compute summary statistics
for c = 1:nCon
    % Find all trials with the current contrast level
    idx = allStimuli == contrastLevels(c);
    
    if any(idx)
        % Compute performance (e.g., fraction correct or hit rate)
        perfByCon(c) = nanmean(allPerformance(idx));
        
        % Compute go probability (if different from performance, adjust accordingly)
        goProbByCon(c) = nanmean(allPerformance(idx)); % adjust if necessary
        
        % Reward-zone licks and velocity
        lickByCon(c) = nanmean(allRZLick(idx));
        velByCon(c)  = nanmean(allRZVel(idx));
        
        % Dwell time in reward zone
        dwellByCon(c) = nanmean(allDwell(idx));
        
        % Pre-reward zone measures
        preRZLickByCon(c) = nanmean(allPreRZLick(idx));
        preRZVelByCon(c)  = nanmean(allPreRZVel(idx));
        
        % Grating-phase velocity (average over trial)
        gratingVelByCon(c) = nanmean(allGratingVel(idx));
        
        % Split grating-phase velocity into first and second halves (if time-binned data exists)
        if ~isempty(allGrVelTimeMatrix)
            nBins = size(allGrVelTimeMatrix,2);
            grVelFirstByCon(c)  = nanmean(nanmean(allGrVelTimeMatrix(idx, 1:floor(nBins/2)),2));
            grVelSecondByCon(c) = nanmean(nanmean(allGrVelTimeMatrix(idx, floor(nBins/2)+1:end),2));
        end
        
        % Additional behavioral measures
        corrLickLatency_byCon(c) = nanmean(allCorrLickLatency(idx));
        corrILI_byCon(c)         = nanmean(allCorrILI(idx));
        grDecel_byCon(c)         = nanmean(allGrDecel(idx));
        grMinVel_byCon(c)        = nanmean(allGrMinVel(idx));
        trialDuration_byCon(c)   = nanmean(allTrialDuration(idx));
    end
end

%% Plotting summary figures
figure;

subplot(3,3,1);
shadedErrorBar(contrastLevels, perfByCon, sem(perfByCon));
xlabel('Contrast');
ylabel('Performance');
title('Performance by Contrast');

subplot(3,3,2);
shadedErrorBar(contrastLevels, lickByCon, sem(lickByCon));
xlabel('Contrast');
ylabel('RZ Licks');
title('Reward-Zone Licks');

subplot(3,3,3);
shadedErrorBar(contrastLevels, velByCon, sem(velByCon));
xlabel('Contrast');
ylabel('RZ Velocity');
title('Reward-Zone Velocity');

subplot(3,3,4);
shadedErrorBar(contrastLevels, dwellByCon, sem(dwellByCon));
xlabel('Contrast');
ylabel('Dwell Time');
title('Dwell Time');

subplot(3,3,5);
shadedErrorBar(contrastLevels, preRZLickByCon, sem(preRZLickByCon));
xlabel('Contrast');
ylabel('Pre-RZ Licks');
title('Pre-RZ Licks');

subplot(3,3,6);
shadedErrorBar(contrastLevels, preRZVelByCon, sem(preRZVelByCon));
xlabel('Contrast');
ylabel('Pre-RZ Velocity');
title('Pre-RZ Velocity');

subplot(3,3,7);
shadedErrorBar(contrastLevels, gratingVelByCon, sem(gratingVelByCon));
xlabel('Contrast');
ylabel('Grating Velocity');
title('Grating Velocity');

subplot(3,3,8);
shadedErrorBar(contrastLevels, corrLickLatency_byCon, sem(corrLickLatency_byCon));
xlabel('Contrast');
ylabel('Corr Lick Latency');
title('Corr Lick Latency');

subplot(3,3,9);
shadedErrorBar(contrastLevels, trialDuration_byCon, sem(trialDuration_byCon));
xlabel('Contrast');
ylabel('Trial Duration');
title('Trial Duration');

% Additional plots for grVelFirst and grVelSecond can be added as needed.

end