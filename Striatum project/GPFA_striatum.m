%% Load data
load("preprocessed_data.mat")

%% Structure appropriately
ianimal = 3;

data = task_data(ianimal).corridorData.binned_spikes;

% Assuming your data variable is named 'data' and is of size:
% [numNeurons x numSpatialBins x numTrials]
numTrials = size(data, 2);

% Create a structure array 'dat' for GPFA input.
dat = struct('trialId', cell(1, numTrials), 'spikes', cell(1, numTrials));

for trial = 1:numTrials
    dat(trial).trialId = trial;
    % If your data is already binary, you can directly assign it.
    % Otherwise, threshold or convert your data to 0/1 as needed.
    dat(trial).spikes = squeeze(data{trial});
end

%% Run GPFA

addpath(genpath('/Users/theoamvr/Desktop/Experiments/gpfa_v0203'))
runIdx = 10;
xDim = 8;
binWidth = 100;  % in msec

% Run the neural trajectory extraction.
result = neuralTraj(runIdx, dat(1:70), 'xDim', xDim, 'binWidth', binWidth);


%% Plot

% Load the results file if not already loaded (adjust the filename as necessary)
res = load('mat_results/run010/gpfa_xDim08.mat');  

% Assume res.xsm is a cell array with one entry per trial.
% trialToPlot = 2;  % change to select a different trial
  % latentTraj is of size [xDim x numBins]
nTrials = numel(res.seqTrain);
dimtoplot = 3;

figure
hold on
for iTrial = 1:nTrials
    latentTraj = res.seqTrain(iTrial).xsm;
    if dimtoplot >= 3
        % Plot in 3D if at least 3 latent dimensions are available.
        if iTrial < 11
            plot3(latentTraj(1,:), latentTraj(2,:), latentTraj(3,:), 'b');
        else
            plot3(latentTraj(1,:), latentTraj(2,:), latentTraj(3,:), 'r');
        end
        xlabel('Latent 1');
        ylabel('Latent 2');
        zlabel('Latent 3');
    else
        % For 2D, simply plot latent 1 vs. latent 2.
        plot(latentTraj(1,:), latentTraj(2,:));
        xlabel('Latent 1');
        ylabel('Latent 2');
    end

end


% figure;
% hold on;
% colormap(jet); % Choose a colormap (you can also try 'parula', 'hot', etc.)
% colorbar;      % Display a colorbar to indicate time progression
% 
% for iTrial = 1:nTrials
%     latentTraj = res.seqTrain(iTrial).xsm;  % [xDim x numBins]
%     T = size(latentTraj,2);
%     % Create a time vector, normalized if desired (or use 1:T)
%     t = linspace(0, 1, T);  
% 
%     if dimtoplot >= 3
%         % Use scatter3 for 3D trajectories
%         scatter3(latentTraj(1,:), latentTraj(2,:), latentTraj(3,:), 36, t, 'filled');
%     else
%         % For 2D trajectories, use scatter
%         scatter(latentTraj(1,:), latentTraj(2,:), 36, t, 'filled');
%     end
% end
% 
% xlabel('Latent 1');
% ylabel('Latent 2');
% zlabel('Latent 3');
% title('Latent Trajectories with Progressive Time Coloring');
% hold off;
