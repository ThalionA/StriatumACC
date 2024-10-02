%%%%%%%%%%%%% COMPLETE DATA COLLATING AND SEQUENCING IN 1 MS TIME SCALE (UNSORTED BY TRIAL, DARK SECTIONS NOT REMOVED) %%%%%%%%%%%%%%
%%% collates all the spike and VR data in 1 ms bin (dark+light) only for the VR length
%%% format: licks|Velocity|position|world|trial no|spout status|spikes
%%% to be run as the first code for data analysis

 %% Read the folder containing all the data %%

folder_name= {'614_M2'};
[folderLength,~]=size(folder_name); 
                                                                                                                                                                                                                                                                                                                           
for kk=1:folderLength
     
     name_now=folder_name{kk};
     
       currentFile=['D:\Final_analysis' '/' name_now ]; %%reads the folder that contains everything for one mouse
       
      cd(currentFile);
end
%% Variable initialization 


tunnel_length=200;  %(AU) the length of the VR corridor. 1 AU= 1.25cm
%time
sr=30000; %samplerate of neuropixel = 30kHz, 30 samples per ms
ttstep=1/1000; %use 1ms window to bin data
AU_cm=1.25;
trial_length=3; % 3s
minsend=140; %maximum experiment duration per mouse in mins
time_array=0:ttstep:minsend*60; %max sample time for trials binned every 1 ms
darkt=10; %10mins darkness after  task

dms_start = 650; dms_end = 1150;
dls_start = 0; dls_end = 450;

%%% VR_data =  1  |2|   3   |   4    | 5   |    6    |  7   |  8  |  9  |10
%%%           time|x|dtframe|velocity|world|valvestat|trial#|lick#|synch|file
%%% VR_times_synched contains the corrected time of the VR data
% this is the file from the sychonization script
% after runing the script save everything and name it accstr
cluster=readNPY('spike_clusters.npy');
spike_times=readNPY('spike_times.npy');
disp('data imported');
%% Selecting the good clusters from manual labeling


[rawDataNum, rawDataStr]=xlsread('info'); %save cluster_info.tsv as info.xls
%%% rawDataNum in the number values array of the KS data 
%%% rawDataStr has the KS and manual 'good' markings
%%% rawDataNum relevent columns: 1-ID, 6-channel, 7-depth
%%% rawDataStr relevent colums: 4-KSLabel, 9-manual label

CH=rawDataNum(:,6); %channel
Depth=rawDataNum(:,7);% depth of cluster
chanMap = readNPY('channel_map.npy');
cluster_id=rawDataNum(:,1);  %cluster ID

for i=2:length(rawDataStr)
    KS_label{i-1,1} = rawDataStr{i,4}; %kilosort labels
    Man_label{i-1,1} = rawDataStr{i,9}; %manual labels

end

goodcluster=[];
clustergroup={};

gc=find(contains(Man_label, 'good')); %index of good labels
goodcluster = rawDataNum(gc, [1,7]); %IDs and depth of neurons manually labeled good

%% Matching spikes to their respective clusters

%%% counting spikes in every ms
for i=1:length(goodcluster)
    cluster2=[];
    clustergroup{i,1} = spike_times(cluster==goodcluster(i,1)); % grouping spike timepoints of a given cluster
    clustergroup{i,2} = goodcluster(i,1); % cluster id of that cluster
    cluster2=double(clustergroup{i,1});
    spike_count(i,:) = histcounts(cluster2(:,1)./sr,time_array); % spike count in every ms
    %spike_count_gf(i,:)=gaussFilter(spike_count(i,:),sigma_spike);
end
clear cluster2 spike_count2 gc2 gc m z i KS_label Man_label
disp('spike counts done');


%% Alining the spike time series with VR time series

idx_vr_to_spike = {}; %spike indices corresponding to VR times. Columns = 1:VR_data indices|2: corresponding spike indices
j = 1;
%%% The indices of spike time scale that correspond to the VR time scale are
%%% matched with each other
for i = 1:length(VR_times_synched)-1
    while VR_times_synched(i) > time_array(j)
        j=j+1;
    end
    start = j;
    while VR_times_synched(i) <= time_array(j) && time_array(j) < VR_times_synched(i+1)
        j=j+1;
    end
    idx_vr_to_spike{i,1} = i;
    idx_vr_to_spike{i,2} = start:j-1;
    % every index i of VR_data corresponds to start:j-1 of the time_array
end

%%% getting the spike count only for the VR duration
scc = spike_count(:,ceil(VR_times_synched(1)*1000):floor(VR_times_synched(end)*1000));
%% Lick correction, calculating lick rate

licks = VR_data(8,:)>=1;
lick_idx=find(licks==1);
lick_times=VR_times_synched(lick_idx);
lick_freq= 1./diff(lick_times);
figure;histogram(lick_freq,'BinEdges',0:1:70); title('Lick frequency before correction')
licks2 = zeros(1,length(licks));
licks_positive = find(licks==1);
licks_valid = licks_positive(1);
for i = 2:length(licks_positive)
    l = find(licks_valid(1:i-1)~=0,1,'last');
    if ~isempty(l)
     if VR_times_synched(licks_positive(i)) - VR_times_synched(licks_valid(l)) >0.1
         licks_valid(i) = licks_positive(i);
     else
         licks_valid(i) = 0;
     end
    else
        licks_valid(i) = licks_positive(i);
    end
end
n=licks_valid((licks_valid~=0));
licks2(n) = 1;
lick_idx=find(licks2==1);
lick_times=VR_times_synched(lick_idx);
lick_freq= 1./diff(lick_times);
figure;histogram(lick_freq,'BinEdges',0:1:70); title('Lick frequency after correction')
disp('Lick correction done');
clear licks licks_valid licks_positive lick_freq lick_times lick_idx

%% Interpolation to 1ms bin
%%% allData is a structure that contains all the relevant data in 1 ms bins
%%% allData format: licks|Velocity|position|world|trial no|spout status|VR_times_synched|spikes
idx = 1;
for i=1:length(VR_data)-1
    allData{1,1}(idx) = licks2(i);%right licks
    allData{1,1}(idx+1:idx+length(idx_vr_to_spike{i,2})-1) = 0;%licks interpolated with zeros 
    allData{1,2}(idx:idx+length(idx_vr_to_spike{i,2})-1) = VR_data(4,i);%velocity
    allData{1,3}(idx:idx+length(idx_vr_to_spike{i,2})-1) = VR_data(2,i);%position
    allData{1,4}(idx:idx+length(idx_vr_to_spike{i,2})-1) = VR_data(5,i);%world
    allData{1,5}(idx:idx+length(idx_vr_to_spike{i,2})-1) = VR_data(7,i);%trial no
    allData{1,6}(idx:idx+length(idx_vr_to_spike{i,2})-1) = VR_data(6,i);%spout status
    if i<length(VR_data)-1
        idx=idx+length(idx_vr_to_spike{i,2});
    end
end
allData{1,2} = gaussFilter(allData{1,2},30); %smoothening velocity with 30ms gaussian sigma width
allData{1,8} = sc_chopped;
allData{2,8} = 'Spikes';
allData{2,1} = 'Licks';
allData{2,2} = 'Velocity';
allData{2,3} = 'Position';
allData{2,4} = 'World';
allData{2,5} = 'Trial no';
allData{2,6} = 'Spout status';
allData{1,7} = VR_times_synched;
allData{2,7} = 'VR times synched';
save('E:/All mice - 1ms binned/614_all data_1ms(raw)/allData_1ms.mat', 'allData');
save('E:/All mice - 1ms binned/614_all data_1ms(raw)/goodcluster.mat', 'goodcluster');
disp('Finished binning and storing the data');