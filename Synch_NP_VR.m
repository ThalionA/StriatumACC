% Evaluate LFP from Neuropixels
%% Open raw data file into matlab
[binName,path_LF] = uigetfile('*.bin', 'Select Binary File');  % Ask user for binary file

%% define intervall to be read out, read out complete synch channel
[meta] = ReadMeta(binName,path_LF); 
synchChannel = ReadChannel(0,Inf,meta,binName,path_LF,385);  %channel 385 is synch channel (global clock)

%% Evaluate virmen VR - V1-CA1_02
[filelist,path_VR] = uigetfile('*.csv', 'Select Multiple Files', 'MultiSelect', 'on' );

% read .csv files
if iscell(filelist) && ~ischar(filelist)    %in case only one file is selected, variable is string, otherwise cell
    n_files=size(filelist,2);
else
    n_files=1;
end

clear fileread;
for i = 1 : n_files
    if n_files == 1
        fid=fopen(path_VR+string(filelist));  
    else
        fid=fopen(path_VR+string(filelist(i)));
    end
    fileread(i,:) = textscan(fid,'%f %f %f %f %f %f %f %f %f','Delimiter',';');  %time|x|dtframe|velocity|world|valvestat|trial#|lick#|synch
    fclose(fid);
end

%% identify long and short synch pulses and create synch_idx
% that defines which times in the Neuropixels recordings belong to which
% time in the VR, synch index is an 1xN array that contains the
% corresponding VR index for a spike.
% Pulses of the NPxl recordings are used as a reference for the alignemnt
% of VR times with the Npxl recording time assuming a linear shift across
% the timespawns between the synch pulses (each min)

VR_data=[];
VR_file=[];
for i = 1 : n_files
    filebegin=size(VR_data,2)+1;
    a=cell2mat(fileread(i,:))';
    if i>1
        a(7,:)=max(VR_data(7,:))+a(7,:);
    end
    VR_data=cat(2,VR_data,a);  % concatenates the files
    fileend=size(VR_data,2);
    VR_file(filebegin:fileend)=i;
    idxPulseChangeVR(filebegin:fileend-1)=diff(VR_data(9,filebegin:fileend));
end
VR_data=cat(1,VR_data,VR_file);

idxPulseChangeNP = diff(synchChannel)/64; %identifies changes in the digital channel (increase of decrease by 64 bit -> into -1,0,1)
idx_NP=find(idxPulseChangeNP~=0);   %ignore the first three, long pulse without VR, and begin of 
idx_VR=find(idxPulseChangeVR~=0)+1; 
clear synchChannel VR_file;

%% find short, long and initialization pulses, and define pulse-off at start of new VR recordings
pulse_class_NP = strings(1,length(idx_NP));    %i - initialization of VR, l - long pulse, s - short pulse
for i=2:length(idx_NP)
    if idxPulseChangeNP(idx_NP(i))==-1
        if idx_NP(i)-idx_NP(i-1)>= 2500*0.8
            pulse_class_NP(i-1) = 'i';
            pulse_class_NP(i) = 'i';
        else
            if idx_NP(i)-idx_NP(i-1) >= 2500*0.4 && idx_NP(i)-idx_NP(i-1) <= 2500*0.6
                pulse_class_NP(i-1) = 'l';
                pulse_class_NP(i) = 'l';
            else
                if idx_NP(i)-idx_NP(i-1) >= 2500*0.15 && idx_NP(i)-idx_NP(i-1) <= 2500*0.3
                    pulse_class_NP(i-1) = 's';
                    pulse_class_NP(i) = 's';
                    if i>2 && pulse_class_NP(i-2) =='i' && pulse_class_NP(i-1) =='s'
                        pulse_class_NP(i) = 'start';
                    end
                end
            end
        end    
    end
end

idx_start=find(pulse_class_NP=='start');
if length(idx_start)>n_files
    i=length(idx_start)-n_files;
    pulse_class_NP(idx_start(1:i))='';
end
    



pulse_class_VR = strings(1,length(idx_VR));
pulse_class_VR(1)='start';  %start - start, L - long pulse, S - short pulse
for i=3:length(idx_VR)
    if idxPulseChangeVR(idx_VR(i)-1)==-1
        if VR_data(1,idx_VR(i))-VR_data(1,idx_VR(i-1)) >= 0.4 && VR_data(1,idx_VR(i))-VR_data(1,idx_VR(i-1)) <= 0.6
            pulse_class_VR(i-1) = 'l';
            pulse_class_VR(i) = 'l';
        else
            if VR_data(10,idx_VR(i))==VR_data(10,idx_VR(i-1))&&(VR_data(1,idx_VR(i))-VR_data(1,idx_VR(i-1)) >= 0.15 && VR_data(1,idx_VR(i))-VR_data(1,idx_VR(i-1)) <= 0.3)
                pulse_class_VR(i-1) = 's';
                pulse_class_VR(i) = 's';
            else
                if VR_data(10,idx_VR(i))~=VR_data(10,idx_VR(i-1))
                    pulse_class_VR(i) = 'start';
                end
            end
        end    
    end
end

%% Match Datapoints of pulse onset and offset to datapoints in the VR:
% a frame in the VR corresponds to a bin that matches several data points
% in the LF and AP file with: 
synchidx=[];
n=1;
for i=1 : length(idx_VR)
    while pulse_class_VR(i)~=pulse_class_NP(n)
        n = n+1;
    end
    synchidx(i)=n;
    n = n+1;
end
%%
VR_times_synched=zeros(1,size(VR_data,2));
VR_offset_time=[];
for i=1:length(synchidx)
    VR_offset_time(i)=idx_NP(synchidx(i))/str2double(meta.imSampRate)-VR_data(1,idx_VR(i));
    if i>1 && i<length(synchidx)
        a = VR_offset_time(i-1);
        b = VR_offset_time(i);
        if VR_data(1,idx_VR(i-1))<VR_data(1,idx_VR(i)) %checking file continuity
            r = (b-a)/((VR_data(1,idx_VR(i))-VR_data(1,idx_VR(i-1)))); % rate of the linear function f(x)=r*x+c to correct the time
            for n = idx_VR(i-1) : idx_VR(i)
                VR_times_synched(n)=(VR_data(1,n)+a)+(VR_data(1,n)-VR_data(1,idx_VR(i-1)))*r;
            end
        else
            idx=find(diff(VR_data(1,idx_VR(i-1):idx_VR(i)))<0)+idx_VR(i-1); % find begin of new file simple linear alignment
            for n = idx_VR(i-1):idx-1
                VR_times_synched(n)=VR_data(1,n)+VR_offset_time(i-1)+r*(VR_data(1,n)-VR_data(1,idx_VR(i-1)));
            end
            for n = idx:idx_VR(i)
                VR_times_synched(n)=VR_data(1,n)+VR_offset_time(i)+r*(VR_data(1,n)-VR_data(1,idx_VR(i)));
            end
        end
    else
        if i==1
            a = VR_offset_time(i);
            b = idx_NP(synchidx(i+1))/str2double(meta.imSampRate)-VR_data(1,idx_VR(i+1));
            r = (b-a)/((VR_data(1,idx_VR(i+1))-VR_data(1,idx_VR(i)))); % rate of the linear function f(x)=r*x+c to correct the time
            for n = 1:idx_VR(i)
                VR_times_synched(n)=VR_data(1,n)+VR_offset_time(i)+r*(VR_data(1,n)-VR_data(1,idx_VR(i)));
            end
        else
            for n = idx_VR(i-1):size(VR_data,2)
                VR_times_synched(n)=VR_data(1,n)+VR_offset_time(i)+r*(VR_data(1,n)-VR_data(1,idx_VR(i-1)));
            end
        end
    end
end
% clear idx_NP idx_VR idxPulseChangeNP idxPulseChangeVR pulse_class_NP pulse_class_VR a b r synchidx VR_offset_time

figure;
plot(VR_times_synched);

figure;
plot(VR_times_synched,VR_data(2,:));
