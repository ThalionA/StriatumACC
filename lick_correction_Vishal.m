%% Lick correction, calculating lick rate

licks = VR_data(8,:)>=1;
lick_idx = find(licks==1);
lick_times = VR_times_synched(lick_idx);
lick_freq = 1./diff(lick_times);
figure; histogram(lick_freq,'BinEdges',0:1:70); title('Lick frequency before correction')
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

%%% lick rate is calculated as a rolling rate at every point for 300 ms durations
lick_rate = zeros(1,length(licks2));
lickIntervals = zeros(2,length(licks2));
for i=1:length(licks2)
    Minus150ms = find(VR_times_synched(i)- VR_times_synched <= 0.15,1,'first');
    Plus150ms = find(VR_times_synched - VR_times_synched(i) >= 0.15,1,'first')-1;
    if Minus150ms>=1 & Plus150ms<=length(licks2)
        lickIntervals(1,i)=(VR_times_synched(Plus150ms)-VR_times_synched(Minus150ms));
        lickIntervals(2,i)=sum(licks2(Minus150ms:Plus150ms));
        lick_rate(i) = sum(licks2(Minus150ms:Plus150ms))./(VR_times_synched(Plus150ms)-VR_times_synched(Minus150ms));
    end
end
%lick_rate = gaussFilter(lick_rate,6);
figure;histogram(lick_rate,'BinEdges',1:20);
%clear licks licks_valid licks_positive lick_freq lick_times lick_idx