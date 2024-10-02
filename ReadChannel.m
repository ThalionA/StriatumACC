function dataArray = ReadChannel(samp0, nSamp, meta, binName, path, channel)

    nChan = str2double(meta.nSavedChans);

    nFileSamp = str2double(meta.fileSizeBytes) / (2 * nChan);
    samp0 = max(samp0, 0);
    nSamp = min(nSamp, nFileSamp - samp0);

    %sizeA = [length(channels), nSamp]; %TF: added
    sizeA = [1, nSamp];
    
    fid = fopen(fullfile(path, binName), 'rb');
    n=length(channel);
    dataArray =zeros(n,nSamp);
    for i = 1:n
        fseek(fid, samp0 * 2 * nChan+(channel(i)-1)*2, 'bof');
        dataArray(i,:) = fread(fid, sizeA, 'int16=>double', (nChan-1)*2); %skipn); %TF: added
        %dataArray = fread(fid, sizeA, 'int16=>double');
    end
    fclose(fid);
end % ReadBin