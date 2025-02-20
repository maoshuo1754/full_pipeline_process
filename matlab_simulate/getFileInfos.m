function fileInfos = getFileInfos(filePath)
    fileInfo = dir(filePath);
    fileSize = fileInfo.bytes;

    paths = split(filePath, '/');
    filename = paths{end};

    suffix = split(filename, '_');
    startFrameIdx = str2double(suffix{4});
    endFrameIdx = str2double(suffix{5});
    
    startWaveIdx = str2double(suffix{7});
    endWaveIdx = str2double(suffix{8});
    waveNum = endWaveIdx - startWaveIdx;
    
    matrixDims = split(suffix{9}, 'x');
    numPulse = str2double(matrixDims{1});
    numRange = str2double(matrixDims{2});

    totalElements = waveNum * numPulse * numRange * 2; 

    % time - IQ - time - IQ
    numFrames = fileSize / (totalElements *4 + 4);
    
    fileInfos = containers.Map();
    fileInfos('dataname') = suffix{1};
    fileInfos('startWaveIdx') = startWaveIdx;
    fileInfos('endWaveIdx') = endWaveIdx;
    fileInfos('startFrameIdx') = startFrameIdx;
    fileInfos('numFrames') = numFrames;
    fileInfos('numPulse') = numPulse;
    fileInfos('numRange') = numRange;
end