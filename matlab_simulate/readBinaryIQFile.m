function [time, data] = readBinaryIQFile(fid, fileInfos)

    numRange = fileInfos('numRange');
    numPulse = fileInfos('numPulse');

    waveNum = fileInfos('endWaveIdx') - fileInfos('startWaveIdx');
    totalElements = waveNum * numPulse * numRange * 2; 

    time = fread(fid, 1, "uint32");
    IQData = fread(fid, totalElements, 'float32');
    

    realPart = IQData(1:2:end);
    imagPart = IQData(2:2:end);

    data = complex(realPart, imagPart);
    data = reshape(data, [numRange, numPulse, waveNum]);
    data = permute(data, [2, 1, 3]);

end