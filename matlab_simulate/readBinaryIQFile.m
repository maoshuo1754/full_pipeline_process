function [time, data] = readBinaryIQFile(fid, waveNum, numPulse, numRange)
    totalElements = waveNum * numPulse * numRange * 2; 

    time = fread(fid, 1, "uint32");
    IQData = fread(fid, totalElements, 'float32');
    

    realPart = IQData(1:2:end);
    imagPart = IQData(2:2:end);

    data = complex(realPart, imagPart);
    data = reshape(data, [numRange, numPulse, waveNum]);
    data = permute(data, [2, 1, 3]);
end