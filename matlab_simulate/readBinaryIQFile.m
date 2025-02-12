function data = readBinaryIQFile(filename, pulseNum, M, N)
    fid = fopen(filename, 'rb');
    if fid == -1
        error("can't open file %s", filename);
    end
    totalElements = pulseNum * M * N * 2; 
    IQData = fread(fid, totalElements, 'float32');
    fclose(fid);

    realPart = IQData(1:2:end);
    imagPart = IQData(2:2:end);

    data = complex(realPart, imagPart);
    data = reshape(data, [N, M, pulseNum]);
    data = permute(data, [2, 1, 3]);
end