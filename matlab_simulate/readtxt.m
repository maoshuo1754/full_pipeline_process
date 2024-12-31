function data = readMatTxt(filePath)
fid = fopen(filePath, 'r');
if fid == -1
error('Error opening file!');
end

% Read dimensions
dims = fscanf(fid, '%d', 2);
m = dims(1);
n = dims(2);

% Read data
data = zeros(m, n);
for i = 1:m
for j = 1:n
        realPart = fscanf(fid, '%f', 1);
imagPart = fscanf(fid, '%f', 1);
data(i, j) = complex(realPart, imagPart);
end
        end

fclose(fid);
end
