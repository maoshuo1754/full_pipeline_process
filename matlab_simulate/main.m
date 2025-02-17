clear;clc;
close all;

%% 参数定义
bandwidth = 6e6;
pulsewidth = 5e-6;
Fs = 31.25e6;
Ts = 1/Fs;
c = 3e8;
numsamples = round(Fs*pulsewidth);

lambda = 0.0369658;
PRT = 120e-6;
NUM_PULSE = 2048;
delta_v = lambda / PRT / NUM_PULSE / 2.0;
delta_range = c / Fs / 2;

fs = 1 / PRT;

Num_V_chnnels = 2048;
f = -fs/2:fs/Num_V_chnnels:fs/2-fs/Num_V_chnnels;
v_chnls = f .* lambda / 2;
% v_chnls = fftshift(v_chnls);

pulseNum = 2048;
NFFT = 4096;

%% 脉压系数
N_pc = round(pulsewidth * Fs); % pulse compress length
t = linspace(-pulsewidth/2, pulsewidth/2-Ts, N_pc);
LFM = exp(1j * pi * bandwidth / pulsewidth * t.^2);
PCcoef = conj(fliplr(LFM));
PCcoef = PCcoef.*hamming(N_pc)';
PCcoef = fft(PCcoef, NFFT);
PCcoef = repmat(PCcoef, pulseNum, 1);

%% 数据读取
timePath = '20250213142451_256GB_res/time.txt';
times = readmatrix(timePath);

outMatrix = zeros(length(times), 19);
outMatrix(:, 3) = times./1000;

folderPath = '/home/csic724/CLionProjects/PcieReader/cmake-build-release/20250213142451_256GB/';
fileList = dir(folderPath);
filenames = strings(0);

for i = 1:length(fileList)
    if ~fileList(i).isdir
        filenames = [filenames; fileList(i).name];
    end
end

figure;
for ii = 1:length(filenames)
    if ii ~= 146
        continue
    end
    disp(ii)
    filename = filenames(ii);
    suffix = split(filename, '.');
    suffix = split(suffix{2}, '_');
    waveIdx = str2double(suffix{3});
    
    startPulseIdx = str2double(suffix{5});
    endPulseIdx = str2double(suffix{6});
    waveNum = endPulseIdx - startPulseIdx;
    
    matrixDims = split(suffix{7}, 'x');
    M = str2double(matrixDims{1});
    N = str2double(matrixDims{2});
    
    data = readBinaryIQFile(folderPath + filename, waveNum, M, N);

    for jj = 1:1
        A = data(:,:,jj);
        A = fft(A, [], 2);
        A = A .* PCcoef;
        A = ifft(A, [], 2);
        A = fft(A, Num_V_chnnels, 1);
            
        
        A = fftshift(A,1);
        inds = find(v_chnls < -20 | v_chnls > -10);
        A(inds, :) = 0;
        A = A ./ (sqrt(bandwidth * pulsewidth) * pulseNum);
        A = abs(A);
        
        A = A(:, (N_pc-1) + 52 - 2:end);
        A(:, 1:round(89+0.7822*ii)) = 0;
        [maxVaule, ind] = max(A, [], "all");
        [row, col] = ind2sub(size(A), ind);
        range = col * delta_range;
        v = v_chnls(row);
        disp(v);
    end
    outMatrix(ii, end) = abs(v) * 100;
    outMatrix(ii, end-1) = 4.8*col;
    disp(['v:', num2str(v_chnls(row)), 'm/s range:', num2str(4.8*col)]);
    imagesc((1:length(A))*delta_range, v_chnls,A);
end

writematrix(outMatrix, '20250213142451_256GB_res/out2.txt', 'Delimiter', 'tab');
