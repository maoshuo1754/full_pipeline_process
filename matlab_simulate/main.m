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
folderPath = '/home/csic724/CLionProjects/PcieReader/cmake-build-release/20250326095501_64GB_frame_1_12_pulse_8_12_2048x4096';

fid = fopen(folderPath, 'rb');
if fid == -1
    error("can't open file %s", folderPath);
end

fileInfos = getFileInfos(folderPath);
startWaveIdx = fileInfos('startWaveIdx');
endWaveIdx = fileInfos('endWaveIdx');
waveNum = endWaveIdx - startWaveIdx;

% h = waitbar(0, 'processing...');

aziTable = readmatrix("azi.txt");

tic;
figure;
for ii = 1:fileInfos('numFrames')
    % if ii ~= 1
    %     continue
    % end
    msg = [num2str(ii), '/', num2str(fileInfos('numFrames'))];
    % waitbar(ii/fileInfos('numFrames'), h, msg);

    
    [time, A] = readBinaryIQFile(fid, fileInfos);
    
    % Get all azimuth values at once
    azi = aziTable(aziTable(:,1) >= startWaveIdx & aziTable(:,1) < startWaveIdx+waveNum, 2);
    azi = fliplr(azi);
    

    % A = data; % Shape: [channels, samples, waves]
    A = fft(A, [], 2); % FFT along time dimension (2nd dim)
    A = A .* PCcoef; % Apply coefficients
    A = ifft(A, [], 2); % IFFT along time dimension
    A = A(:, N_pc+53:500, :); % Range cut
    A = A ./ 2048;
    A = fft(A, Num_V_chnnels, 1); % FFT along channel dimension

    A = A ./ (3000);
    tmp = A(:, :, 1);
    A(1, :, :) = 0;
    % A = fftshift(A, 1); % Shift zero-frequency component
    
    % Normalize entire array
    A = A ./ (sqrt(bandwidth * pulsewidth) * pulseNum);
    
    % A = permute(A, [2, 1, 3]);
    A = cfar(A);
    
    A = A ./ 1024 .* 255;
    % Get maximum along wave dimension (3rd dim)
    % A = max(A, [], 1);
    % A = permute(A, [3, 2, 1]);
    imagesc((1:length(A))*4.8, azi, A);
    title([num2str(ii), '帧']);
    xlabel('距离(m)');ylabel('方位角');
    saveas(gcf, ['res/', num2str(ii), '.png']);
end

t = toc;
fprintf('%.2f\n', t);
fclose(fid);

