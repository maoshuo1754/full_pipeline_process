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
folderPath = '20250209143437_256GB_frame_1_60_pulse_13_16_2048x4096';

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



for ii = 1:fileInfos('numFrames')
    msg = [num2str(ii), '/', num2str(fileInfos('numFrames'))];
    % waitbar(ii/fileInfos('numFrames'), h, msg);
    if ii ~= 1
        continue;
    end
    figure;
    [time, data] = readBinaryIQFile(fid, fileInfos);

    for jj = 1:waveNum
        azi = aziTable(aziTable(:,1) == startWaveIdx+jj-1, 2);
        A = data(:,:,jj);
        A = fft(A, [], 2);
        A = A .* PCcoef;
        A = ifft(A, [], 2);
        A = A(:, N_pc+53:500);

        A = fft(A, Num_V_chnnels, 1);
        A = fftshift(A, 1);
        % inds = find(v_chnls < -20 | v_chnls > -10);
        
        A = A ./ (sqrt(bandwidth * pulsewidth) * pulseNum);                
        A = abs(A);
        A = 20*log10(A);
        A(1:2, :) = 0;
        A = max(A);
    end
end

fclose(fid);

