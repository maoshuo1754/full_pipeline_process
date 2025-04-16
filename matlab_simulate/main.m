%% 此代码用于将C++ debug输出的文件显示CFAR后结果

clear; clc;
close all;

%% 数据读取
folderPath = '/home/csic724/CLionProjects/PcieReader/cmake-build-debug/20250411165556_128GB_100m_frame_20_30_pulse_14_18_2048x4096';

fid = fopen(folderPath, 'rb');
if fid == -1
    error("can't open file %s", folderPath);
end

% 获取文件总大小
fseek(fid, 0, 'eof'); % 移动到文件末尾
file_size = ftell(fid); % 获取文件大小
fseek(fid, 0, 'bof'); % 返回文件开头

c = 2.99792458e8;
bandwidth = fread(fid, 1, 'double');
pulsewidth = fread(fid, 1, 'double');
Fs = fread(fid, 1, 'double');
lambda = fread(fid, 1, 'double');
PRT = fread(fid, 1, 'double');
start_wave = fread(fid, 1, 'int32');
end_wave = fread(fid, 1, 'int32');
pulseNum = fread(fid, 1, 'int32');
NFFT = fread(fid, 1, 'int32');

azi = fread(fid, 32, 'double');  % 读取32个波束的方位


% 计算波束数量和每个波束的大小
waveNum = end_wave - start_wave;
Ts = 1 / Fs;

param_size = 6 * 8 + 2 * 4 + 32 * 8; % 312字节

% 计算每帧大小
time_size = 4; % uint32时间戳，4字节
data_size = waveNum * pulseNum * NFFT * 8; % cufftComplex数据大小
frame_size = time_size + data_size;

numFrames = floor((file_size - param_size) / frame_size);

% 参数定义
numsamples = round(Fs * pulsewidth);
delta_v = lambda / PRT / pulseNum / 2.0;
delta_range = c / Fs / 2;
fs = 1 / PRT;
Num_V_chnnels = 2048;
f = -fs/2 : fs/Num_V_chnnels : fs/2 - fs/Num_V_chnnels;
v_chnls = f .* lambda / 2;

%% 脉压系数
N_pc = round(pulsewidth * Fs); % pulse compress length
t = linspace(-pulsewidth/2, pulsewidth/2 - Ts, N_pc);
LFM = exp(1j * pi * bandwidth / pulsewidth * t.^2);
PCcoef = conj(fliplr(LFM));
PCcoef = PCcoef .* hamming(N_pc)';
PCcoef = fft(PCcoef, NFFT);
PCcoef = repmat(PCcoef, pulseNum, 1);

%% 读取并处理帧数据
figure;
for ii = 1:numFrames
    % 读取时间和数据
    [time, A] = readBinaryIQFile(fid, waveNum, pulseNum, NFFT);

    % 数据处理
    A = fft(A, [], 2); % FFT along time dimension (2nd dim)
    A = A .* PCcoef; % Apply coefficients
    A = ifft(A, [], 2); % IFFT along time dimension
    A = A(:, N_pc+53:900, :); % Range cut
    A = A ./ 2048;
    A = fft(A, Num_V_chnnels, 1); % FFT along channel dimension

    A = A ./ 3000;
    A(1:20, :, :) = 0;
    A(end-20:end, :, :) = 0;
    
    % Normalize entire array
    A = A ./ (sqrt(bandwidth * pulsewidth) * pulseNum);
    A = abs(A);
    A = squeeze(max(A, [], 1));  % 取最大值，形状变为[range, waveNum]
    A = A';  % 转置为[waveNum, range]

    % 绘图
    A = A ./ 1024 .* 255;
    imagesc((1:size(A, 2)) * 4.8, azi(start_wave+1:end_wave), A);
    title([num2str(ii), '帧']);
    xlabel('距离(m)');
    ylabel('方位角');
    saveas(gcf, ['res/', num2str(ii), '.png']);
end

fclose(fid);