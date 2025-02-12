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
f = -fs/2:fs/NUM_PULSE:fs/2-fs/NUM_PULSE;
v_chnls = f .* lambda / 2;
v_chnls = fftshift(v_chnls);

%% 数据读取
filename = '20250209143316_256GB.dat_wave_1_pulse_19_23_2048x4096';

suffix = split(filename, '.');
suffix = split(suffix{2}, '_');
waveIdx = str2double(suffix{3});

startPulseIdx = str2double(suffix{5});
endPulseIdx = str2double(suffix{6});
waveNum = endPulseIdx - startPulseIdx;

matrixDims = split(suffix{7}, 'x');
M = str2double(matrixDims{1});
N = str2double(matrixDims{2});

data = readBinaryIQFile(filename, waveNum, M, N);

%% 脉压系数
N_pc = round(pulsewidth * Fs); % pulse compress length
t = linspace(-pulsewidth/2, pulsewidth/2-Ts, N_pc);
LFM = exp(1j * pi * bandwidth / pulsewidth * t.^2);
PCcoef = conj(fliplr(LFM));
PCcoef = PCcoef.*hamming(N_pc)';
PCcoef = fft(PCcoef, N);
PCcoef_cpp = readmatrix("PcCoefMatrix.txt");
if areMareicesEqual(PCcoef, PCcoef_cpp, 1e-4)
    disp("PCcoef is OK");
else
    disp("PCcoef is WRONG");
end

PCcoef = repmat(PCcoef, M, 1);

A = data(:, :, 1);
B = readmatrix("data_pulse19_beforePC.txt");
if areMareicesEqual(A, B, 1e-6)
    disp("Before PC is OK");
else
    disp("Before PC is WRONG");
end


A = fft(A, [], 2);
A = A .* PCcoef;

A = ifft(A, [], 2);

C = readmatrix("data_pulse19_afterPC.txt");
C = C ./ N;
if areMareicesEqual(A, C, 1)
    disp("After PC is OK");
else
    disp("After PC is WRONG");
end
