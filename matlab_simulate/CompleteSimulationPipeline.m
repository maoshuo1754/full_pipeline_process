clc; clear all; close all;

C = 3e8;              % speed of light

% para
BandWidth = 15e6;     %
PulseWidth = 2e-6;    % 
Fs = 31.25e6;         % 
PRT = 100e-6;         % 
PulseNumber = 2048;   % 
Fc = 10e9;            % 
Ts = 1/Fs;            % 
Lambda = C / Fc;      % wavelength 0.03m

% target parameter
R0 = 1500;            % m
v = 10;               % m/s
a = 0;                % m/s^2
N = round(PulseWidth * Fs);
RangeNumber = ceil(PRT * Fs); 

PRF = 1 / PRT;        % pulse repeat frequency
PRF_Ind = linspace(-PRF/2, PRF/2 - PRF/PulseNumber, PulseNumber);

% �������Ե�Ƶ�źź���ѹϵ��
t = linspace(-PulseWidth/2, PulseWidth/2-Ts, N);
LFM = exp(1j * pi * BandWidth / PulseWidth * t.^2);
PCcoef = conj(fliplr(LFM));
TargetEcho = repmat(LFM, PulseNumber, 1);

% ÿ������ʱ�������ȷֿ���ÿ��Ӧ�ö�����Ƶ��
t = ((0:PulseNumber-1) * PRT).';
fd = 2 * (v + a * t) / Lambda;
DopplerFreqMod = repmat(exp(1i * 2 * pi * fd .* t), 1, N);
TargetEcho = TargetEcho .* DopplerFreqMod;

% �ز��ź����Ŀ���ź�
echo = zeros(PulseNumber, RangeNumber);
TargetRange = R0 + v*t + 0.5*a*t.^2;
DeltRange = C * Ts / 2; % range delta
TargetStartIdx = round(TargetRange / DeltRange);

% add target
for i = 1 : PulseNumber
    echo(i, TargetStartIdx(i):TargetStartIdx(i)+N-1) = echo(i, TargetStartIdx(i):TargetStartIdx(i)+N-1) + TargetEcho(i,:);
end

% add noise
SNR = 0;
SignalPower = mean(abs(echo(:)).^2);
NoisePower = SignalPower / 10^(SNR / 10);
Noise = sqrt(NoisePower / 2) * (randn(size(echo)) + sqrt(NoisePower / 2)*1i*randn(size(echo)));
echo = echo + Noise;

% [m, n] = size(echo);
% fileID = fopen('data.txt', 'w');
% fprintf(fileID, '%d %d\n', m, n);
% for i = 1:m
%     for j=1:n
%         fprintf(fileID, '%f %f\n', real(echo(i, j)), imag(echo(i, j)));
%     end
% end
% fclose(fileID);


% pulse compression
PCIQBuf = zeros(PulseNumber, RangeNumber);
N_seq = RangeNumber + N - 1;
NFFT = 2^nextpow2(N_seq); 

echoFFT = fft(echo, NFFT, 2);
fft2 = fft(PCcoef, NFFT);

for ii = 1 : PulseNumber
    xx = ifft(echoFFT(ii,:) .* fft2);
    PCIQBuf(ii,:) = xx(N:N+RangeNumber-1);
end


figure;
rangeInd = (0:RangeNumber-1) .* DeltRange;
PCIQBufAbs = abs(PCIQBuf) / max(max(abs(PCIQBuf)));
PCIQBufAbsMax = max(PCIQBufAbs);
plot(rangeInd, 20*log10(PCIQBufAbsMax));
xlabel("range(m)");ylabel("amplitude(dB)");
title('fast time-domain waveform');
grid on;


PCIQBuf_fft = fft(PCIQBuf, [], 1);
PCIQBuf_fft = fftshift(PCIQBuf_fft, 1);
PCIQBuf_fft = abs(PCIQBuf_fft);
PCIQBuf_fft = PCIQBuf_fft ./ max(PCIQBuf_fft, [], 'all');
PCIQBuf_fft = 20*log10(PCIQBuf_fft);

figure;
v_ind = PRF_Ind * Lambda / 2;
imagesc(rangeInd, v_ind, PCIQBuf_fft);
xlabel('range(m)');
ylabel('v(m/s)');
title('result of coherent integration');