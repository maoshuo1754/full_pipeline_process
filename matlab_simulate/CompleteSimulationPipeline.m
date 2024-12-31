clc; clear all; close all;

% ��������
C = 3e8;              % ���٣�m/s��

% �״����
BandWidth = 15e6;     % ���� ��Hz��
PulseWidth = 2e-6;    % �����ȣ��룩
Fs = 31.25e6;         % ����Ƶ�ʣ�Hz��
PRT = 100e-6;         % �����ظ�ʱ�䣨�룩��������������֮���ʱ����
PulseNumber = 2048;   % �����ظ�����
Fc = 10e9;            % �ز�Ƶ��
Ts = 1/Fs;            % ����ʱ�������룩
Lambda = C / Fc;      % ����0.03m

% Ŀ�����
R0 = 1500;            % Ŀ����� m
v = 10;                % Ŀ���ٶȣ�m/s��
a = 0;                % Ŀ����ٶȣ�m/s^2)
N = round(PulseWidth * Fs);   % ����ĵ�Ԫ��
RangeNumber = ceil(PRT * Fs); % ���뵥Ԫ��

PRF = 1 / PRT;        % �����ظ�Ƶ��
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
DeltRange = C * Ts / 2; % ���뵥Ԫ��m��
TargetStartIdx = round(TargetRange / DeltRange);

% ����ź�
for i = 1 : PulseNumber
    echo(i, TargetStartIdx(i):TargetStartIdx(i)+N-1) = echo(i, TargetStartIdx(i):TargetStartIdx(i)+N-1) + TargetEcho(i,:);
end

% ������������
SNR = 0;
SignalPower = mean(abs(echo(:)).^2);
NoisePower = SignalPower / 10^(SNR / 10);
Noise = sqrt(NoisePower / 2) * (randn(size(echo)) + sqrt(NoisePower / 2)*1i*randn(size(echo)));
echo = echo + Noise;

[m, n] = size(echo);
fileID = fopen('data.txt', 'w');
fprintf(fileID, '%d %d\n', m, n);
for i = 1:m
    for j=1:n
        fprintf(fileID, '%f %f\n', real(echo(i, j)), imag(echo(i, j)));
    end
end
fclose(fileID);


% ��������ѹ��
PCIQBuf = zeros(PulseNumber, RangeNumber);
N_seq = RangeNumber + N - 1;
NFFT = 2^nextpow2(N_seq); 
for ii = 1 : PulseNumber
    fft1 = fft(echo(ii,:), NFFT);
    fft2 = fft(PCcoef, NFFT);
    xx = ifft(fft1 .* fft2);
    startInd = (N);
    PCIQBuf(ii,:) = xx(startInd:startInd+RangeNumber-1);
end


figure;
rangeInd = (0:RangeNumber-1) .* DeltRange;
PCIQBufAbs = abs(PCIQBuf) / max(max(abs(PCIQBuf)));
PCIQBufAbsMax = max(PCIQBufAbs);
plot(rangeInd, 20*log10(PCIQBufAbsMax));
xlabel("����(m)");ylabel("����(dB)");
title('���˲���ʱ������');
grid on;

PCIQBuf_fft = zeros(size(PCIQBuf.'));
PCIQBuf_fft_tmp = PCIQBuf.';

for i = 1:RangeNumber
    PCIQBuf_fft(i,:) = abs(fftshift(fft(PCIQBuf_fft_tmp(i,:))));
    yy = PCIQBuf_fft(i,:);
    PCIQBuf_fft(i,:) = 20*log10(yy / max(yy));
end
figure;
v_ind = PRF_Ind * Lambda / 2;
imagesc(rangeInd, v_ind, PCIQBuf_fft.');
xlabel('����(m)');
ylabel('�ٶ�(m/s)');

% figure;
% yy = PCIQBuf_fft(275,:);
% plot(v_ind, yy);



% PCIQBuf_fft = PCIQBuf.';
% yy = PCIQBuf_fft(313, :);
% yy = abs(fftshift(fft(yy)));
% 
% v_ind = PRF_Ind * Lambda / 2;
% figure;
% plot(v_ind, 20*log10(yy / max(yy)));
% xlabel('�ٶ�(m/s)');
% ylabel('����(dB)');
% title('���˲���ʱ������');


