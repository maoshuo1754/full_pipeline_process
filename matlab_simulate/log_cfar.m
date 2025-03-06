function cfar_res = log_cfar(A)
    % Convert input to logarithmic power domain
    % 10*log10(|A|²) = 20*log10(|A|)
    A_log = 20*log10(abs(A));

    % Parameters
    Pfa = 1e-6;
    numGuardCells = 4;
    numRefCells = 20;

    % Get dimensions of input matrix
    [PulseNumber, RangeNumber, WaveNumber] = size(A);

    % Create reference kernel for frequency domain processing
    refKernel = [ones(1, numRefCells), zeros(1, 1 + 2*numGuardCells), ones(1, numRefCells)];
    kernel_length = length(refKernel);


    % Prepare kernel for FFT (pad to match range dimension)
    fft_length = 2^nextpow2(RangeNumber + kernel_length - 1);
    padded_kernel = zeros(1, fft_length);
    padded_kernel(1:length(refKernel)) = refKernel;

    % Compute kernel FFT (only needs to be done once)
    kernel_fft = fft(padded_kernel);

    % Compute FFT for all pulses and waves at once
    A_log_fft = fft(A_log, fft_length, 2);

    % Multiply in frequency domain using broadcasting
    conv_result = ifft(A_log_fft .* kernel_fft, [], 2);
    start_idx = floor((kernel_length-1)/2) + 1;
    end_idx = start_idx + RangeNumber - 1;
    conv_result = conv_result(:, start_idx:end_idx, :);


    % For logarithmic CFAR, threshold is additive rather than multiplicative
    % Calculate threshold factor based on Pfa (different calculation for log domain)
    % In log domain, we use T = μ + α where μ is the mean of reference cells
    % α is calculated differently for log domain
    alpha_log = 10*log10(-log(Pfa));

    % Calculate adaptive thresholds (mean of reference cells + alpha)
    thresholds = conv_result / (numRefCells * 2) + alpha_log;

    % Detect targets (where signal exceeds threshold)
    detections = A_log > thresholds;

    % Get detection values (zero out non-detections)
    % Convert A back to linear scale for consistency with the original function
    detection_values = A_log .* detections;

    % Find maximum values across pulses for each range bin and wave
    cfar_res = squeeze(max(detection_values, [], 1));

    % Transpose if needed to match the expected output dimensions
    if size(cfar_res, 1) ~= WaveNumber
            cfar_res = cfar_res';
    end
end