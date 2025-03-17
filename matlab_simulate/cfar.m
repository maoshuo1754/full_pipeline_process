function cfar_res = cfar(A)
    % Convert input to power domain
    A = abs(A).^2;
    
    % Parameters
    Pfa = 1e-6;
    numGuardCells = 4;
    numRefCells = 20;
    boundary_length = numRefCells + numGuardCells;
    
    % Calculate alpha based on the false alarm probability
    alpha = numRefCells * 2 * (Pfa^(-1/(numRefCells*2)) - 1);
    
    % Get dimensions of input matrix
    [PulseNumber, RangeNumber, WaveNumber] = size(A);
    
    % Create reference kernel for frequency domain processing
    refKernel = [ones(1, numRefCells), zeros(1, 1 + 2*numGuardCells), ones(1, numRefCells)];
    kernel_length = length(refKernel);

    % Prepare kernel for FFT (pad to match range dimension)
    fft_length = 2^nextpow2(RangeNumber + kernel_length - 1); 

    % Compute kernel FFT (only needs to be done once)
    kernel_fft = fft(refKernel, fft_length);
    
    % Compute FFT for all pulses and waves at once
    A_fft = fft(A, fft_length, 2);
    
    % Multiply in frequency domain using broadcasting
    % MATLAB automatically broadcasts the kernel_fft across all pulses and waves
    conv_result = ifft(A_fft .* kernel_fft, [], 2);
    start_idx = floor((kernel_length-1)/2) + 1;
    end_idx = start_idx + RangeNumber - 1;
    conv_result = conv_result(:, start_idx:end_idx, :);

    % Calculate adaptive thresholds
    thresholds = alpha * conv_result / (numRefCells * 2);
    
    thresholds(:, 1:boundary_length, :) = Inf;
    thresholds(:, RangeNumber-boundary_length+1:RangeNumber,:) = Inf;
    
    % Detect targets (where signal exceeds threshold)
    detections = A > thresholds;
    
    % Get detection values (zero out non-detections)
    detection_values = A .* detections;
    
    % Find maximum values across pulses for each range bin and wave
    cfar_res = squeeze(max(detection_values, [], 1));
    
    % Transpose if needed to match the expected output dimensions
    if size(cfar_res, 1) ~= WaveNumber
        cfar_res = cfar_res';
    end
end
