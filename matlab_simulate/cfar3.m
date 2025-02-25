function cfar_res = cfar3(A)
    A = abs(A).^2;

    % Parameters
    Pfa = 1e-6;
    numGuardCells = 4;
    numRefCells = 20;
    
    % Calculate alpha based on the false alarm probability
    alpha = numRefCells * 2 * (Pfa^(-1/(numRefCells*2)) - 1);
    
    % Get dimensions of input matrix
    [PulseNumber, RangeNumber, WaveNumber] = size(A);
    
    % Initialize result matrix
    cfar_res = zeros(WaveNumber, RangeNumber);
    
    % Create a 3D array to store all detections before selecting maximum
    all_detections = zeros(PulseNumber, RangeNumber, WaveNumber);
    
    % Create convolution kernels for reference cells
    refKernel = [ones(1, numRefCells), zeros(1, 1 + 2*numGuardCells), ones(1, numRefCells)];
    
    for wave = 1:WaveNumber
        wave_data = A(:, :, wave);
        
        for pulse = 1:PulseNumber
            % Get current pulse data
            pulse_data = wave_data(pulse, :);
            
            % Calculate noise levels using convolution
            noise_levels = conv(pulse_data, refKernel, 'same');
            
            % Calculate adaptive thresholds
            thresholds = alpha * noise_levels / (numRefCells * 2);
            
            % Compare signal with threshold
            detections = pulse_data > thresholds;
            
            % Store detection values
            all_detections(pulse, detections, wave) = abs(pulse_data(detections));
        end
    end
    
    % After all pulses and waves are processed, select the maximum values across pulses
    for wave = 1:WaveNumber
        % Find the maximum detection value across all pulses for each range bin
        cfar_res(wave, :) = max(all_detections(:, :, wave), [], 1);
    end
end