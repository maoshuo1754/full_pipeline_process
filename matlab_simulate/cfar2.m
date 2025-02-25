function cfar_res = cfar2(A)
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
    
    % Create convolution kernels for reference cells
    leftRefKernel = [ones(1, numRefCells), zeros(1, 1 + 2*numGuardCells + numRefCells)];
    rightRefKernel = [zeros(1, numRefCells + 2*numGuardCells + 1), ones(1, numRefCells)];
    
    % Combined reference cells kernel
    refKernel = leftRefKernel + rightRefKernel;
    
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
            detections = abs(pulse_data) > thresholds;
            
            % Update result matrix where detections occur
            detected_indices = find(detections);
            for idx = detected_indices
                % Only update if the current region is valid (outside guard cells)
                if idx > (numRefCells + numGuardCells) && idx <= (RangeNumber - numRefCells - numGuardCells)
                    cfar_res(wave, idx) = max(abs(pulse_data(idx)), cfar_res(wave, idx));
                end
            end
        end
    end
end


