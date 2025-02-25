function cfar_res = cfar(A)

    Pfa = 1e-6;       
    numGuardCells = 4;  
    numRefCells = 20;  
    total_training_cells = numGuardCells + numRefCells;
    alpha = numRefCells * 2 * (Pfa^(-1/(numRefCells*2)) - 1);
    
    [PulseNumber, RangeNumber, WaveNumber] = size(A);
    cfar_res = zeros(WaveNumber, RangeNumber);
    for wave = 1:WaveNumber
        PCIQBufAbs = A(:, :, wave);
        
        for pulse = 1:PulseNumber
            for i = total_training_cells+1: RangeNumber-total_training_cells
                if i == total_training_cells+1
                    noise_level = sum(PCIQBufAbs(pulse, 1:numRefCells)) + ...
                              sum(PCIQBufAbs(pulse, total_training_cells+numGuardCells+2:2*total_training_cells+1));
                else
                    noise_level = noise_level + ...
                                  PCIQBufAbs(pulse, i-1-numGuardCells) + ...
                                  PCIQBufAbs(pulse, i+numGuardCells+numRefCells);
        
                    noise_level = noise_level - ...
                          (PCIQBufAbs(pulse, i-numRefCells-numGuardCells-1)) - ...
                          (PCIQBufAbs(pulse, i+numGuardCells));
                end
        
                threshold = alpha * noise_level / numRefCells / 2;
                if abs(PCIQBufAbs(pulse, i)) > threshold
                    cfar_res(wave, i) = max(PCIQBufAbs(pulse, i), cfar_res(wave, i));
                    % disp(sprintf("%d pulse, ind %d", pulse, i));
                end
            end
        end
    end
end