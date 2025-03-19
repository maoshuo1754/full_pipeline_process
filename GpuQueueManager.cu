//
// Created by csic724 on 2025/3/18.
//

#include "GpuQueueManager.cuh"
#include "kelnels.cuh"

void GpuQueueManager::launch_update_kernel(const cufftComplex* d_frame)
{
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (WAVE_NUM + blockDim.x - 1) / blockDim.x,
        (NFFT + blockDim.y - 1) / blockDim.y
    );
    update_queues_kernel<<<gridDim, blockDim>>>(
        d_frame, d_queues, d_queues_speed, d_indices, WAVE_NUM, PULSE_NUM, NFFT, CLUTTER_QUEUE_SIZE, SPEED_CHANNELS
    );
}

void GpuQueueManager::launch_clutter_kernel()
{
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (WAVE_NUM + blockDim.x - 1) / blockDim.x,
        (NFFT + blockDim.y - 1) / blockDim.y
    );
    compute_clutter_kernel<<<gridDim, blockDim>>>(
        d_queues, d_queues_speed, d_indices, d_clutter, WAVE_NUM, NFFT, CLUTTER_QUEUE_SIZE, SPEED_CHANNELS
    );
}
