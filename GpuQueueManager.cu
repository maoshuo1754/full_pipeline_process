//
// Created by csic724 on 2025/3/18.
//

#include "GpuQueueManager.cuh"
#include "kelnels.cuh"
#include "ipp.h"

void GpuQueueManager::update_queues(const cufftComplex* d_frame, int wave_idx)
{
    std::lock_guard<std::mutex> lock(mutex_);
    launch_update_kernel(d_frame, wave_idx);
    update_count++;
    if (update_count >= CLUTTER_QUEUE_SIZE) {
        launch_clutter_kernel(wave_idx);
    }
    cudaDeviceSynchronize(); // Ensure kernel completion
}

void GpuQueueManager::launch_update_kernel(const cufftComplex* d_frame, int wave_idx)
{
    dim3 blockDim(16, 16);
    dim3 gridDim(
        1,
        (NFFT + blockDim.y - 1) / blockDim.y
    );

    update_queues_kernel<<<gridDim, blockDim>>>(
        d_frame + wave_idx * PULSE_NUM * NFFT,
        d_queues + wave_idx * NFFT * CLUTTER_QUEUE_SIZE,
        d_queues_speed + wave_idx * NFFT * SPEED_CHANNELS * CLUTTER_QUEUE_SIZE,
        d_indices + wave_idx * NFFT,
        PULSE_NUM,
        NFFT,
        CLUTTER_QUEUE_SIZE,
        SPEED_CHANNELS
    );
}

void GpuQueueManager::launch_clutter_kernel(int wave_idx)
{
    dim3 blockDim(16, 16);
    dim3 gridDim(
        1,
        (NFFT + blockDim.y - 1) / blockDim.y
    );
    compute_clutter_kernel<<<gridDim, blockDim>>>(
        d_queues + wave_idx * NFFT * CLUTTER_QUEUE_SIZE,
        d_queues_speed + wave_idx * NFFT * SPEED_CHANNELS * CLUTTER_QUEUE_SIZE,
        d_indices + wave_idx * NFFT,
        d_clutter + wave_idx * NFFT,
        NFFT,
        CLUTTER_QUEUE_SIZE,
        SPEED_CHANNELS
    );
}

// 基于d_data(脉压后数据)计算距离多普勒每个点杂波图是否为杂波
// 超过阈值的d_clutterMap_masked为true，否则为false
// d_data保持不变
void GpuQueueManager::processClutterMap(cufftComplex* d_data, bool* d_clutterMap_masked, int wave_idx, int clutterMap_range_num)
{
    std::lock_guard<std::mutex> lock(clutter_map_mutex_); // 线程安全锁
    size_t size = PULSE_NUM * NFFT;

    // 计算该波束的起始指针
    cufftComplex* d_data_wave = d_data + wave_idx * size;
    float* d_clutter_map_wave = d_clutter_map + wave_idx * size;
    bool* d_clutterMap_masked_wave = d_clutterMap_masked + wave_idx * size;
    // 启动 kernel
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (size + blockSize - 1) / blockSize;
    processClutterMapKernel<<<gridSize, blockSize>>>(d_data_wave, d_clutter_map_wave, d_clutterMap_masked_wave, size, clutterMap_range_num, alpha, forgetting_factor, clutter_db_offset);
}
