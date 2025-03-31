//
// Created by csic724 on 2025/3/18.
//

#include "GpuQueueManager.cuh"
#include "kelnels.cuh"

void GpuQueueManager::update_queues(const cufftComplex* d_frame)
{
    std::lock_guard<std::mutex> lock(mutex_); // 线程安全锁
    launch_update_kernel(d_frame);            // 更新队列数据
    update_count++;                           // 更新计数器加1
    if (update_count >= CLUTTER_QUEUE_SIZE) {
        launch_clutter_kernel();
    }
}

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

// 基于d_data(脉压后数据)计算距离多普勒每个点杂波图是否为杂波
// 超过阈值的d_clutterMap_masked为true，否则为false
// d_data保持不变
void GpuQueueManager::processClutterMap(cufftComplex* d_data, bool* d_clutterMap_masked, int clutterMap_range_num)
{
    std::lock_guard<std::mutex> lock(clutter_map_mutex_); // 线程安全锁
    size_t size = WAVE_NUM * PULSE_NUM * NFFT;

    // 启动 kernel
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (size + blockSize - 1) / blockSize;
    processClutterMapKernel<<<gridSize, blockSize>>>(d_data, d_clutter_map, d_clutterMap_masked, size, clutterMap_range_num, alpha, forgetting_factor, clutter_db_offset);
}
