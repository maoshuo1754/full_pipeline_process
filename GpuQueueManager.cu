//
// Created by csic724 on 2025/3/18.
//

#include "GpuQueueManager.cuh"
#include "kelnels.cuh"


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
    double* d_rasterize_thresholds_wave = d_rasterize_thresholds + wave_idx * NFFT; // 栅格化门限控制
    // 启动 kernel
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (size + blockSize - 1) / blockSize;
    processClutterMapKernel<<<gridSize, blockSize>>>(d_data_wave, d_clutter_map_wave, d_clutterMap_masked_wave,
        size, clutterMap_range_num, alpha, forgetting_factor, clutter_db_offset, d_rasterize_thresholds_wave);

}

double* GpuQueueManager::wave_thresholds(int wave_idx) {
    return d_rasterize_thresholds + wave_idx * NFFT;
}

double* GpuQueueManager::wave_min_speed(int wave_idx) {
    return d_rasterize_min_speed + wave_idx * NFFT;
}

double* GpuQueueManager::wave_max_speed(int wave_idx) {
    return d_rasterize_max_speed + wave_idx * NFFT;
}
