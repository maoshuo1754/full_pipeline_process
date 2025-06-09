//
// Created by csic724 on 2025/3/18.
//

#ifndef GPUQUEUEMANAGER_CUH
#define GPUQUEUEMANAGER_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include "Config.h"
#include "utils.h"
#include <thrust/execution_policy.h>
#include <mutex>


// 队列管理类（单例模式）  占用显存32MB
// 用于处理多线程之间需要共享的资源，比如队列和杂波图
class GpuQueueManager {
public:
    // 获取单例实例
    static GpuQueueManager& getInstance() {
        static GpuQueueManager instance; // 静态实例，保证唯一性
        return instance;
    }

    // 删除拷贝构造函数和赋值操作符，确保单例
    GpuQueueManager(const GpuQueueManager&) = delete;
    GpuQueueManager& operator=(const GpuQueueManager&) = delete;

    void processClutterMap(cufftComplex* d_data, bool* d_clutterMap_masked_, int wave_idx, int clutterMap_range_num);
    double* wave_thresholds(int wave_idx);
    double* wave_min_speed(int wave_idx);
    double* wave_max_speed(int wave_idx);

private:
    double* d_rasterize_thresholds;         // 栅格化门限
    double* d_rasterize_min_speed;          // 栅格化最小速度
    double* d_rasterize_max_speed;          // 栅格化最大速度

    // 下面是杂波图的参数
    float* d_clutter_map = nullptr;         // 杂波图历史记录
    double alpha;
    std::mutex clutter_map_mutex_;          // 保护 d_clutter_map 的访问互斥锁

    // 构造函数：私有化以实现单例
    GpuQueueManager() {
        checkCudaErrors(cudaMalloc(&d_clutter_map, WAVE_NUM * PULSE_NUM * NFFT * sizeof(float)));
        checkCudaErrors(cudaMemset(d_clutter_map, 0, WAVE_NUM * PULSE_NUM * NFFT * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_rasterize_thresholds, WAVE_NUM * NFFT * sizeof(double)));
        checkCudaErrors(cudaMalloc(&d_rasterize_min_speed, WAVE_NUM * NFFT * sizeof(double)));
        checkCudaErrors(cudaMalloc(&d_rasterize_max_speed, WAVE_NUM * NFFT * sizeof(double)));

        int m, n;
        size_t size = WAVE_NUM * NFFT * sizeof(double);
        auto h_rasterize_thresholds = readCSVToGPU("/home/csic724/CLionProjects/PcieReader/matlab_simulate/栅格化生成/threshold.csv", m, n);
        assert(m == WAVE_NUM && n == NFFT);
        if (!rasterize_manage_enable) {
            checkCudaErrors(cudaMemset(d_rasterize_thresholds, 0, size));
        }
        else {
            checkCudaErrors(cudaMemcpy(d_rasterize_thresholds, h_rasterize_thresholds.data(), size, cudaMemcpyHostToDevice));
        }

        auto h_rasterize_min_speed = readCSVToGPU("/home/csic724/CLionProjects/PcieReader/matlab_simulate/栅格化生成/min_speed.csv", m, n);
        assert(m == WAVE_NUM && n == NFFT);
        checkCudaErrors(cudaMemcpy(d_rasterize_min_speed, h_rasterize_thresholds.data(), size, cudaMemcpyHostToDevice));

        auto h_rasterize_max_speed = readCSVToGPU("/home/csic724/CLionProjects/PcieReader/matlab_simulate/栅格化生成/max_speed.csv", m, n);
        assert(m == WAVE_NUM && n == NFFT);
        checkCudaErrors(cudaMemcpy(d_rasterize_max_speed, h_rasterize_thresholds.data(), size, cudaMemcpyHostToDevice));

        alpha = getClutterMapAlpha(forgetting_factor, Pfa_clutter_map);
    }

    // 析构函数：释放显存资源
    ~GpuQueueManager() {
        checkCudaErrors(cudaFree(d_clutter_map));
    }
};


#endif //GPUQUEUEMANAGER_CUH
