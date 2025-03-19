//
// Created by csic724 on 2025/3/18.
//

#ifndef GPUQUEUEMANAGER_CUH
#define GPUQUEUEMANAGER_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include "CudaMatrix.h"
#include "Config.h"
#include "utils.h"
#include <thrust/execution_policy.h>
#include <mutex>


// 队列管理类（单例模式）  占用显存32MB
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

    // 更新队列并在必要时进行杂波判断
    void update_queues(const cufftComplex* d_frame) {
        std::lock_guard<std::mutex> lock(mutex_); // 线程安全锁
        launch_update_kernel(d_frame);            // 更新队列数据
        update_count++;                           // 更新计数器加1
        if (update_count >= CLUTTER_QUEUE_SIZE) {
            launch_clutter_kernel();
        }
    }

    // 获取当前杂波判断数组的指针（线程安全）
    bool* get_clutter() {
        std::lock_guard<std::mutex> lock(clutter_mutex_); // 保护 d_clutter 访问
        return d_clutter;
    }

    // 线程安全的 d_clutter 读取方法（拷贝到线程）
    void get_clutter_copy(bool* device_buffer, size_t size) {
        if (size != WAVE_NUM * NFFT) {
            throw std::runtime_error("Invalid buffer size for d_clutter copy");
        }
        std::lock_guard<std::mutex> lock(clutter_mutex_);
        cudaMemcpy(device_buffer, d_clutter, WAVE_NUM * NFFT * sizeof(bool), cudaMemcpyDeviceToDevice);
    }

private:
    cufftComplex* d_queues = nullptr;       // 显存中的0速通道队列数据
    cufftComplex* d_queues_speed = nullptr; // 显存中的20个速度通道队列数据
    int* d_indices = nullptr;               // 显存中的队列索引
    bool* d_clutter = nullptr;              // 当前杂波判断数组
    int update_count;                       // 更新计数器
    std::mutex mutex_;                      // 更新队列的互斥锁
    std::mutex clutter_mutex_;              // 保护 d_clutter 访问的互斥锁

    // 构造函数：私有化以实现单例
    GpuQueueManager() {
        cudaMalloc(&d_queues, WAVE_NUM * NFFT * CLUTTER_QUEUE_SIZE * sizeof(cufftComplex));
        cudaMalloc(&d_queues_speed, WAVE_NUM * NFFT * SPEED_CHANNELS * CLUTTER_QUEUE_SIZE * sizeof(cufftComplex));
        cudaMalloc(&d_indices, WAVE_NUM * NFFT * sizeof(int));
        cudaMalloc(&d_clutter, WAVE_NUM * NFFT * sizeof(bool));
        cudaMemset(d_indices, 0, WAVE_NUM * NFFT * sizeof(int));
        cudaMemset(d_clutter, 0, WAVE_NUM * NFFT * sizeof(bool));
        update_count = 0;
    }

    // 析构函数：释放显存资源
    ~GpuQueueManager() {
        cudaFree(d_queues);
        cudaFree(d_queues_speed);
        cudaFree(d_indices);
        cudaFree(d_clutter);
    }

    // 启动更新队列的 CUDA 内核
    void launch_update_kernel(const cufftComplex* d_frame);

    // 启动杂波判断的 CUDA 内核
    void launch_clutter_kernel();
};


#endif //GPUQUEUEMANAGER_CUH
