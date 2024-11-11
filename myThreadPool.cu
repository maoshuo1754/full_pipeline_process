#include "myThreadPool.h"
#include <cuda_runtime.h>
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads, SharedQueue* sharedQueue)
        : stop(false), sharedQueue(sharedQueue), processingFlags(numThreads, false),
          conditionVariables(numThreads), mutexes(numThreads) { // 初始化 conditionVariables 和 mutexes
    // 创建并初始化线程
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
    }
}

ThreadPool::~ThreadPool() {
    stop = true;
    for (auto& cv : conditionVariables) cv.notify_all(); // 通知所有线程退出
    for (std::thread& thread : threads) {
        if (thread.joinable()) thread.join();
    }
}

void ThreadPool::run() {
    while (!stop) {
        if (dataReadyCondition()) {
            for (size_t i = 0; i < threads.size(); ++i) {
                copyToThreadMemory(i);   // 复制数据块到线程的 CUDA 内存
                notifyThread(i);         // 通知线程开始处理
            }
        }
    }
}

void ThreadPool::threadLoop(int threadID) {
    // 为每个线程分配 2GB 的 CUDA 内存
    void* cudaMemory;
    cudaMalloc(&cudaMemory, 2 * 1024 * 1024 * 1024); // 2GB

    while (!stop) {
        waitForProcessingSignal(threadID);

        if (stop) break; // 退出循环

        processCUDAData(cudaMemory, threadID); // 处理 CUDA 内存中的数据

        // 处理完毕后重置标志
        {
            std::lock_guard<std::mutex> lock(mutexes[threadID]);
            processingFlags[threadID] = false;
        }
    }

    cudaFree(cudaMemory);
}

bool ThreadPool::dataReadyCondition() {
    // 根据 sharedQueue 定义数据就绪条件
    return /* 根据 sharedQueue 的情况返回条件 */;
}

void ThreadPool::copyToThreadMemory(int threadID) {
    // 从 sharedQueue 复制较大的数据块（例如 1GB）到每个线程的 CUDA 内存
    char* dataPtr = &sharedQueue->buffer[sharedQueue->read_index * BLOCK_SIZE];
    void* cudaMemory; // 特定线程的 CUDA 内存
    cudaMemcpy(cudaMemory, dataPtr, /* 大小 */, cudaMemcpyHostToDevice);
    sharedQueue->read_index = (sharedQueue->read_index + 1) % QUEUE_SIZE;
}

void ThreadPool::notifyThread(int threadID) {
    // 设置处理标志并通知线程
    {
        std::lock_guard<std::mutex> lock(mutexes[threadID]);
        processingFlags[threadID] = true;
    }
    conditionVariables[threadID].notify_one();
}

void ThreadPool::waitForProcessingSignal(int threadID) {
    std::unique_lock<std::mutex> lock(mutexes[threadID]);
    conditionVariables[threadID].wait(lock, [this, threadID]() {
        return processingFlags[threadID] || stop;
    });
}

void ThreadPool::processCUDAData(void* cudaMemory, int threadID) {
    // 实现 CUDA 内核处理逻辑
    // 例如： kernel<<<grid, block>>>(cudaMemory, threadID);
}
