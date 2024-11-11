#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include "queue.h" // 包含 SharedQueue 的定义

class ThreadPool {
public:
    ThreadPool(size_t numThreads, SharedQueue* sharedQueue);
    ~ThreadPool();

    void run();

private:
    std::vector<std::thread> threads;
    std::vector<std::condition_variable> conditionVariables;
    std::vector<std::mutex> mutexes;
    std::vector<bool> processingFlags;
    bool stop;
    SharedQueue* sharedQueue;

    void threadLoop(int threadID);
    bool dataReadyCondition();
    void copyToThreadMemory(int threadID);
    void notifyThread(int threadID);
    void waitForProcessingSignal(int threadID);
    void processCUDAData(void* cudaMemory, int threadID);
};

#endif // THREADPOOL_H
