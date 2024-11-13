#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include "memory"
#include "queue.h" // 包含 SharedQueue 的定义

#define THREADS_MEM_SIZE  (2ULL * 1024 * 1024 * 1024)  // 每个线程的空间2GB

using namespace std;

class ThreadPool {
public:
    ThreadPool(size_t numThreads, SharedQueue* sharedQueue);
    ~ThreadPool();

    void run();

private:
    size_t numThreads;
    std::vector<std::thread> threads;
    std::vector<char*> threadsMemory;
    std::vector<std::vector<size_t>> headPositions;
    std::vector<cudaStream_t> streams;
    std::vector<size_t> currentPos;
    std::vector<size_t> currentAddrOffset;   // 记录当前数据复制的偏移
    int cur_thread_id;
    bool inPacket;  // 记录是否正在一个脉组中
    std::vector<std::condition_variable> conditionVariables;
    std::vector<std::mutex> mutexes;
    std::vector<bool> processingFlags;
    bool stop;   //控制所有线程结束
    SharedQueue* sharedQueue;
    unsigned int prevSeqNum;
    bool unEnd;
    uint64_t uint64Pattern;



    void allocateThreadMemory();
    void freeThreadMemory();
    void threadLoop(int threadID);
    void copyToThreadMemory();
    void notifyThread(int threadID);
    void waitForProcessingSignal(int threadID);
//    void processCUDAData(void* cudaMemory, int threadID);
//    void processData(int threadID);
    static unsigned int FourChars2Uint(const char *startAddr);

    void processData(int threadID, size_t *d_headPositions, bool *d_result);
};

#endif // THREADPOOL_H
