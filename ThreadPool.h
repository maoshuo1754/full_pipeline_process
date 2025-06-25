#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <fstream>
#include <ctime>
#include "memory"
#include "SharedQueue.h"      // 包含 SharedQueue 的定义
#include "SendVideo.h"  // 包含 SendVideo 的定义
#include "WaveGroupProcessor.cuh"
#include "utils.h"
#include <chrono>

using namespace std;
using namespace std::chrono;

class ThreadPool {
public:
    ThreadPool(size_t numThreads, SharedQueue *sharedQueue);

    ~ThreadPool();

    void run();

private:
    size_t numThreads;
    std::vector<std::thread> threads;               // 线程
    std::vector<unsigned char *> threadsMemory;     // 每个线程独立的显存
    std::vector<std::vector<int>> headPositions;    // 每个线程独立空间中，packet的起始地址
    std::vector<cudaStream_t> streams;              // 异步拷贝的时候用，现在没用
    std::vector<size_t> currentPos;                 // 记录每个包头的位置
    int cur_thread_id;                              // 当前线程id (0~numThreads-1)
    std::vector<bool> processingFlags;              // 线程是否正在运行
    bool stop;                                      // 控制所有线程结束

    std::vector<std::unique_ptr<WaveGroupProcessor>> waveGroupProcessors;

    std::vector<std::condition_variable> conditionVariables; // 条件变量
    std::vector<std::mutex> mutexes;                // 互斥锁
    SharedQueue *sharedQueue;
    bool inPacket;                                  // 记录是否正在一个脉组中
    unsigned int prevSeqNum;
    unsigned int prevIndexValue;                    // 上一个packet相对于1GB的起始地址
    uint64_t uint64Pattern;

    ofstream debugFile;
    string debug_folder_path;
    char timebuf[100];
    ofstream logFile;
    DataRateTracker dataRateTracker;

    // 发送对象以及控制发送顺序的数据结构
    SendVideo sender;
    std::map<int, RadarParams*> resultMap;
    std::mutex mapMutex;
    std::condition_variable sender_cv_;

    void threadLoop(int threadID);

    void copyToThreadMemory();

    void notifyThread(int threadID);

    void waitForProcessingSignal(int threadID);

    void processData(std::unique_ptr<WaveGroupProcessor>& waveGroupProcessor_, int threadID);

    void memcpyDataToThread(unsigned int startAddr, unsigned int endAddr);

    void freeThreadMemory();

    void senderThread();
};

#endif // THREADPOOL_H
