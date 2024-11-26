#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <fstream>
#include <ctime>
#include "memory"
#include "queue.h"      // 包含 SharedQueue 的定义
#include "CudaMatrix.h" // 包含 CudaMatrix 的定义
#include "utils.h"
#include <chrono>


#define THREADS_MEM_SIZE  (512 * 1024 * 1024)  // 存放未解包数据
#define WAVE_NUM 32    // 波束数
#define CAL_WAVE_NUM 16

// TODO: 这里数据更改后需要变
#define NUM_PULSE 256     // 一个脉组中的脉冲数
#define RANGE_NUM 8192   // 一个脉冲中的距离单元数

using namespace std;
using namespace std::chrono;

class ThreadPool {
public:
    ThreadPool(size_t numThreads, SharedQueue* sharedQueue);
    ~ThreadPool();

    void run();

private:
    size_t numThreads;
    std::vector<std::thread> threads;           // 线程
    std::vector<char*> threadsMemory;           // 每个线程独立的显存
    std::vector<std::vector<size_t>> headPositions; // 每个线程独立空间中，packet的起始地址
    std::vector<cudaStream_t> streams;          // 异步拷贝的时候用，现在没用
    std::vector<size_t> currentPos;             // 记录每个包头的位置
    std::vector<size_t> currentAddrOffset;      // 记录当前数据复制的偏移
    int cur_thread_id;                          // 当前线程id (0~numThreads-1)
    std::vector<bool> processingFlags;          // 线程是否正在运行
    bool stop;                                  //控制所有线程结束
    time_point<system_clock> circleStartTime;

    std::vector<std::condition_variable> conditionVariables; // 条件变量
    std::vector<std::mutex> mutexes;            // 互斥锁
    SharedQueue* sharedQueue;
    bool inPacket;                              // 记录是否正在一个脉组中
    unsigned int prevSeqNum;
    unsigned int prevIndexValue; // 上一个packet相对于1GB的起始地址
    uint64_t uint64Pattern;

    char timebuf[100];
    ofstream logFile;

    CudaMatrix PCcoefMatrix;    // 脉压系数矩阵
    int NFFT;                   // 脉压时做fft的点数
    int numSamples;             // 脉压采样点数

    void threadLoop(int threadID);
    void copyToThreadMemory();
    void notifyThread(int threadID);
    void waitForProcessingSignal(int threadID);

    static unsigned int FourChars2Uint(const char *startAddr);

    void processData(int threadID, cufftComplex *pComplex, vector<CudaMatrix> &matrices, size_t *d_headPositions);

    void memcpyDataToThread(unsigned int startAddr, unsigned int endAddr);

    void initPCcoefMatrix();

    void processPulseGroupData(vector<CudaMatrix> &matrices);

    void allocateThreadMemory();

    void freeThreadMemory();

    float TwoChars2float(const char *startAddr);
};

void checkCudaErrors(cudaError_t result);
#endif // THREADPOOL_H
