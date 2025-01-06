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
#include "SendVideo.h"  // 包含 SendVideo 的定义
#include "utils.h"
#include <chrono>
#include <string>

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
    std::vector<unsigned char *> threadsMemory;              // 每个线程独立的显存
    std::vector<std::vector<int>> headPositions;    // 每个线程独立空间中，packet的起始地址
    std::vector<cudaStream_t> streams;              // 异步拷贝的时候用，现在没用
    std::vector<size_t> currentPos;                 // 记录每个包头的位置
    size_t currentAddrOffset;                       // 记录当前数据复制的偏移
    int cur_thread_id;                              // 当前线程id (0~numThreads-1)
    std::vector<bool> processingFlags;              // 线程是否正在运行
    bool stop;                                      //控制所有线程结束
    time_point<system_clock> circleStartTime;

    std::vector<std::condition_variable> conditionVariables; // 条件变量
    std::vector<std::mutex> mutexes;                // 互斥锁
    SharedQueue *sharedQueue;
    bool inPacket;                                  // 记录是否正在一个脉组中
    unsigned int prevSeqNum;
    unsigned int prevIndexValue;                    // 上一个packet相对于1GB的起始地址
    uint64_t uint64Pattern;
    double Bandwidth;
    double pulseWidth;

    char timebuf[100];
    ofstream logFile;

    int numSamples;             // 脉压采样点数
    CudaMatrix PcCoefMatrix;
    SendVideo sender;

    void threadLoop(int threadID);

    void copyToThreadMemory();

    void notifyThread(int threadID);

    void waitForProcessingSignal(int threadID);

    void processData(int threadID, cufftComplex *pComplex, vector<CudaMatrix> &matrices, int *d_headPositions,
                     vector<CudaMatrix> &CFAR_res, vector<CudaMatrix> &Max_res, cufftComplex *pMaxRes_d,
                     cufftComplex *pMaxRes_h, cufftHandle &pcPlan, cufftHandle &rowPlan, cufftHandle &colPlan);

    void memcpyDataToThread(unsigned int startAddr, unsigned int endAddr);

    void processPulseGroupData(int threadID, vector<CudaMatrix> &matrices, vector<CudaMatrix> &CFAR_res,
                               vector<CudaMatrix> &Max_res, int rangeNum, cufftHandle &pcPlan,
                               cufftHandle &rowPlan, cufftHandle &colPlan);

    void allocateThreadMemory();

    void freeThreadMemory();

    void generatePCcoefMatrix(unsigned char *rawMessage, cufftHandle &pcPlan, cudaStream_t _stream);

};

__global__ void processKernel(unsigned char *threadsMemory, cufftComplex *pComplex,
                              const int *headPositions, int numHeads, int rangeNum);

#endif // THREADPOOL_H
