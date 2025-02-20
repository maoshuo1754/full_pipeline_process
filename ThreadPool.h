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
#include "CudaMatrix.h" // 包含 CudaMatrix 的定义
#include "SendVideo.h"  // 包含 SendVideo 的定义
#include "utils.h"
#include <chrono>
#include <string>
#include <sstream>

using namespace std;
using namespace std::chrono;

// 线程池中各个线程独占的资源
// 线程的独立显存拷贝空间和stream不在这，是因为拷贝线程要用
struct ThreadPoolResources {
    int threadID;                   // 线程ID
    cudaStream_t stream;            // cuda流，用于异步处理
    cufftComplex* pComplex_d;       // 解包后的脉组数据

    int* pHeadPositions_d;          // 头位置

    vector<CudaMatrix> IQmatrices;  // 原始 IQ 数据
    vector<CudaMatrix> CFAR_res;    // CFAR 结果
    vector<CudaMatrix> Max_res;     // 选大结果

    cufftComplex* pMaxRes_d;        // 选大结果 (device) ，用这个初始化CudaMatrix矩阵指针
    cufftComplex* pMaxRes_h;        // 选大结果 (host)
    int* pSpeed_d;                  // 选大速度结果 (device 通道数)
    int* pSpeed_h;                  // 选大速度结果 (host 通道数)

    cufftHandle pcPlan;             // 脉压 FFT plan
    cufftHandle rowPlan;            // 按行 FFT plan
    cufftHandle colPlan;            // 按列 FFT plan
    unsigned char rawMessage[DATA_OFFSET]; // 存储报文信息头中的原始信息

    ThreadPoolResources(int threadID_, cudaStream_t stream_) {
        threadID = threadID_;
        stream = stream_;

        memset(rawMessage, 0, sizeof(rawMessage));
        checkCudaErrors(cudaMalloc(&pComplex_d, sizeof(cufftComplex) * WAVE_NUM * NUM_PULSE * NFFT));
        checkCudaErrors(cudaMalloc(&pHeadPositions_d, NUM_PULSE * 1.1 * sizeof(size_t)));
        checkCudaErrors(cudaMalloc(&pMaxRes_d, sizeof(cufftComplex) * WAVE_NUM * NFFT));
        pMaxRes_h = new cufftComplex[WAVE_NUM * NFFT];
        checkCudaErrors(cudaMalloc(&pSpeed_d, sizeof(int) * WAVE_NUM * NFFT));
        pSpeed_h = new int[WAVE_NUM * NFFT];

        // 初始化 IQmatrices 和 CFAR_res
        for (int i = 0; i < WAVE_NUM; i++) {
            IQmatrices.emplace_back(NUM_PULSE, NFFT, pComplex_d + i * NUM_PULSE * NFFT, true);
            CFAR_res.emplace_back(NUM_PULSE, NFFT);
            Max_res.emplace_back(1, NFFT, pMaxRes_d + i * NFFT, true);
        }

        // 创建 cufft plan
        int nrows = NUM_PULSE;
        int ncols = NFFT;
        checkCufftErrors(cufftPlan1d(&pcPlan, ncols, CUFFT_C2C, 1));
        checkCufftErrors(cufftSetStream(pcPlan, stream));

        checkCufftErrors(cufftPlan1d(&rowPlan, ncols, CUFFT_C2C, nrows));
        checkCufftErrors(cufftSetStream(rowPlan, stream));

        checkCufftErrors(cufftPlanMany(&colPlan, 1, &nrows,
                                       &ncols, ncols, 1,
                                       &ncols, ncols, 1,
                                       CUFFT_C2C, NFFT));
        checkCufftErrors(cufftSetStream(colPlan, stream));
    }

    ~ThreadPoolResources() {
        // 释放内存和资源
        checkCudaErrors(cudaFree(pComplex_d));
        checkCudaErrors(cudaFree(pHeadPositions_d));
        checkCudaErrors(cudaFree(pMaxRes_d));
        delete[] pMaxRes_h;
        checkCudaErrors(cudaFree(pSpeed_d));
        delete[] pSpeed_h;

        checkCufftErrors(cufftDestroy(pcPlan));
        checkCufftErrors(cufftDestroy(rowPlan));
        checkCufftErrors(cufftDestroy(colPlan));
    }
};


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
    size_t currentAddrOffset;                       // 记录当前数据复制的偏移
    int cur_thread_id;                              // 当前线程id (0~numThreads-1)
    std::vector<bool> processingFlags;              // 线程是否正在运行
    bool stop;                                      // 控制所有线程结束
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
    vector<int> chnSpeeds;

    char timebuf[100];
    ofstream logFile;
    ofstream debugFile;

    int numSamples;             // 脉压采样点数
    CudaMatrix PcCoefMatrix;
    SendVideo sender;

    void threadLoop(int threadID);

    void copyToThreadMemory();

    void notifyThread(int threadID);

    void waitForProcessingSignal(int threadID);

    void processData(ThreadPoolResources &resources);

    void memcpyDataToThread(unsigned int startAddr, unsigned int endAddr);

    void processPulseGroupData(ThreadPoolResources &resources, int rangeNum);

    void allocateThreadMemory();

    void freeThreadMemory();

    void generatePCcoefMatrix(unsigned char *rawMessage, cufftHandle &pcPlan, cudaStream_t _stream);

    void writeToDebugFile(unsigned char *rawMessage, const cufftComplex* d_data);

};

__global__ void processKernel(unsigned char *threadsMemory, cufftComplex *pComplex,
                              const int *headPositions, int numHeads, int rangeNum);

#endif // THREADPOOL_H
