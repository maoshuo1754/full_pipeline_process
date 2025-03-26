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
#include "WaveGroupProcessor.cuh"
#include "utils.h"
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;


struct RadarParams
{
    uint8_t* rawMessage;        // 未解包的报文头
    double bandWidth;           // 带宽
    double pulseWidth;          // 脉宽
    double PRT;                 // 脉冲重复时间
    double lambda;              // 波长
    int numSamples;             // 脉压采样点数
    float scale;                // 归一化系数(脉压和ifft之后)
    float* h_max_results_;      // 选大结果 (wave_num_ x range_num_)
    int* h_speed_channels_;     // 速度通道 (wave_num_ x range_num_)
    vector<int> chnSpeeds;      // 速度通道对应的速度
    vector<int> detect_rows;           // 速度范围内的行
    vector<cufftComplex> pcCoef;    // 脉压系数
    vector<cufftComplex> cfarCoef;  // CFAR系数


    RadarParams(): cfarCoef(NFFT, {0, 0}) {
        rawMessage = new uint8_t[2 * DATA_OFFSET];
        h_max_results_ = new float[WAVE_NUM * NFFT];
        h_speed_channels_ = new int[WAVE_NUM * NFFT];
    }

    ~RadarParams() {
        delete[] rawMessage;
        delete[] h_max_results_;
        delete[] h_speed_channels_;
    }

    void getCoef() {
        pcCoef = PCcoef(bandWidth, pulseWidth, Fs, NFFT, hamming_window_enable);
        numSamples = round(pulseWidth * Fs);

        for(int i = 0; i < numRefCells; i++) {
            cfarCoef[i].x = 1.0f;
        }

        int startIdx = numRefCells + numGuardCells * 2 + 1;
        for(int i = startIdx; i < startIdx + numRefCells; i++) {
            cfarCoef[i].x = 1.0f;
        }
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

    RadarParams* radar_params_;
    ofstream debugFile;
    DataRateTracker dataRateTracker;

    SendVideo sender;

    void threadLoop(int threadID);

    void copyToThreadMemory();

    void notifyThread(int threadID);

    void waitForProcessingSignal(int threadID);

    void processData(std::unique_ptr<WaveGroupProcessor>& waveGroupProcessor_, int threadID);

    void getRadarParams(std::unique_ptr<WaveGroupProcessor>& waveGroupProcessor, int frame);

    void memcpyDataToThread(unsigned int startAddr, unsigned int endAddr);

    void freeThreadMemory();

    void generatePCcoefMatrix(unsigned char *rawMessage, cufftHandle &pcPlan, cudaStream_t _stream);

    void writeToDebugFile(unsigned char *rawMessage, const cufftComplex* d_data);

};

__global__ void processKernel(unsigned char *threadsMemory, cufftComplex *pComplex,
                              const int *headPositions, int numHeads, int rangeNum);

#endif // THREADPOOL_H
