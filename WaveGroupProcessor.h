#ifndef WAVE_GROUP_PROCESSOR_H
#define WAVE_GROUP_PROCESSOR_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include "CudaMatrix.h"
#include "Config.h"

class WaveGroupProcessor {
public:
    WaveGroupProcessor(int waveNum, int pulseNum, int rangeNum, cudaStream_t stream);
    ~WaveGroupProcessor();

    // 禁止拷贝和赋值
    WaveGroupProcessor(const WaveGroupProcessor&) = delete;
    WaveGroupProcessor& operator=(const WaveGroupProcessor&) = delete;

    // 数据接口
    cufftComplex* deviceData() { return d_data_; }
    cufftComplex* maxResults() { return d_max_results_; }
    int* speedChannels() { return d_speed_channels_; }

    // 处理流程
    void unpackData(unsigned char* rawData, const int* headPositions, int numHeads);
    void processPulseCompression(const CudaMatrix& pcCoefMatrix);
    void processCoherentIntegration();
    void processCFAR(double Pfa, int numGuard, int numRef, int leftBound, int rightBound);
    void processMaxSelection();

    // 资源访问
    cufftHandle rowPlan() const { return row_plan_; }
    cufftHandle colPlan() const { return col_plan_; }
    cufftHandle pcPlan() const { return pc_plan_; }

private:
    // 三维数据维度
    const int wave_num_;
    const int pulse_num_;
    const int range_num_;
    
    // CUDA资源
    cudaStream_t stream_;
    cufftHandle row_plan_;   // 行FFT
    cufftHandle col_plan_;   // 列FFT
    cufftHandle pc_plan_;    // 脉压FFT

    // 设备内存
    cufftComplex* d_data_;           // 原始数据 (wave_num_ x pulse_num_ x range_num_)
    cufftComplex* d_max_results_;    // 选大结果 (wave_num_ x range_num_)
    int* d_speed_channels_;          // 速度通道 (wave_num_ x range_num_)

    // 中间结果
    std::vector<CudaMatrix> cfars_;  // CFAR结果
    std::vector<CudaMatrix> temps_;  // 临时存储

    void setupFFTPlans();
    void allocateDeviceMemory();
    void freeDeviceMemory();
};

#endif // WAVE_GROUP_PROCESSOR_H