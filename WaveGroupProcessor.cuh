#ifndef WAVE_GROUP_PROCESSOR_H
#define WAVE_GROUP_PROCESSOR_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include "CudaMatrix.h"
#include "Config.h"
#include "utils.h"
#include <thrust/execution_policy.h>
#include <mutex>
#include "GpuQueueManager.cuh"


class WaveGroupProcessor {
public:
    WaveGroupProcessor(int waveNum, int pulseNum, int rangeNum);
    ~WaveGroupProcessor();

    // 禁止拷贝和赋值
    WaveGroupProcessor(const WaveGroupProcessor&) = delete;
    WaveGroupProcessor& operator=(const WaveGroupProcessor&) = delete;

    // 处理流程
    int copyRawData(const uint8_t* h_raw_data, size_t data_size);
    void getPackegeHeader(uint8_t* h_rawData, size_t data_size);
    cufftComplex* getData();
    void getResult(float* h_max_results_, int* h_speed_channels_);
    void unpackData(const int* headPositions);
    void streamSynchronize();
    void processPulseCompression(int numSamples);
    void processMTI();
    void processCoherentIntegration(float scale);
    void processFFTshift();
    void processClutterMap();
    void processCFAR();
    void cfar(int numSamples);
    void cfar_by_col();
    void processMaxSelection();
    void getCoef(std::vector<cufftComplex>& pcCoef, std::vector<cufftComplex>& cfarCoef, std::vector<int> &detect_rows, int numSamples);
    void resetAddr();

private:
    // 三维数据维度
    const int wave_num_;
    const int pulse_num_;
    const int range_num_;
    
    // CUDA资源
    cudaStream_t stream_;
    thrust::cuda_cub::execute_on_stream exec_policy_;
    cufftHandle row_plan_;   // 行FFT
    cufftHandle col_plan_;   // 列FFT

    // 设备内存 (显存)
    size_t currentAddrOffset;        // 未解包数据的拷贝偏移量
    uint8_t* d_unpack_data_;         // 未解包数据
    int* d_headPositions_;           // 报文头在d_unpack_data_中的位置，用于解包
    cufftComplex* d_data_;           // 原始数据   (wave_num_ x pulse_num_ x range_num_)
    cufftComplex* d_cfar_res_;       // cfar结果  (wave_num_ x pulse_num_ x range_num_)
    float* d_max_results_;           // 选大结果   (wave_num_ x range_num_)
    int* d_speed_channels_;          // 速度通道   (wave_num_ x range_num_)
    int* d_detect_rows_;             // 需要检测的通道
    bool* d_clutterMap_masked_;      // 杂波图

    int detect_rows_num_;            // 需要检测的通道数
    int clutterMap_range_num_;          // 做杂波图的距离单元数

    thrust::device_ptr<cufftComplex> thrust_data_;
    thrust::device_ptr<cufftComplex> thrust_cfar_;

    // 脉压系数和cfar系数
    bool coef_is_initialized_;
    cufftHandle pc_plan_;            // 脉压FFT，用于对下面两个系数做脉压
    cufftComplex* d_pc_coeffs_;      // 脉压系数    (1 x range_num_)
    cufftComplex* d_cfar_coeffs_;    // cfar系数   (1 x range_num_)
    bool* d_is_masked_;              // 需要杂波处理的区域

    // 杂波区域判断类
    GpuQueueManager& gpu_manager;    // 共享的单例引用

    static void cleanup();
    void setupFFTPlans();
    void allocateDeviceMemory();
    void freeDeviceMemory();
};



#endif // WAVE_GROUP_PROCESSOR_H