#ifndef WAVE_GROUP_PROCESSOR_H
#define WAVE_GROUP_PROCESSOR_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include "Config.h"
#include "utils.h"
#include <thrust/execution_policy.h>
#include <mutex>
#include "GpuQueueManager.cuh"
#include <thrust/device_ptr.h>
#include "SharedQueue.h"
using namespace std;

struct RadarParams
{
    bool isInit;
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
        isInit = false;
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
    RadarParams* getParams();
    void getResult();
    void unpackData(const int* headPositions);
    void streamSynchronize();
    void fullPipelineProcess();
    void processPulseCompression();
    void processMTI();
    void processCoherentIntegration(float scale);
    void clutterNoiseClassify();
    void processFFTshift();
    void processClutterMap();
    void processCFAR();
    void cfar(int numSamples);
    void cfar_by_col();
    void processMaxSelection();
    void getRadarParams();
    void saveToDebugFile(int frame, std::ofstream& debugFile);
    void getCoef();
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
    int* d_speed_channels_;          // 每个距离单元选出来的速度   (wave_num_ x range_num_)
    int* d_chnSpeeds;                // 多普勒维度对应的速度
    int* d_detect_rows_;             // 需要检测的通道
    bool* d_clutterMap_masked_;      // 杂波图
    int cur_wave_;                   // 正在处理的波束号

    int detect_rows_num_;            // 需要检测的通道数
    int clutterMap_range_num_;          // 做杂波图的距离单元数

    thrust::device_ptr<cufftComplex> thrust_data_;
    thrust::device_ptr<cufftComplex> thrust_cfar_;

    // 脉压系数和cfar系数
    bool coef_is_initialized_;
    cufftHandle pc_plan_;            // 脉压FFT，用于对下面两个系数做脉压
    cufftComplex* d_pc_coeffs_;      // 脉压系数    (1 x range_num_)
    cufftComplex* d_cfar_coeffs_;    // cfar系数   (1 x range_num_)

    cufftComplex* d_filtered_coeffs_;// 滤波器系数
    bool* d_is_masked_;              // 需要杂波处理的区域

    RadarParams* radar_params_;
    // 杂波区域判断类
    GpuQueueManager& gpu_manager;    // 共享的单例引用

    static void cleanup();
    void setupFFTPlans();
    void allocateDeviceMemory();
    void freeDeviceMemory();
};



#endif // WAVE_GROUP_PROCESSOR_H