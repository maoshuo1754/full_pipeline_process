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
#include <ipp.h>
using namespace std;

struct RadarParams
{
    bool isInit;
    uint8_t* rawMessage;        // 未解包的报文头
    double bandWidth;           // 带宽
    double pulseWidth;          // 脉宽
    double PRT;                 // 脉冲重复时间
    double lambda;              // 波长
    double distance_resolution; // 距离分辨力
    int numSamples;             // 脉压采样点数
    float scale;                // 归一化系数(脉压和ifft之后)
    float* h_max_results_;      // 选大结果 (wave_num_ x range_num_)
    int* h_speed_channels_;     // 速度通道 (wave_num_ x range_num_)
    vector<int> chnSpeeds;      // 速度通道对应的速度
    map<int, int> speedsMap;    // 速度对应的速度通道
    vector<int> detect_rows;           // 速度范围内的行
    vector<cufftComplex> pcCoef;    // 脉压系数
    vector<cufftComplex> cfarCoef;  // CFAR系数

    float* h_azi_densify_results_;  // 方位加密结果 (wave_num_ x range_num_)
    double* h_azi_theta;             // 方位


    RadarParams(): cfarCoef(NFFT, {0, 0}) {
        isInit = false;
        rawMessage = new uint8_t[2 * DATA_OFFSET];
        h_max_results_ = new float[WAVE_NUM * NFFT];
        h_azi_densify_results_ = new float[WAVE_NUM * NFFT];
        h_azi_theta = new double[azi_densify_crow_num];
        ippsSet_32f(azi_densify_invalid_num, h_azi_densify_results_, WAVE_NUM * NFFT);
        h_speed_channels_ = new int[WAVE_NUM * NFFT];
    }

    ~RadarParams() {
        delete[] rawMessage;
        delete[] h_max_results_;
        delete[] h_azi_densify_results_;
        delete[] h_azi_theta;
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

        for (int i = 0; i < azi_densify_crow_num; i++) {
            // h_azi_theta[azi_densify_crow_num - i - 1] = asind((i - azi_densify_crow_num / 2) * lambda / azi_densify_crow_num / azi_densify_d);
            h_azi_theta[i] = asind((i - azi_densify_crow_num / 2) * lambda / azi_densify_crow_num / azi_densify_d);
            // std::cout << i + 1 << " " << h_azi_theta[i] << std::endl;
        }

    }
};


class FFTProcessor {
public:
    // 构造函数：初始化 FFT 规格
    FFTProcessor(int dim) : dim_(dim) {
        order_ = (int)(log2(dim_)); // 计算 FFT 阶数

        // 获取 FFT 所需的缓冲区大小
        IppStatus status = ippsFFTGetSize_C_32fc(order_, IPP_FFT_NODIV_BY_ANY, ippAlgHintAccurate,
                                                &sizeSpec_, &sizeInit_, &sizeBuf_);
        if (status != ippStsNoErr) {
            throw std::runtime_error("获取 FFT 大小失败");
        }

        // 分配内存
        pSpec_ = (Ipp8u*)ippMalloc(sizeSpec_);
        pInit_ = (Ipp8u*)ippMalloc(sizeInit_);
        pBuf_ = sizeBuf_ ? (Ipp8u*)ippMalloc(sizeBuf_) : nullptr;
        // if (!pSpec_ || !pInit_ || (sizeBuf_ && !pBuf_)) {
        //     throw std::runtime_error("内存分配失败");
        // }

        // 初始化 FFT 规格
        status = ippsFFTInit_C_32fc(&pFFTSpec_, order_, IPP_FFT_NODIV_BY_ANY, ippAlgHintAccurate,
                                    pSpec_, pInit_);
        if (status != ippStsNoErr) {
            throw std::runtime_error("初始化 FFT 失败");
        }
    }

    // 析构函数：释放内存
    ~FFTProcessor() {
        ippFree(pSpec_);
        ippFree(pInit_);
        if (pBuf_) ippFree(pBuf_);
    }

    IppStatus perform_fft_inplace(Ipp32fc* buffer) {
        // 执行就地 FFT，不进行缩放以匹配 MATLAB 的 fft
        return ippsFFTFwd_CToC_32fc_I(buffer, pFFTSpec_, pBuf_);
    }

    // 执行 IFFT 的优化函数
    IppStatus perform_ifft_inplace(Ipp32fc* buffer) {
        // 执行就地 IFFT
        IppStatus status = ippsFFTInv_CToC_32fc_I(buffer, pFFTSpec_, pBuf_);
        if (status != ippStsNoErr) {
            return status;
        }
        // 归一化以匹配 MATLAB 的 ifft（除以 dim_）
        return ippsDivC_32fc_I((Ipp32fc){(float)dim_, 0}, buffer, dim_);
    }

    // fftshift 方法：将零频分量移到数组中心，匹配 MATLAB 的 fftshift
    void fftshift(Ipp32fc* buffer) {
        if (dim_ <= 1) return; // 如果维度小于等于1，无需移位

        int half_dim = dim_ / 2;
        Ipp32fc* temp = (Ipp32fc*)ippMalloc(half_dim * sizeof(Ipp32fc));
        if (!temp) {
            throw std::runtime_error("fftshift 内存分配失败");
        }

        // 将前半部分复制到临时缓冲区
        ippsCopy_32fc(buffer, temp, half_dim);

        // 将后半部分移到前半部分
        ippsCopy_32fc(buffer + half_dim, buffer, dim_ - half_dim);

        // 将临时缓冲区的前半部分复制到后半部分
        ippsCopy_32fc(temp, buffer + (dim_ - half_dim), half_dim);

        ippFree(temp);
    }

private:
    int dim_;                     // 输入的 ifft阶数
    int order_;                  // FFT 阶数
    int sizeSpec_;               // FFT 规格缓冲区大小
    int sizeInit_;               // 初始化缓冲区大小
    int sizeBuf_;                // 工作缓冲区大小
    Ipp8u* pSpec_;               // FFT 规格缓冲区
    Ipp8u* pInit_;               // 初始化缓冲区
    Ipp8u* pBuf_;                // 工作缓冲区
    IppsFFTSpec_C_32fc* pFFTSpec_; // FFT 规格指针
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
    void processClutterMap();
    void processCFAR();
    void cfar(int numSamples);
    void processMaxSelection();
    void processAziDensify();
    void getRadarParams();
    void saveToDebugFile(int frame, std::ofstream& debugFile);
    void saveToDebugFile_new(int frame, std::string debug_folder_path);
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
    int* d_chnSpeeds;                // 多普勒维度对应的速度 (pulse_num_)
    int* d_detect_rows_;             // 需要检测的通道
    bool* d_clutterMap_masked_;      // 杂波图
    int cur_wave_;                   // 正在处理的波束号

    // 主机内存，用于波束加密
    cufftComplex* h_data_after_Integration;  //积累后数据  (wave_num_ x pulse_num_ x range_num_)
    Ipp32fc* h_azi_densify_buffer;      // 方位加密中间变量
    Ipp32f* h_azi_densify_abs_buffer;   // 方位加密求模中间变量

    int detect_rows_num_;            // 需要检测的通道数
    int clutterMap_range_num_;          // 做杂波图的距离单元数

    thrust::device_ptr<cufftComplex> thrust_data_;
    thrust::device_ptr<cufftComplex> thrust_cfar_;

    // 脉压系数和cfar系数
    bool coef_is_initialized_;
    cufftHandle pc_plan_;            // 脉压FFT，用于对下面两个系数做脉压
    cufftComplex* d_pc_coeffs_;      // 脉压系数    (1 x range_num_)
    cufftComplex* d_cfar_coeffs_;    // cfar系数   (1 x range_num_)

    RadarParams* radar_params_;
    // 杂波区域判断类
    GpuQueueManager& gpu_manager;    // 共享的单例引用
    FFTProcessor ifft_processor_32;         // ipp库进行ifft的对象
    FFTProcessor ifft_processor_8192;         // ipp库进行ifft的对象

    static void cleanup();
    void setupFFTPlans();
    void allocateDeviceMemory();
    void freeDeviceMemory();
};



#endif // WAVE_GROUP_PROCESSOR_H