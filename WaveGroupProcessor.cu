#include "WaveGroupProcessor.h"
#include "utils.h"
#include "SharedQueue.h"

WaveGroupProcessor::WaveGroupProcessor(int waveNum, int pulseNum, int rangeNum, cudaStream_t stream)
    : wave_num_(waveNum),
      pulse_num_(pulseNum),
      range_num_(rangeNum),
      stream_(stream) {
    allocateDeviceMemory();
    setupFFTPlans();

    // 初始化中间结果矩阵
    for (int i = 0; i < wave_num_; ++i) {
        cfars_.emplace_back(pulse_num_, range_num_);
        temps_.emplace_back(pulse_num_, range_num_);
    }
}

WaveGroupProcessor::~WaveGroupProcessor() {
    freeDeviceMemory();
    cufftDestroy(row_plan_);
    cufftDestroy(col_plan_);
    cufftDestroy(pc_plan_);
}

void WaveGroupProcessor::setupFFTPlans() {
    // 脉压FFT (按行)
    cufftPlan1d(&pc_plan_, range_num_, CUFFT_C2C, pulse_num_);
    cufftSetStream(pc_plan_, stream_);

    // 行FFT (批量处理)
    cufftPlan1d(&row_plan_, range_num_, CUFFT_C2C, pulse_num_);
    cufftSetStream(row_plan_, stream_);

    // 列FFT (多行处理)
    int rank = 1;
    int n[] = {pulse_num_};
    int inembed[] = {range_num_};
    cufftPlanMany(&col_plan_, rank, n,
                  inembed, range_num_, 1,
                  inembed, range_num_, 1,
                  CUFFT_C2C, range_num_);
    cufftSetStream(col_plan_, stream_);
}

void WaveGroupProcessor::allocateDeviceMemory() {
    const size_t total_size = wave_num_ * pulse_num_ * range_num_;
    cudaMalloc(&d_data_, sizeof(cufftComplex) * total_size);
    cudaMalloc(&d_max_results_, sizeof(cufftComplex) * wave_num_ * range_num_);
    cudaMalloc(&d_speed_channels_, sizeof(int) * wave_num_ * range_num_);
}

void WaveGroupProcessor::freeDeviceMemory() {
    cudaFree(d_data_);
    cudaFree(d_max_results_);
    cudaFree(d_speed_channels_);
}

__global__ void unpackKernel3D(unsigned char* raw, cufftComplex* output,
                              const int* heads, int waveNum, int pulseNum, int rangeNum) {
    const int waveIdx = blockIdx.z;
    const int pulseIdx = blockIdx.y;
    const int rangeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rangeIdx >= rangeNum) return;

    const int headOffset = heads[pulseIdx];
    unsigned char* blockStart = raw + headOffset + DATA_OFFSET;
    const int byteOffset = rangeIdx * waveNum * 4 + waveIdx * 4;
    
    output[(waveIdx * pulseNum + pulseIdx) * rangeNum + rangeIdx].x = 
        __int2half_rn(*(int16_t*)(blockStart + byteOffset + 2));
    output[(waveIdx * pulseNum + pulseIdx) * rangeNum + rangeIdx].y = 
        __int2half_rn(*(int16_t*)(blockStart + byteOffset));
}

void WaveGroupProcessor::unpackData(unsigned char* rawData, const int* headPositions, int numHeads) {
    dim3 block(256);
    dim3 grid((range_num_ + block.x - 1) / block.x, pulse_num_, wave_num_);
    unpackKernel3D<<<grid, block, 0, stream_>>>(
        rawData, d_data_, headPositions, wave_num_, pulse_num_, range_num_);
}

__global__ void batchPulseCompression(cufftComplex* data, const cufftComplex* pcCoef,
                                      int waveNum, int pulseNum, int rangeNum) {
    const int waveIdx = blockIdx.z;
    const int pulseIdx = blockIdx.y;
    const int rangeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rangeIdx >= rangeNum) return;

    cufftComplex* waveData = data + waveIdx * pulseNum * rangeNum;
    cufftComplex val = waveData[pulseIdx * rangeNum + rangeIdx];
    cufftComplex coef = pcCoef[rangeIdx];
    
    // 频域相乘
    waveData[pulseIdx * rangeNum + rangeIdx] = cuCmulf(val, coef);
}

void WaveGroupProcessor::processPulseCompression(const CudaMatrix& pcCoefMatrix) {
    // 执行行FFT
    for (int w = 0; w < wave_num_; ++w) {
        cufftComplex* wavePtr = d_data_ + w * pulse_num_ * range_num_;
        cufftExecC2C(row_plan_, wavePtr, wavePtr, CUFFT_FORWARD);
    }

    // 频域相乘
    dim3 block(256);
    dim3 grid((range_num_ + block.x - 1) / block.x, pulse_num_, wave_num_);
    batchPulseCompression<<<grid, block, 0, stream_>>>(
        d_data_, pcCoefMatrix.getData(), wave_num_, pulse_num_, range_num_);

    // 执行逆FFT
    for (int w = 0; w < wave_num_; ++w) {
        cufftComplex* wavePtr = d_data_ + w * pulse_num_ * range_num_;
        cufftExecC2C(row_plan_, wavePtr, wavePtr, CUFFT_INVERSE);
    }
}

// 其他处理函数实现类似，使用三维核函数处理所有波束...