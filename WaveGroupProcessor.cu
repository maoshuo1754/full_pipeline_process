#include "WaveGroupProcessor.h"
#include "utils.h"
#include "SharedQueue.h"
#include <vector>

cufftHandle WaveGroupProcessor::pc_plan_ = 0;            // 脉压FFT，用于对下面两个系数做脉压
cufftComplex* WaveGroupProcessor::d_pc_coeffs_ = nullptr;      // 脉压系数    (1 x range_num_)
cufftComplex* WaveGroupProcessor::d_cfar_coeffs_ = nullptr;    // cfar系数   (1 x range_num_)


WaveGroupProcessor::WaveGroupProcessor(int waveNum, int pulseNum, int rangeNum)
    : wave_num_(waveNum),
      pulse_num_(pulseNum),
      range_num_(rangeNum) {
    allocateDeviceMemory();
    setupFFTPlans();
}

WaveGroupProcessor::~WaveGroupProcessor() {
    cleanup();
    freeDeviceMemory();
    checkCufftErrors(cufftDestroy(row_plan_));
    checkCufftErrors(cufftDestroy(col_plan_));
    // checkCufftErrors(cufftDestroy(pc_plan_));
}

void WaveGroupProcessor::setupFFTPlans() {
    // 创建流
    checkCudaErrors(cudaStreamCreate(&stream_));

    // 行FFT (批量处理)

    checkCufftErrors(cufftPlan1d(&row_plan_, range_num_, CUFFT_C2C, wave_num_ * pulse_num_));
    checkCufftErrors(cufftSetStream(row_plan_, stream_));

    // 列FFT (多行处理)
    int rank = 1;
    int n[] = {pulse_num_};
    int inembed[] = {range_num_};
    checkCufftErrors(cufftPlanMany(&col_plan_, rank, n,
                  inembed, range_num_, 1,
                  inembed, range_num_, 1,
                  CUFFT_C2C, range_num_));
    checkCufftErrors(cufftSetStream(col_plan_, stream_));
}

void WaveGroupProcessor::allocateDeviceMemory() {
    const size_t total_size = wave_num_ * pulse_num_ * range_num_;
    currentAddrOffset = 0;
    checkCudaErrors(cudaMalloc(&d_unpack_data_, THREADS_MEM_SIZE));
    checkCudaErrors(cudaMalloc(&d_headPositions_, sizeof(int) * pulse_num_ * 1.1));
    checkCudaErrors(cudaMalloc(&d_data_, sizeof(cufftComplex) * total_size));
    checkCudaErrors(cudaMalloc(&d_cfar_res_, sizeof(cufftComplex) * total_size));
    checkCudaErrors(cudaMalloc(&d_max_results_, sizeof(float) * wave_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_speed_channels_, sizeof(int) * wave_num_ * range_num_));
}

void WaveGroupProcessor::freeDeviceMemory() {
    checkCudaErrors(cudaFree(d_unpack_data_));
    checkCudaErrors(cudaFree(d_headPositions_));
    checkCudaErrors(cudaFree(d_data_));
    checkCudaErrors(cudaFree(d_cfar_res_));
    checkCudaErrors(cudaFree(d_max_results_));
    checkCudaErrors(cudaFree(d_speed_channels_));
}

void WaveGroupProcessor::cleanup() {
    checkCudaErrors(cudaFree(d_pc_coeffs_));
    checkCudaErrors(cudaFree(d_cfar_coeffs_));
    checkCufftErrors(cufftDestroy(pc_plan_));
}

int WaveGroupProcessor::copyRawData(const uint8_t* h_raw_data, size_t data_size)  {

    if ((currentAddrOffset + data_size) <= THREADS_MEM_SIZE) {
        checkCudaErrors(cudaMemcpyAsync(d_unpack_data_ + currentAddrOffset,
                                 h_raw_data,
                                 data_size,
                                 cudaMemcpyHostToDevice,
                                 stream_));
        currentAddrOffset += data_size;
        return 0;
    }

    currentAddrOffset = 0;
    return -1;
}

void WaveGroupProcessor::getPackegeHeader(uint8_t* h_raw_data, size_t data_size) {
    checkCudaErrors(cudaMemcpyAsync(h_raw_data, d_unpack_data_, data_size, cudaMemcpyDeviceToHost, stream_));
}

void WaveGroupProcessor::getCoef(std::vector<cufftComplex>& pcCoef, std::vector<cufftComplex>& cfarCoef) {

    checkCufftErrors(cufftPlan1d(&pc_plan_, NFFT, CUFFT_C2C, 1));

    checkCudaErrors(cudaMalloc(&d_pc_coeffs_, NFFT * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_cfar_coeffs_, NFFT * sizeof(cufftComplex)));
    checkCudaErrors(cudaMemcpy(d_pc_coeffs_, pcCoef.data(), NFFT * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cfar_coeffs_, cfarCoef.data(), NFFT * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    checkCufftErrors(cufftExecC2C(pc_plan_, d_pc_coeffs_, d_pc_coeffs_, CUFFT_FORWARD));
    checkCufftErrors(cufftExecC2C(pc_plan_, d_cfar_coeffs_, d_cfar_coeffs_, CUFFT_FORWARD));

}

void WaveGroupProcessor::getResult(float* h_max_results_, int* h_speed_channels_) {
    // 选大结果拷贝回内存
    checkCudaErrors(cudaMemcpyAsync(h_max_results_, d_max_results_, sizeof(float) * WAVE_NUM * NFFT,
                            cudaMemcpyDeviceToHost,
                            stream_));

    // 速度通道拷贝回内存
    checkCudaErrors(cudaMemcpyAsync(h_speed_channels_, d_speed_channels_, sizeof(int) * WAVE_NUM * NFFT,
                                    cudaMemcpyDeviceToHost,
                                    stream_));
}

void WaveGroupProcessor::unpackData(const int* headPositions) {
    currentAddrOffset = 0;

    checkCudaErrors(cudaMemcpyAsync(d_headPositions_, headPositions, pulse_num_ * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream_));

    dim3 gridDim1(wave_num_, (range_num_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, pulse_num_);

    unpackKernel3D<<<gridDim1, CUDA_BLOCK_SIZE, 0, stream_>>>(
        d_unpack_data_, d_data_, d_headPositions_, PULSE_NUM, RANGE_NUM);


}

void WaveGroupProcessor::streamSynchronize() {
    cudaStreamSynchronize(stream_);
}

void WaveGroupProcessor::processPulseCompression(int numSamples) {
    int size = wave_num_ * pulse_num_ * range_num_;
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (size + blockSize - 1) / blockSize;

    // fft
    checkCufftErrors(cufftExecC2C(row_plan_, d_data_, d_data_, CUFFT_FORWARD));
    // .*
    rowWiseMulKernel<<<gridSize, blockSize, 0, stream_>>>(d_data_, d_pc_coeffs_, wave_num_ * pulse_num_, range_num_);
    // ifft
    checkCufftErrors(cufftExecC2C(row_plan_, d_data_, d_data_, CUFFT_INVERSE));

    this->streamSynchronize();
    writeComplexToFile(d_pc_coeffs_, 1, range_num_, "pccoef.txt");
    writeComplexToFile(d_data_, pulse_num_, range_num_, "2.txt");

    // 设置线程块和网格大小
    int nrows = wave_num_ * pulse_num_;
    int blocksPerGrid = (nrows + blockSize - 1) / blockSize;

    // 启动kernel
    int startIdx = numSamples;
    int endIdx = startIdx + RANGE_NUM - 1;
    moveAndZeroKernel<<<blocksPerGrid, blockSize, 0, stream_>>>(d_data_, nrows, range_num_, startIdx, endIdx);


}

void WaveGroupProcessor::processCoherentIntegration(float scale) {
    // 执行行FFT
    for (int w = 0; w < wave_num_; ++w) {
        cufftComplex* wavePtr = d_data_ + w * pulse_num_ * range_num_;
        checkCufftErrors(cufftExecC2C(col_plan_, wavePtr, wavePtr, CUFFT_FORWARD));
    }

    // 归一化，抵消脉压增益和列fft增益
    int size = wave_num_ * pulse_num_ * range_num_;
    thrust::device_ptr<cufftComplex> thrust_data(d_data_);
    auto exec_policy = thrust::cuda::par.on(stream_);
    thrust::transform(exec_policy, thrust_data, thrust_data + size, thrust_data, ScaleFunctor(scale));
}

void WaveGroupProcessor::processCFAR() {
    // .^2
    int size = wave_num_ * pulse_num_ * range_num_;
    thrust::device_ptr<cufftComplex> thrust_data(d_data_);
    auto exec_policy = thrust::cuda::par.on(stream_);
    thrust::transform(exec_policy, thrust_data, thrust_data + size, thrust_data, SquareFunctor());

    // fft
    checkCufftErrors(cufftExecC2C(row_plan_, d_data_, d_cfar_res_, CUFFT_FORWARD));
    // .*
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (size + blockSize - 1) / blockSize;
    rowWiseMulKernel<<<gridSize, blockSize, 0, stream_>>>(d_cfar_res_, d_cfar_coeffs_, wave_num_ * pulse_num_, range_num_);
    // ifft
    checkCufftErrors(cufftExecC2C(row_plan_, d_cfar_res_, d_cfar_res_, CUFFT_INVERSE));
    int cfarKernelSize = 2 * numGuardCells + 2 * numRefCells + 1;
    int startIdx = floor((cfarKernelSize - 1) / 2);
    int endIdx = startIdx + RANGE_NUM;

    // 左移抵消卷积扩展
    int nrows = wave_num_ * pulse_num_;
    int blocksPerGrid = (nrows + blockSize - 1) / blockSize;
    moveAndZeroKernel<<<blocksPerGrid, blockSize, 0, stream_>>>(d_cfar_res_, nrows, range_num_, startIdx, endIdx);

    // 根据alpha计算噪底
    double alpha = numRefCells * 2 * (pow(Pfa, -1.0 / (numRefCells * 2)) - 1);
    thrust::device_ptr<cufftComplex> cfar_data(d_cfar_res_);
    thrust::transform(exec_policy, cfar_data, cfar_data + size, cfar_data, ScaleFunctor(alpha/2.0/numRefCells/pulse_num_));


    // 对比噪底选结果
    cmpKernel<<<gridSize, blockSize, 0, stream_>>>(d_data_, d_cfar_res_, wave_num_ * pulse_num_, range_num_);

    thrust::transform(exec_policy, thrust_data, thrust_data + size, thrust_data, ScaleFunctor(1.0f/normFactor));
    // this->streamSynchronize();
    // writeComplexToFile(d_data_, pulse_num_, range_num_, "2.txt");

}

void WaveGroupProcessor::processMaxSelection() {

    dim3 blockDim_(CUDA_BLOCK_SIZE);
    dim3 gridDim_((range_num_ + blockDim_.x - 1) / blockDim_.x);

    for (int w = 0; w < wave_num_; ++w) {
        auto* cfarPtr = d_data_ + w * pulse_num_ * range_num_;
        float* maxPtr = d_max_results_ + w * range_num_;
        int* speedPtr = d_speed_channels_ + w * range_num_;
        maxKernel<<<gridDim_, blockDim_, 0, stream_>>>(cfarPtr, maxPtr, speedPtr, pulse_num_, range_num_);
    }
}