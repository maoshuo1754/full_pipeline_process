#include "WaveGroupProcessor.cuh"
#include "utils.h"
#include "SharedQueue.h"
#include <vector>
#include "kelnels.cuh"
#include "nlohmann/json.hpp"


WaveGroupProcessor::WaveGroupProcessor(int waveNum, int pulseNum, int rangeNum)
    : wave_num_(waveNum),
      pulse_num_(pulseNum),
      range_num_(rangeNum),
      coef_is_initialized_(false),
      gpu_manager(GpuQueueManager::getInstance())
{
    allocateDeviceMemory();
    setupFFTPlans();
}

WaveGroupProcessor::~WaveGroupProcessor() {

    freeDeviceMemory();
    checkCufftErrors(cufftDestroy(row_plan_));
    checkCufftErrors(cufftDestroy(col_plan_));
    checkCufftErrors(cufftDestroy(pc_plan_));
}

void WaveGroupProcessor::setupFFTPlans() {
    // 创建流
    checkCudaErrors(cudaStreamCreate(&stream_));
    exec_policy_ = thrust::cuda::par.on(stream_);
    // 行FFT (批量处理)
    checkCufftErrors(cufftPlan1d(&pc_plan_, NFFT, CUFFT_C2C, 1));
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
    checkCudaErrors(cudaMalloc(&d_pc_coeffs_, range_num_ * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_cfar_coeffs_, range_num_ * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_is_masked_, wave_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_unpack_data_, THREADS_MEM_SIZE));
    checkCudaErrors(cudaMalloc(&d_headPositions_, sizeof(int) * pulse_num_ * 1.1));
    checkCudaErrors(cudaMalloc(&d_data_, sizeof(cufftComplex) * total_size));
    checkCudaErrors(cudaMalloc(&d_cfar_res_, sizeof(cufftComplex) * total_size));
    checkCudaErrors(cudaMemset(d_cfar_res_, 0, sizeof(cufftComplex) * total_size));
    checkCudaErrors(cudaMalloc(&d_max_results_, sizeof(float) * wave_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_speed_channels_, sizeof(int) * wave_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_detect_rows_, sizeof(int) * pulse_num_));
    thrust_data_ = thrust::device_ptr<cufftComplex>(d_data_);
    thrust_cfar_ = thrust::device_ptr<cufftComplex>(d_cfar_res_);
}

void WaveGroupProcessor::freeDeviceMemory() {
    checkCudaErrors(cudaFree(d_pc_coeffs_));
    checkCudaErrors(cudaFree(d_cfar_coeffs_));
    checkCudaErrors(cudaFree(d_is_masked_));
    checkCufftErrors(cufftDestroy(pc_plan_));
    checkCudaErrors(cudaFree(d_unpack_data_));
    checkCudaErrors(cudaFree(d_headPositions_));
    checkCudaErrors(cudaFree(d_data_));
    checkCudaErrors(cudaFree(d_cfar_res_));
    checkCudaErrors(cudaFree(d_max_results_));
    checkCudaErrors(cudaFree(d_speed_channels_));
    checkCudaErrors(cudaFree(d_detect_rows_));
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

cufftComplex* WaveGroupProcessor::getData()
{
    return d_data_;
}

void WaveGroupProcessor::getCoef(std::vector<cufftComplex>& pcCoef, std::vector<cufftComplex>& cfarCoef, std::vector<int> &detect_rows) {
    if (coef_is_initialized_) {
        return;
    }
    coef_is_initialized_ = true;
    checkCudaErrors(cudaMemcpy(d_pc_coeffs_, pcCoef.data(), NFFT * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cfar_coeffs_, cfarCoef.data(), NFFT * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    detect_rows_num_ = detect_rows.size();
    checkCudaErrors(cudaMemcpy(d_detect_rows_, detect_rows.data(), detect_rows_num_ * sizeof(int), cudaMemcpyHostToDevice));

    checkCufftErrors(cufftExecC2C(pc_plan_, d_pc_coeffs_, d_pc_coeffs_, CUFFT_FORWARD));
    checkCufftErrors(cufftExecC2C(pc_plan_, d_cfar_coeffs_, d_cfar_coeffs_, CUFFT_FORWARD));

    int mask_size = WAVE_NUM * NFFT;
    bool* h_isMasked = new bool[mask_size];
    memset(h_isMasked, 0, mask_size);

    for (const auto& region: clutterRegions) {
        for (int wave = region.waveStartIdx; wave < region.waveEndIdx; wave++) {
            float startRange = region.startRange;
            float endRange = region.endRange;
            double delta_range = c_speed / Fs / 2.0;
            int startIdx = static_cast<int>(startRange / delta_range) + range_correct;
            int endIdx = static_cast<int>(endRange / delta_range) + range_correct;
            assert(startRange < endRange);
            assert(startIdx < endIdx);
            assert(wave >= 0 && wave < WAVE_NUM);
            assert(startIdx >= 0 && startIdx < NFFT);
            assert(endIdx >= 0 && endIdx < NFFT);
            for (int i = startIdx; i < endIdx; i++) {
                h_isMasked[wave * NFFT + i] = true;
            }
        }
    }
    checkCudaErrors(cudaMemcpy(d_is_masked_, h_isMasked, mask_size, cudaMemcpyHostToDevice));
    delete[] h_isMasked;
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

void WaveGroupProcessor::resetAddr()
{
    currentAddrOffset = 0;
}

void WaveGroupProcessor::unpackData(const int* headPositions) {
    checkCudaErrors(cudaMemcpyAsync(d_headPositions_, headPositions, pulse_num_ * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream_));

    dim3 gridDim1(wave_num_, (range_num_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, pulse_num_);

    unpackKernel3D<<<gridDim1, CUDA_BLOCK_SIZE, 0, stream_>>>(
        d_unpack_data_, d_data_, d_headPositions_, PULSE_NUM, RANGE_NUM);

    // this->streamSynchronize();
    // writeComplexToFile(d_data_ + pulse_num_ * range_num_, pulse_num_, range_num_, "data2.txt");
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
}

void WaveGroupProcessor::processCoherentIntegration(float scale) {
    // 执行列FFT
    dim3 blockDim_(CUDA_BLOCK_SIZE);
    dim3 gridDim_((range_num_ + blockDim_.x - 1) / blockDim_.x);

    for (int w = 0; w < wave_num_; ++w) {
        cufftComplex* wavePtr = d_data_ + w * pulse_num_ * range_num_;
        checkCufftErrors(cufftExecC2C(col_plan_, wavePtr, wavePtr, CUFFT_FORWARD));
        fftshift_columns_inplace_kernel<<<gridDim_, blockDim_, 0, stream_>>>(wavePtr, pulse_num_, range_num_);
    }

    // 抵消脉压增益，同时除以range_num_是ifft之后必须除以ifft才能和matlab结果一样
    int size = wave_num_ * pulse_num_ * range_num_;
    thrust::transform(exec_policy_, thrust_data_, thrust_data_ + size, thrust_data_, ScaleFunctor(scale / range_num_ / normFactor));

    // static int count = 0;
    // count++;
    // std::string filename1 = "WaveGroupProcessor_" + std::to_string(count) + ".txt";
    // std::string filename2 = "bool" + std::to_string(count) + ".txt";
    // this->streamSynchronize();
    // writeComplexToFile(d_data_ + 16*pulse_num_*range_num_, pulse_num_, range_num_, filename1);
    // gpu_manager.update_queues(d_data_);
    // gpu_manager.get_clutter_copy(d_is_masked_, wave_num_ * range_num_);
    // this->streamSynchronize();
    // writeBoolToFile(d_is_masked_ + 16*range_num_, 1, range_num_, filename2);
}


void WaveGroupProcessor::processCFAR() {
    // .^2
    int size = wave_num_ * pulse_num_ * range_num_;
    thrust::transform(exec_policy_, thrust_data_, thrust_data_ + size, thrust_data_, SquareFunctor());

    // fft
    checkCufftErrors(cufftExecC2C(row_plan_, d_data_, d_cfar_res_, CUFFT_FORWARD));
    // .*
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (size + blockSize - 1) / blockSize;
    rowWiseMulKernel<<<gridSize, blockSize, 0, stream_>>>(d_cfar_res_, d_cfar_coeffs_, wave_num_ * pulse_num_, range_num_);

    // ifft
    checkCufftErrors(cufftExecC2C(row_plan_, d_cfar_res_, d_cfar_res_, CUFFT_INVERSE));

    thrust::transform(exec_policy_, thrust_cfar_, thrust_cfar_ + size, thrust_cfar_, ScaleFunctor(1.0 / range_num_ ));

    int cfarKernelSize = 2 * numGuardCells + 2 * numRefCells + 1;

    // 用于抵消频域卷积的偏移量
    int offset = floor((cfarKernelSize - 1) / 2);

    // 根据alpha计算噪底
    double alpha = numRefCells * 2 * (pow(Pfa, -1.0 / (numRefCells * 2)) - 1);
    thrust::transform(exec_policy_, thrust_cfar_, thrust_cfar_ + size, thrust_cfar_, ScaleFunctor(alpha/2.0/numRefCells));

    // 对比噪底选结果，(结果开根号)
    cmpKernel<<<gridSize, blockSize, 0, stream_>>>(d_data_, d_cfar_res_, wave_num_ * pulse_num_, range_num_, offset);
}

void WaveGroupProcessor::cfar(int numSamples)  {
    double alpha = (numRefCells * 2 * (pow(Pfa, -1.0 / (numRefCells * 2)) - 1));

    // .^2
    int size = wave_num_ * pulse_num_ * range_num_;
    thrust::transform(exec_policy_, thrust_data_, thrust_data_ + size, thrust_data_, SquareFunctor());

    // Configure the CUDA kernel launch parameters
    int colsPerThread = CFAR_LENGTH; // 每个线程处理的列数
    int threadsPerBlock = range_num_ / colsPerThread; // 每个线程块中的线程数
    int blocksPerRow = (range_num_ + colsPerThread - 1) / colsPerThread / threadsPerBlock; // 每行的线程块数
    dim3 blockDim(threadsPerBlock, 1); // 线程块大小：1 行 x 32 列
    int nrows = wave_num_ * pulse_num_;
    dim3 gridDim(blocksPerRow, nrows); // 网格大小：每行 block 数 x 总行数

    cfarKernel<<<gridDim, blockDim, 0, stream_>>>(d_data_, d_cfar_res_, nrows, range_num_, alpha, numGuardCells,
                                                  numRefCells, numSamples-1, numSamples+RANGE_NUM-200);
}

void WaveGroupProcessor::cfar_by_col()
{
    double alpha = (numRefCells * 2 * (pow(Pfa, -1.0 / (numRefCells * 2)) - 1));
    dim3 blockDim_(CUDA_BLOCK_SIZE);
    dim3 gridDim_((range_num_ + blockDim_.x - 1) / blockDim_.x);

    int size = wave_num_ * pulse_num_ * range_num_;
    thrust::transform(exec_policy_, thrust_data_, thrust_data_ + size, thrust_data_, SquareFunctor());

    for (int w = 0; w < WAVE_NUM; ++w)
    {
        auto* waveDataPtr = d_data_ + w * pulse_num_ * range_num_;
        auto* cfarPtr = d_cfar_res_ + w * pulse_num_ * range_num_;
        cfar_col_kernel<<<gridDim_, blockDim_, 0, stream_>>>(waveDataPtr, cfarPtr, pulse_num_, range_num_, alpha,
                                                             numGuardCells, numRefCells);
    }
}


void WaveGroupProcessor::processMaxSelection() {
    // 使用2D block和grid
    dim3 blockDim_(16, 16);  // 可以根据需要调整block大小
    dim3 gridDim_(
        (range_num_ + blockDim_.x - 1) / blockDim_.x,
        (wave_num_ + blockDim_.y - 1) / blockDim_.y
    );

    // 直接一次调用处理所有wave
    maxKernel2D<<<gridDim_, blockDim_, 0, stream_>>>(
        d_data_,           // 输入数据
        d_max_results_,    // 最大值输出
        d_speed_channels_, // 通道索引输出
        d_detect_rows_,    // 通道范围（row 索引数组）
        detect_rows_num_,  // 通道数量
        pulse_num_,        // 总行数
        range_num_,        // 总列数
        wave_num_          // 总波数
    );
}

