#include "WaveGroupProcessor.cuh"
#include "utils.h"
#include "SharedQueue.h"
#include <vector>
#include "kelnels.cuh"
#include "nlohmann/json.hpp"
#include "fdacoefs.h"
#include "ThreadPool.h"

WaveGroupProcessor::WaveGroupProcessor(int waveNum, int pulseNum, int rangeNum)
    : wave_num_(waveNum),
      pulse_num_(pulseNum),
      range_num_(rangeNum),
      coef_is_initialized_(false),
      gpu_manager(GpuQueueManager::getInstance())
{
    radar_params_ = new RadarParams();
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
    checkCufftErrors(cufftPlan1d(&row_plan_, range_num_, CUFFT_C2C, pulse_num_));
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
    checkCudaErrors(cudaMalloc(&d_filtered_coeffs_, range_num_ * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_is_masked_, wave_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_clutterMap_masked_, wave_num_ * pulse_num_ * range_num_));
    checkCudaErrors(cudaMemset(d_clutterMap_masked_, 1, wave_num_ * pulse_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_chnSpeeds, pulse_num_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_unpack_data_, THREADS_MEM_SIZE));
    checkCudaErrors(cudaMalloc(&d_headPositions_, sizeof(int) * pulse_num_ * 1.1));
    checkCudaErrors(cudaMalloc(&d_data_, sizeof(cufftComplex) * total_size));
    checkCudaErrors(cudaMalloc(&d_cfar_res_, sizeof(cufftComplex) * pulse_num_ * range_num_));
    checkCudaErrors(cudaMemset(d_cfar_res_, 0, sizeof(cufftComplex) * pulse_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_max_results_, sizeof(float) * wave_num_ * range_num_));
    checkCudaErrors(cudaMemset(d_max_results_, 0, sizeof(float) * wave_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_speed_channels_, sizeof(int) * wave_num_ * range_num_));
    checkCudaErrors(cudaMemset(d_speed_channels_, 0, sizeof(int) * wave_num_ * range_num_));
    checkCudaErrors(cudaMalloc(&d_detect_rows_, sizeof(int) * pulse_num_));

    thrust_cfar_ = thrust::device_ptr<cufftComplex>(d_cfar_res_);
}

void WaveGroupProcessor::freeDeviceMemory() {
    checkCudaErrors(cudaFree(d_pc_coeffs_));
    checkCudaErrors(cudaFree(d_cfar_coeffs_));
    checkCudaErrors(cudaFree(d_filtered_coeffs_));
    checkCudaErrors(cudaFree(d_is_masked_));
    checkCudaErrors(cudaFree(d_clutterMap_masked_));
    checkCudaErrors(cudaFree(d_chnSpeeds));
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

RadarParams* WaveGroupProcessor::getParams()
{
    return radar_params_;
}

void WaveGroupProcessor::getCoef() {
    if (coef_is_initialized_) {
        return;
    }
    double delta_range = c_speed / Fs / 2.0;

    coef_is_initialized_ = true;
    clutterMap_range_num_ = ceil(clutter_map_range / delta_range) + radar_params_->numSamples + range_correct;
    checkCudaErrors(cudaMemcpy(d_pc_coeffs_, radar_params_->pcCoef.data(), NFFT * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cfar_coeffs_, radar_params_->cfarCoef.data(), NFFT * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_chnSpeeds, radar_params_->chnSpeeds.data(), pulse_num_ * sizeof(int), cudaMemcpyHostToDevice));
    detect_rows_num_ = radar_params_->detect_rows.size();
    checkCudaErrors(cudaMemcpy(d_detect_rows_, radar_params_->detect_rows.data(), detect_rows_num_ * sizeof(int), cudaMemcpyHostToDevice));

    checkCufftErrors(cufftExecC2C(pc_plan_, d_pc_coeffs_, d_pc_coeffs_, CUFFT_FORWARD));
    checkCufftErrors(cufftExecC2C(pc_plan_, d_cfar_coeffs_, d_cfar_coeffs_, CUFFT_FORWARD));

    int mask_size = WAVE_NUM * NFFT;
    bool* h_isMasked = new bool[mask_size];
    memset(h_isMasked, 0, mask_size);

    for (const auto& region: clutterRegions) {
        for (int wave = region.waveStartIdx; wave < region.waveEndIdx; wave++) {
            float startRange = region.startRange;
            float endRange = region.endRange;

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

    cufftComplex* h_filtered_coeffs_ = new cufftComplex[range_num_];
    memset(h_filtered_coeffs_, 0, range_num_ * sizeof(cufftComplex));
    for (int i = 0; i < BL; i++)
    {
        h_filtered_coeffs_[i].x = B[i];
    }
    checkCudaErrors(cudaMemcpy(d_filtered_coeffs_, h_filtered_coeffs_, range_num_ * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    checkCufftErrors(cufftExecC2C(pc_plan_, d_filtered_coeffs_, d_filtered_coeffs_, CUFFT_FORWARD));

    delete[] h_filtered_coeffs_;
    delete[] h_isMasked;
}

void WaveGroupProcessor::getResult() {
    // 选大结果拷贝回内存
    checkCudaErrors(cudaMemcpyAsync(radar_params_->h_max_results_, d_max_results_, sizeof(float) * WAVE_NUM * NFFT,
                            cudaMemcpyDeviceToHost,
                            stream_));

    // 速度通道拷贝回内存
    checkCudaErrors(cudaMemcpyAsync(radar_params_->h_speed_channels_, d_speed_channels_, sizeof(int) * WAVE_NUM * NFFT,
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

    checkCudaErrors(cudaMemsetAsync(d_data_, 0, wave_num_ * pulse_num_ * range_num_ * sizeof(cufftComplex), stream_));
    unpackKernel3D<<<gridDim1, CUDA_BLOCK_SIZE, 0, stream_>>>(
        d_unpack_data_, d_data_, d_headPositions_, PULSE_NUM, RANGE_NUM);

}

void WaveGroupProcessor::streamSynchronize() {
    cudaStreamSynchronize(stream_);
}

void WaveGroupProcessor::fullPipelineProcess()
{

    for (cur_wave_ = start_wave; cur_wave_ < end_wave; cur_wave_++)
    {

        this->processPulseCompression();

        if (MTI_enable)
        {
            this->processMTI();
        }
        this->processCoherentIntegration(radar_params_->scale);
        // this->clutterNoiseClassify();
        this->processFFTshift();
        if (clutter_map_enable)
        {
            this->processClutterMap();
        }
        this->processCFAR();
        this->processMaxSelection();
    }
}

void WaveGroupProcessor::processPulseCompression() {
    int size = pulse_num_ * range_num_;
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (size + blockSize - 1) / blockSize;

    size_t offset = cur_wave_ * pulse_num_ * range_num_;
    auto* data = d_data_ + offset;

    // fft
    checkCufftErrors(cufftExecC2C(row_plan_, data, data, CUFFT_FORWARD));
    // .*
    rowWiseMulKernel<<<gridSize, blockSize, 0, stream_>>>(data, d_pc_coeffs_, pulse_num_, range_num_);
    // ifft
    checkCufftErrors(cufftExecC2C(row_plan_, data, data, CUFFT_INVERSE));
    //
    // this->streamSynchronize();
    // writeArrayToFile(data, pulse_num_, range_num_, "unpackdata.dat");
}

void WaveGroupProcessor::processMTI()
{
    dim3 blockDim_(CUDA_BLOCK_SIZE);
    dim3 gridDim_((range_num_ + blockDim_.x - 1) / blockDim_.x);

    auto* waveDataPtr = d_data_ + cur_wave_ * pulse_num_ * range_num_;
    if (MTI_pulse_num == 2)
    {
        MTIkernel2<<<gridDim_, blockDim_, 0, stream_>>>(waveDataPtr, pulse_num_, range_num_);
    }
    else
    {
        MTIkernel3<<<gridDim_, blockDim_, 0, stream_>>>(waveDataPtr, pulse_num_, range_num_);
    }
}

void WaveGroupProcessor::processCoherentIntegration(float scale) {

    cufftComplex* wavePtr = d_data_ + cur_wave_ * pulse_num_ * range_num_;
    checkCufftErrors(cufftExecC2C(col_plan_, wavePtr, wavePtr, CUFFT_FORWARD));

    thrust_data_ = thrust::device_ptr<cufftComplex>(wavePtr);
    // 抵消脉压增益，同时除以range_num_是ifft之后必须除以ifft才能和matlab结果一样
    int size = pulse_num_ * range_num_;
    thrust::transform(exec_policy_, thrust_data_, thrust_data_ + size, thrust_data_, ScaleFunctor(scale / range_num_ / normFactor));
}

void WaveGroupProcessor::clutterNoiseClassify()
{
    static int count = 0;
    count++;
    // std::string filename1 = "WaveGroupProcessor_" + std::to_string(count) + ".txt";
    // std::string filename2 = "bool" + std::to_string(count) + ".txt";
    this->streamSynchronize();
    // writeComplexToFile(d_data_ + 16*pulse_num_*range_num_, pulse_num_, range_num_, filename1);
    gpu_manager.update_queues(d_data_, cur_wave_);
    // 获取杂噪分类的结果到d_is_masked_
    gpu_manager.get_clutter_copy(d_is_masked_, wave_num_ * range_num_);
    // this->streamSynchronize();
    // writeBoolToFile(d_is_masked_ + 16*range_num_, 1, range_num_, filename2);
}

void WaveGroupProcessor::processFFTshift()
{
    dim3 blockDim_(CUDA_BLOCK_SIZE);
    dim3 gridDim_((range_num_ + blockDim_.x - 1) / blockDim_.x);

    cufftComplex* wavePtr = d_data_ + cur_wave_ * pulse_num_ * range_num_;
    fftshift_columns_inplace_kernel<<<gridDim_, blockDim_, 0, stream_>>>(wavePtr, pulse_num_, range_num_);
}

void WaveGroupProcessor::processClutterMap()
{
    gpu_manager.processClutterMap(d_data_, d_clutterMap_masked_, cur_wave_, clutterMap_range_num_);
}


void WaveGroupProcessor::processCFAR() {
    size_t offset = cur_wave_ * pulse_num_ * range_num_;
    auto* wavePtr = d_data_ + offset;
    // .^2
    int size = pulse_num_ * range_num_;

    thrust_data_ = thrust::device_ptr<cufftComplex>(wavePtr);


    thrust::transform(exec_policy_, thrust_data_, thrust_data_ + size, thrust_data_, SquareFunctor());

    // fft
    checkCufftErrors(cufftExecC2C(row_plan_, wavePtr, d_cfar_res_, CUFFT_FORWARD));
    // .*
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = (size + blockSize - 1) / blockSize;
    rowWiseMulKernel<<<gridSize, blockSize, 0, stream_>>>(d_cfar_res_, d_cfar_coeffs_, pulse_num_, range_num_);

    // ifft
    checkCufftErrors(cufftExecC2C(row_plan_, d_cfar_res_, d_cfar_res_, CUFFT_INVERSE));

    thrust::transform(exec_policy_, thrust_cfar_, thrust_cfar_ + size, thrust_cfar_, ScaleFunctor(1.0 / range_num_ ));

    int cfarKernelSize = 2 * numGuardCells + 2 * numRefCells + 1;

    // 用于抵消频域卷积的偏移量
    int shift_offset = floor((cfarKernelSize - 1) / 2);

    // 根据alpha计算噪底
    double alpha = numRefCells * 2 * (pow(Pfa_cfar, -1.0 / (numRefCells * 2)) - 1);
    thrust::transform(exec_policy_, thrust_cfar_, thrust_cfar_ + size, thrust_cfar_, ScaleFunctor(alpha/2.0/numRefCells));

    // 对比噪底选结果，(结果开根号)
    cmpKernel<<<gridSize, blockSize, 0, stream_>>>(
        wavePtr,                        // 原始数据
        d_cfar_res_,                    // cfar门限
        d_clutterMap_masked_ + offset,  // 杂波图结果
        pulse_num_,                     // 需要计算的脉冲数
        range_num_,                     // 需要计算的距离单元数
        shift_offset,                   // 平移，抵消频域滤波偏移量
        cfar_enable,                    // cfar控制参数，0代表不做cfar
        cfar_db_offset
    );
}

void WaveGroupProcessor::cfar(int numSamples)  {
    double alpha = (numRefCells * 2 * (pow(Pfa_cfar, -1.0 / (numRefCells * 2)) - 1));
    size_t offset = cur_wave_ * pulse_num_ * range_num_;
    // .^2
    int size = pulse_num_ * range_num_;

    auto* wavePtr = d_data_ + offset;
    thrust_data_ = thrust::device_ptr<cufftComplex>(wavePtr);
    thrust::transform(exec_policy_, thrust_data_, thrust_data_ + size, thrust_data_, SquareFunctor());

    // Configure the CUDA kernel launch parameters
    int colsPerThread = CFAR_LENGTH; // 每个线程处理的列数
    int threadsPerBlock = range_num_ / colsPerThread; // 每个线程块中的线程数
    int blocksPerRow = (range_num_ + colsPerThread - 1) / colsPerThread / threadsPerBlock; // 每行的线程块数
    dim3 blockDim(threadsPerBlock, 1); // 线程块大小：1 行 x 32 列
    int nrows = pulse_num_;
    dim3 gridDim(blocksPerRow, nrows); // 网格大小：每行 block 数 x 总行数

    cfarKernel<<<gridDim, blockDim, 0, stream_>>>(d_data_+offset, d_cfar_res_, nrows, range_num_, alpha, numGuardCells,
                                                  numRefCells, numSamples-1, numSamples+RANGE_NUM-range_correct);
}

void WaveGroupProcessor::cfar_by_col()
{
    double alpha = (numRefCells * 2 * (pow(Pfa_cfar, -1.0 / (numRefCells * 2)) - 1));
    dim3 blockDim_(CUDA_BLOCK_SIZE);
    dim3 gridDim_((range_num_ + blockDim_.x - 1) / blockDim_.x);

    int size = wave_num_ * pulse_num_ * range_num_;
    thrust::transform(exec_policy_, thrust_data_, thrust_data_ + size, thrust_data_, SquareFunctor());

    for (int w = start_wave; w < end_wave; ++w)
    {
        auto* waveDataPtr = d_data_ + w * pulse_num_ * range_num_;
        auto* cfarPtr = d_cfar_res_ + w * pulse_num_ * range_num_;
        cfar_col_kernel<<<gridDim_, blockDim_, 0, stream_>>>(waveDataPtr, cfarPtr, pulse_num_, range_num_, alpha,
                                                             numGuardCells, numRefCells);
    }
}


void WaveGroupProcessor::processMaxSelection() {
    size_t offset = cur_wave_ * pulse_num_ * range_num_;
    size_t offset2 = cur_wave_ * range_num_;
    // 使用2D block和grid
    dim3 blockDim_(16, 16);  // 可以根据需要调整block大小
    dim3 gridDim_(
        (range_num_ + blockDim_.x - 1) / blockDim_.x,
        1
    );

    maxKernel2D<<<gridDim_, blockDim_, 0, stream_>>>(
        d_data_ + offset,           // 输入数据
        d_max_results_ + offset2,    // 最大值输出
        d_speed_channels_ + offset2, // 通道索引输出
        d_chnSpeeds,
        d_detect_rows_,    // 通道范围（row 索引数组）
        detect_rows_num_,  // 通道数量
        pulse_num_,        // 总行数
        range_num_,        // 总列数
        1                  // 总波束数
    );
}

void WaveGroupProcessor::getRadarParams() {
    static double delta_range = 0.0;

    checkCudaErrors(cudaMemcpyAsync(radar_params_->rawMessage, d_unpack_data_, DATA_OFFSET, cudaMemcpyDeviceToHost, stream_));

    if (!radar_params_->isInit) {
        radar_params_->isInit = true;
        auto* packageArr = (uint32_t *)(radar_params_->rawMessage);

        auto freqPoint = packageArr[11] & 0x000000ff;
        radar_params_->lambda = c_speed / ((freqPoint * 10 + initCarryFreq) * 1e6);
        radar_params_->pulseWidth = (packageArr[13] & 0xfffff) / Fs_system; //5e-6
        radar_params_->PRT = packageArr[14] / Fs_system;  //120e-6
        auto fLFMStartWord = packageArr[16];
        radar_params_->bandWidth = (Fs_system - fLFMStartWord / pow(2.0f, 32) * Fs_system) * 2.0;

        double fs = 1.0 / radar_params_->PRT;
        double f_step = fs / PULSE_NUM;
        radar_params_->chnSpeeds.clear();

        for(int i = 0; i < PULSE_NUM; ++i) {
            double f = -fs/2.0 + (f_step * i);
            double v = f * radar_params_->lambda / 2.0;
            int v_int = static_cast<int>(std::round(v * 100));
            radar_params_->chnSpeeds.push_back(v_int);
        }

        radar_params_->detect_rows.clear();
        radar_params_->numSamples = round(radar_params_->pulseWidth * Fs);
        for (int row = 0; row < PULSE_NUM; ++row) {
            int speed = std::abs(radar_params_->chnSpeeds[row]);
            if (speed >= v1 && speed <= v2) {
                radar_params_->detect_rows.push_back(row);
            }
        }
        if (delta_range == 0)
        {
            delta_range = c_speed / Fs / 2.0;
        }
        radar_params_->scale = 1.0f / sqrt(radar_params_->bandWidth * radar_params_->pulseWidth) / PULSE_NUM;
        radar_params_->getCoef();
        this->getCoef();
    }
}



void WaveGroupProcessor::saveToDebugFile(int frame, ofstream& debugFile)
{
    if (!debug_mode || frame < start_frame || frame >= end_frame)
    {
        return;
    }

    static bool firstCall = true;  // 静态变量，标记是否为第一次调用
    // 静态成员，用于排序控制
    static std::mutex saveMutex;        // 用于保护 save_cv
    static std::mutex fileMutex;        // 全局互斥锁，用于保护文件写入
    static std::condition_variable save_cv;
    static std::set<int> readyFrames;   // 已准备好保存的 frame
    static int nextToSave = start_frame;         // 下一个待保存的 frame

    // 排序控制逻辑
    {
        std::unique_lock<std::mutex> lock(saveMutex);
        readyFrames.insert(frame);  // 标记当前 frame 已准备好
        // 等待直到当前 frame 是下一个要保存的
        save_cv.wait(lock, [&]{ return frame == nextToSave; });

        // 当前 frame 是 nextToSave，移除并更新 nextToSave
        readyFrames.erase(frame);
        lock.unlock();  // 在文件操作前释放锁
    }

    std::lock_guard<std::mutex> lock(fileMutex);  // 加锁，确保线程安全

    if (firstCall)
    {
        // 定义需要保存的参数
        int pulseNum = PULSE_NUM;  // 假设PULSE_NUM已定义
        int nfft = NFFT;           // 假设NFFT已定义

        // 写入double类型的参数
        debugFile.write(reinterpret_cast<char*>(&radar_params_->bandWidth), sizeof(double));
        debugFile.write(reinterpret_cast<char*>(&radar_params_->pulseWidth), sizeof(double));
        debugFile.write(reinterpret_cast<char*>(&Fs), sizeof(double));
        debugFile.write(reinterpret_cast<char*>(&radar_params_->lambda), sizeof(double));
        debugFile.write(reinterpret_cast<char*>(&radar_params_->PRT), sizeof(double));

        // 写入start_wave和end_wave
        debugFile.write(reinterpret_cast<char*>(&start_wave), sizeof(int));
        debugFile.write(reinterpret_cast<char*>(&end_wave), sizeof(int));

        // 写入矩阵大小
        debugFile.write(reinterpret_cast<char*>(&pulseNum), sizeof(int));
        debugFile.write(reinterpret_cast<char*>(&nfft), sizeof(int));

        // 计算并保存32个波束的方位
        std::vector<double> azi(32);
        for (int ii = 0; ii < 32; ++ii)
        {
            int nAzmCode = (azi_table[ii] & 0xffff);
            if (nAzmCode > 32768)
                nAzmCode -= 65536;
            double rAzm = 249.0633 + asin((nAzmCode * radar_params_->lambda) / (65536 * d)) / 3.1415926 * 180.0;
            if (rAzm < 0)
                rAzm += 360.0;
            azi[ii] = rAzm;
        }
        debugFile.write(reinterpret_cast<char*>(azi.data()), 32 * sizeof(double));

        firstCall = false;  // 标记首次写入已完成
    }

    // 以下是原有逻辑，保存当前帧的时间和数据
    int oneWaveSize = PULSE_NUM * NFFT;
    int waveNum = end_wave - start_wave;

    auto* startAddr = d_data_ + start_wave * oneWaveSize;
    size_t size = waveNum * oneWaveSize;

    auto* h_data = new cufftComplex[size];  // 在主机上分配内存

    // 从显存复制数据到主机内存
    cudaMemcpy(h_data, startAddr, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // 写入时间和数据
    auto rawMsg = reinterpret_cast<uint32_t*>(radar_params_->rawMessage);
    auto time = rawMsg[6] / 10 + 8 * 60 * 60 * 1000;  // FPGA时间，0.1ms转为1ms并加8小时

    debugFile.write(reinterpret_cast<char*>(&time), 4);  // 写入时间（4字节）
    debugFile.write(reinterpret_cast<char*>(h_data), size * sizeof(cufftComplex));  // 写入数据

    delete[] h_data;  // 释放主机内存

    // 更新 nextToSave 并通知其他线程
    {
        std::lock_guard<std::mutex> lock(saveMutex);
        nextToSave++;  // 移到下一个 frame
        save_cv.notify_all();  // 通知等待的线程
    }

}
