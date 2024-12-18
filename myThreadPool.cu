#include "myThreadPool.h"
#include "Config.h"
#include <cuda_runtime.h>
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads, SharedQueue *sharedQueue)
        : stop(false), sharedQueue(sharedQueue), processingFlags(numThreads, false),
          conditionVariables(numThreads), mutexes(numThreads),
          headPositions(numThreads, std::vector<int>()), currentPos(numThreads, 0),
          currentAddrOffset(0), numThreads(numThreads), inPacket(false),
          cur_thread_id(0), prevSeqNum(0), sender() { // 初始化 conditionVariables 和 mutexes
    // 创建并初始化线程

    logFile = ofstream("error_log.txt", ios_base::app);

    allocateThreadMemory();

    uint64Pattern = *(uint64_t *) pattern;

    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
    }


//    cout << 1 << endl;


    circleStartTime = high_resolution_clock::now();

    cout << "Initial Finished" << endl;
}

ThreadPool::~ThreadPool() {

    stop = true;
    for (auto &cv: conditionVariables) cv.notify_all(); // 通知所有线程退出
    for (std::thread &thread: threads) {
        if (thread.joinable()) thread.join();
    }
    freeThreadMemory();
    logFile.close();
}

void ThreadPool::allocateThreadMemory() {
    cout << "cudaMalloc " << THREADS_MEM_SIZE / 1024 / 1024 << " MB for each thread" << endl;
    for (size_t i = 0; i < numThreads; ++i) {
        char *d_memory = nullptr;
        cudaError_t err = cudaMalloc((void **) &d_memory, THREADS_MEM_SIZE);  // 在显存中分配内存
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed for thread " << i
                      << ": " << cudaGetErrorString(err) << std::endl;
            return;
        }
        threadsMemory.emplace_back(d_memory);

        cudaStream_t stream;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            cerr << "Failed to create stream" << endl;
        }
        streams.push_back(stream);
    }
}

void ThreadPool::freeThreadMemory() {
    for (size_t i = 0; i < numThreads; ++i) {
        checkCudaErrors(cudaFree(threadsMemory[i]));
        cudaStreamSynchronize(streams[i]); // 等待流中的所有操作完成
        cudaStreamDestroy(streams[i]);
    }
    threadsMemory.clear();
}

//// TODO: 这里要做可更改系数的
//void ThreadPool::initPCcoefMatrix() {
////    double C = 3e8;
//    double BandWidth = 6e6;
//    double PulseWidth = 5e-6;
//
////    double PRT = 100e-6;
//    numSamples = round(PulseWidth * Fs);
//    NFFT = RANGE_NUM;
//
//    auto LFM = generateLFM(BandWidth, PulseWidth, Fs);
//    auto PCcoef = generatePCcoef(LFM);
//    PCcoef = repmat(PCcoef, NUM_PULSE, 1);
//
//    PCcoefMatrix = CudaMatrix(NUM_PULSE, numSamples, PCcoef);
//    PCcoefMatrix.fft_N(NFFT);   // 提前做fft
//}

void ThreadPool::run() {
    while (!stop) {
        sem_wait(&sharedQueue->items_available); // 等待可用数据
        sem_wait(&sharedQueue->mutex); // 锁住共享资源

        copyToThreadMemory();

        sem_post(&sharedQueue->mutex); // 解锁
        sem_post(&sharedQueue->slots_available); // 增加空槽位
    }
}

void ThreadPool::threadLoop(int threadID) {
    // 创建 CudaMatrix 对象和 pComplex 指针， 存放解包完后的脉组数据
//    cufftComplex* pComplex = new cufftComplex[WAVE_NUM * NUM_PULSE * RANGE_NUM];
    cufftComplex *pComplex;
    checkCudaErrors(cudaMalloc(&pComplex, sizeof(cufftComplex) * WAVE_NUM * NUM_PULSE * RANGE_NUM));

    int *d_headPositions;
    checkCudaErrors(cudaMalloc(&d_headPositions, NUM_PULSE * 1.1 * sizeof(size_t)));

    CudaMatrix PcCoefMatrix(1, RANGE_NUM); // 脉压系数

    // 原始IQ
    vector<CudaMatrix> matrices;
    for (int i = 0; i < WAVE_NUM; i++) {
        matrices.emplace_back(NUM_PULSE, RANGE_NUM, pComplex + i * NUM_PULSE * RANGE_NUM, true);
    }
    // CFAR结果
    vector<CudaMatrix> CFAR_res(WAVE_NUM, CudaMatrix(NUM_PULSE, RANGE_NUM));

    // 选大结果 device
    cufftComplex *pMaxRes_d;
    checkCudaErrors(cudaMalloc(&pMaxRes_d, sizeof(cufftComplex) * WAVE_NUM * RANGE_NUM));

    vector<CudaMatrix> Max_res;
    for (int i = 0; i < WAVE_NUM; i++) {
        Max_res.emplace_back(1, RANGE_NUM, pMaxRes_d + i * RANGE_NUM, true);
    }

    // 选大结果，host
    cufftComplex* pMaxRes_h = new cufftComplex[WAVE_NUM * RANGE_NUM];

    cufftHandle pcPlan;                             // 脉压fft的plan
    cufftHandle rowPlan;                            // 按行做fft的plan
    cufftHandle colPlan;                            // 按列做fft的plan

    int nrows = NUM_PULSE;
    int ncols = RANGE_NUM;
    checkCufftErrors(cufftPlan1d(&pcPlan, ncols, CUFFT_C2C, 1));
    checkCufftErrors(cufftSetStream(pcPlan, streams[threadID]));

    checkCufftErrors(cufftPlan1d(&rowPlan, ncols, CUFFT_C2C, nrows));
    checkCufftErrors(cufftSetStream(rowPlan, streams[threadID]));

    checkCufftErrors(cufftPlanMany(&colPlan, 1, &nrows,     // Rank and size of the FFT
                                   &ncols, ncols, 1,              // Input data layout
                                   &ncols, ncols, 1,            // Output data layout
                                   CUFFT_C2C, RANGE_NUM)      // FFT type and number of FFTs
    );
    checkCufftErrors(cufftSetStream(colPlan, streams[threadID]));

    while (!stop) {
        waitForProcessingSignal(threadID);

        if (stop) break; // 退出循环

        processData(threadID, pComplex, matrices, PcCoefMatrix, d_headPositions, CFAR_res, Max_res, pMaxRes_d,
                    pMaxRes_h,pcPlan, rowPlan, colPlan); // 处理 CUDA 内存中的数据

        // 处理完毕后重置标志
        {
            std::lock_guard<std::mutex> lock(mutexes[threadID]);
            processingFlags[threadID] = false;
        }
    }

    // 释放内存
    checkCudaErrors(cudaFree(d_headPositions));
//    delete[] pComplex;
    checkCudaErrors(cudaFree(pComplex));
    checkCudaErrors(cudaFree(pMaxRes_d));
    delete[] pMaxRes_h;

    checkCufftErrors(cufftDestroy(pcPlan));
    checkCufftErrors(cufftDestroy(rowPlan));
    checkCufftErrors(cufftDestroy(colPlan));
}

__global__ void processKernel(char *threadsMemory, cufftComplex *pComplex,
                              const int *headPositions, int numHeads, int rangeNum) {
    // 获取线程和网格索引
    int headIdx = blockIdx.z;  // 每个block.z处理一个头位置
    int rangeIdx = blockIdx.y * blockDim.x + threadIdx.x; // 每个线程处理一个距离单元
    int beamIdx = blockIdx.x;  // 每个block.x处理一个波束

    // 检查索引是否越界
    if (headIdx < numHeads && rangeIdx < rangeNum && beamIdx < WAVE_NUM) {
        // 计算头位置的起始地址
        int headOffset = headPositions[headIdx];
        char *blockIQstartAddr = threadsMemory + headOffset + DATA_OFFSET;

        // 计算当前数据块的偏移和新索引
        int blockOffset = rangeIdx * WAVE_NUM * 4 + beamIdx * 4;
        int newIndex = beamIdx * NUM_PULSE * RANGE_NUM + headIdx * RANGE_NUM + rangeIdx;

        // 提取IQ数据并存储到结果数组
        pComplex[newIndex].x = TwoChars2float(blockIQstartAddr + blockOffset);
        pComplex[newIndex].y = TwoChars2float(blockIQstartAddr + blockOffset + 2);
    }
}

void ThreadPool::generatePCcoefMatrix(CudaMatrix &PcCoefMatrix, char *rawMessage, cudaStream_t _stream) {
    if(*(uint64_t *) (rawMessage) != uint64Pattern) {
        cerr << "data error" << endl;
        return;
    }
//    double BandWidth = 6e6;
//    double PulseWidth = 5e-6;

    auto PRT = FourChars2Uint(rawMessage + 14 * 4) / Fs_system;

    auto pulseWidth = (FourChars2Uint(rawMessage + 13 * 4)  & 0xfffff) / Fs_system;

    numSamples = round(pulseWidth * Fs);

    auto fLFMStartWord = FourChars2Uint(rawMessage + 16 * 4);
    double bandWidth = (Fs_system - fLFMStartWord / pow(2.0f,32) * Fs_system) * 2.0;

    vector<cufftComplex> PcCoef = PCcoef(bandWidth, pulseWidth, Fs,  RANGE_NUM);

    PcCoefMatrix.copyFromHost(_stream, 1, RANGE_NUM, PcCoef.data());
}

// 线程池中的数据处理函数
void
ThreadPool::processData(int threadID, cufftComplex *pComplex, vector<CudaMatrix> &matrices, CudaMatrix &PcCoefMatrix,
                        int *d_headPositions, vector<CudaMatrix> &CFAR_res, vector<CudaMatrix> &Max_res,
                        cufftComplex *pMaxRes_d, cufftComplex *pMaxRes_h, cufftHandle &pcPlan, cufftHandle &rowPlan,
                        cufftHandle &colPlan) {
//    if (threadID) return;

    cout << "thread " << threadID << " start" << endl;
//    auto start = high_resolution_clock::now();

    int numHeads = headPositions[threadID].size();       // 256
    int headLength = headPositions[threadID][1] - headPositions[threadID][0];
    int rangeNum = floor((headLength - DATA_OFFSET) / WAVE_NUM / 4.0);

    // 头的位置拷贝到显存
    cudaStreamSynchronize(streams[threadID]); // 等待流中的拷贝操作完成
    checkCudaErrors(cudaMemcpyAsync(d_headPositions, headPositions[threadID].data(), numHeads * sizeof(int), cudaMemcpyHostToDevice,
                    streams[threadID]));

    cudaStreamSynchronize(streams[threadID]); // 等待流中的拷贝操作完成

    char rawMessage[DATA_OFFSET];

    // 包头信息拷贝回内存(仅第一个包头)
    checkCudaErrors(cudaMemcpyAsync(rawMessage, threadsMemory[threadID], sizeof(rawMessage), cudaMemcpyDeviceToHost,
                    streams[threadID]));

    generatePCcoefMatrix(PcCoefMatrix, rawMessage, streams[threadID]);

    // 定义 blockDim 的大小
    const int threadsPerBlock = 256; // 每个block的线程数，根据硬件性能选择

    // 计算 gridDim 的大小
    dim3 gridDim1(WAVE_NUM, (rangeNum + threadsPerBlock - 1) / threadsPerBlock, numHeads);

    // 在gpu解包
    processKernel<<<gridDim1, threadsPerBlock>>>(threadsMemory[threadID], pComplex,
                                                                       d_headPositions, numHeads, rangeNum);
//    if (!threadID) {
//        for (int i = 0; i < WAVE_NUM; i++) {
//            matrices[i].print(i, i+1);
//        }
//    }
    processPulseGroupData(threadID, matrices, PcCoefMatrix, CFAR_res, Max_res, rangeNum,
                          pcPlan, rowPlan, colPlan);

    // 选大结果拷贝回内存
    cudaStreamSynchronize(streams[threadID]); // 等待流中的拷贝操作完成
    checkCudaErrors(cudaMemcpyAsync(pMaxRes_h, pMaxRes_d, sizeof(cufftComplex) * CAL_WAVE_NUM * RANGE_NUM, cudaMemcpyDeviceToHost,
                    streams[threadID]));


    cudaStreamSynchronize(streams[threadID]); // 等待流中的拷贝操作完成
    if (!threadID)
        sender.send(rawMessage, pMaxRes_h, numSamples, rangeNum);

    // 记录结束时间
//    auto endTime = high_resolution_clock::now();

    // 输出时间统计
//    cout << "thread " << threadID << " process finished after " << duration_cast<milliseconds>(endTime - start).count() << " ms" << endl;
    cout << "thread " << threadID << " process finished"  << endl;
    cudaStreamSynchronize(streams[threadID]); // 等待流中的拷贝操作完成
}



// 在GPU处理一个脉组的所有波束的数据，全流程处理，包括脉压、积累、CFAR、选大。
void ThreadPool::processPulseGroupData(int threadID, vector<CudaMatrix> &matrices, CudaMatrix &PcCoefMatrix,
                                       vector<CudaMatrix> &CFAR_res, vector<CudaMatrix> &Max_res, int rangeNum,
                                       cufftHandle &pcPlan, cufftHandle &rowPlan, cufftHandle &colPlan) {

    PcCoefMatrix.fft(pcPlan);
//    matrices[0].printShape();
    for (int i = 0; i < CAL_WAVE_NUM; i++) {
        /*Pulse Compression*/
        matrices[i].fft(rowPlan);
        matrices[i].rowWiseMul(PcCoefMatrix, streams[threadID]);
        matrices[i].ifft(streams[threadID], rowPlan);

        /*coherent integration*/
        for (int j = 0; j < INTEGRATION_TIMES; j++) {
            matrices[i].fft_by_col(colPlan);
        }

        /*cfar*/
        matrices[i].cfar(CFAR_res[i], streams[threadID], Pfa, numGuardCells, numRefCells, numSamples - 1,
                           numSamples - 1 + rangeNum);
        CFAR_res[i].max(Max_res[i], streams[threadID], 1);
    }
}

// 通知线程池里的线程开始干活
void ThreadPool::notifyThread(int threadID) {
    bool currenState = processingFlags[threadID];
    if (currenState) {
        cerr << "thread " << threadID << " is busy now" << endl;
    }

    // 设置处理标志并通知线程
    {
        std::lock_guard<std::mutex> lock(mutexes[threadID]);
        processingFlags[threadID] = true;
    }
    conditionVariables[threadID].notify_one();
}

void ThreadPool::waitForProcessingSignal(int threadID) {
    std::unique_lock<std::mutex> lock(mutexes[threadID]);
    conditionVariables[threadID].wait(lock, [this, threadID]() {
        return processingFlags[threadID] || stop;
    });
}

// 循环将内存的数据拷贝到显存(未解包)，每个线程对应一个脉组的数据
void ThreadPool::copyToThreadMemory() {
    int block_index = sharedQueue->read_index;
//    std::cout << "Block index: " << block_index << std::endl << std::endl;
//    cout << "block_index： " << block_index << endl;
    unsigned int seqNum;

    unsigned int indexValue; // 当前packet相对于1GB的起始地址
    unsigned long copyStartAddr = block_index * BLOCK_SIZE; // 当前Block相对于1GB的复制起始地址
    bool startFlag;
    for (int i = 0; i < 2048; i++) {
        size_t indexOffset = block_index * INDEX_SIZE + i * 4;
        indexValue = FourChars2Uint(sharedQueue->index_buffer + indexOffset);
        // Check pattern match
        if (*(uint64_t *) (sharedQueue->buffer + indexValue) == uint64Pattern) {
            seqNum = FourChars2Uint(sharedQueue->buffer + indexValue + SEQ_OFFSET);
            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0) {
                inPacket = false;
                currentAddrOffset = 0;
                std::cerr << "Error! Sequence number not continuous!" << std::endl;
                std::time_t now = std::time(nullptr);
                std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                logFile << "[" << timebuf << "] " << "Error! Sequence number not continuous!" << std::endl;
            }
            prevSeqNum = seqNum;

            startFlag = static_cast<uint8_t>(sharedQueue->buffer[indexValue + 23]) & 0x02;

            if (startFlag) {
//                cout << "Packege start. seqNum:" << seqNum << endl;
//                auto startTime = high_resolution_clock::now();
//                cout << "Total processing time for pulse group data: "
//                     << duration_cast<microseconds>(startTime - circleStartTime).count() << " us" << endl;
//                circleStartTime = startTime;

                // 发送上一个脉组的数据
                if (inPacket) {
                    memcpyDataToThread(copyStartAddr, indexValue);
                    notifyThread(cur_thread_id);
                    cur_thread_id = (cur_thread_id + 1) % numThreads;
                }

                // 初始化当前脉冲的参数
                inPacket = true;
                currentPos[cur_thread_id] = 0;
                prevIndexValue = indexValue;
                currentAddrOffset = 0;
                headPositions[cur_thread_id].clear();
                copyStartAddr = indexValue;
//                std::cout << "Start flag detected, starting at address: " << copyStartAddr << std::endl;
            }

            if (inPacket) {
                if (block_index == 0 && i == 0) {
                    currentPos[cur_thread_id] +=
                            (indexValue + QUEUE_SIZE * BLOCK_SIZE - prevIndexValue) % (QUEUE_SIZE * BLOCK_SIZE);
                } else {
                    currentPos[cur_thread_id] += (indexValue - prevIndexValue);
                }
                headPositions[cur_thread_id].push_back(currentPos[cur_thread_id]);
            }
            prevIndexValue = indexValue;
        } else {
            if (inPacket) {
                unsigned int copyEndAddr = (block_index + 1) * BLOCK_SIZE;
                memcpyDataToThread(copyStartAddr, copyEndAddr);
            }
            break;
        }
    }
    sharedQueue->read_index = (sharedQueue->read_index + 1) % QUEUE_SIZE;
}

// 从startAddr到endAddr的数据拷贝给线程的独立空间，Addr是相对于共享内存的起始地址
void ThreadPool::memcpyDataToThread(unsigned int startAddr, unsigned int endAddr) {
    size_t copyLength = endAddr - startAddr;
//    std::cout << "Copying " << copyLength / 1024 / 1024 << " MB to thread " << cur_thread_id << std::endl;
//    cout << cur_thread_id << ": " << (currentAddrOffset + copyLength) / 1024 / 1024 << " MB" << endl;

    if ((currentAddrOffset + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds

//        // 内存拷贝到显存
        checkCudaErrors(cudaMemcpyAsync(threadsMemory[cur_thread_id] + currentAddrOffset,
                        sharedQueue->buffer + startAddr,
                        copyLength,
                        cudaMemcpyHostToDevice,
                        streams[cur_thread_id]
        ));
//        cudaMemcpy(threadsMemory[cur_thread_id] + currentAddrOffset,
//                        sharedQueue->buffer + startAddr,
//                        copyLength,
//                        cudaMemcpyHostToDevice
//        );

        currentAddrOffset += copyLength;
    } else {
//        cout << (currentAddrOffset + copyLength) / 1024 / 1024 << " MB !!!!" << endl;
        inPacket = false;
        currentAddrOffset = 0;
        std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
    }
}

void checkCufftErrors(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: ";
        switch (result) {
            case CUFFT_INVALID_PLAN:
                std::cerr << "CUFFT_INVALID_PLAN";
                break;
            case CUFFT_ALLOC_FAILED:
                std::cerr << "CUFFT_ALLOC_FAILED";
                break;
            case CUFFT_INVALID_TYPE:
                std::cerr << "CUFFT_INVALID_TYPE";
                break;
            case CUFFT_INVALID_VALUE:
                std::cerr << "CUFFT_INVALID_VALUE";
                break;
            case CUFFT_INTERNAL_ERROR:
                std::cerr << "CUFFT_INTERNAL_ERROR";
                break;
            case CUFFT_EXEC_FAILED:
                std::cerr << "CUFFT_EXEC_FAILED";
                break;
            case CUFFT_SETUP_FAILED:
                std::cerr << "CUFFT_SETUP_FAILED";
                break;
            case CUFFT_INVALID_SIZE:
                std::cerr << "CUFFT_INVALID_SIZE";
                break;
            default:
                std::cerr << "Unknown error";
        }
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cudaError_t code: " << result << endl;
        throw runtime_error(cudaGetErrorString(result));
    }
}


unsigned int ThreadPool::FourChars2Uint(const char *startAddr) {
    return static_cast<uint8_t>(startAddr[0]) << 24
           | static_cast<uint8_t>(startAddr[1]) << 16
           | static_cast<uint8_t>(startAddr[2]) << 8
           | static_cast<uint8_t>(startAddr[3]);
}


// TwoChars2float 函数，将两个字符转换为 float 类型
__device__ float TwoChars2float(const char *startAddr) {
    return static_cast<float>(  static_cast<uint8_t>(startAddr[0]) << 8
                                | static_cast<uint8_t>(startAddr[1]));
}