#include "myThreadPool.h"
#include <cuda_runtime.h>
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads, SharedQueue *sharedQueue)
        : stop(false), sharedQueue(sharedQueue), processingFlags(numThreads, false),
          conditionVariables(numThreads), mutexes(numThreads),
          headPositions(numThreads, std::vector<int>()), currentPos(numThreads, 0),
          currentAddrOffset(0), numThreads(numThreads), inPacket(false),
          cur_thread_id(0), prevSeqNum(0) { // 初始化 conditionVariables 和 mutexes
    // 创建并初始化线程

    logFile = ofstream("error_log.txt", ios_base::app);
    uint64Pattern = *(uint64_t *) pattern;
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
    }
    allocateThreadMemory();
    initPCcoefMatrix();
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
    for (size_t i = 0; i < numThreads; ++i) {
        char* d_memory = nullptr;
        cudaError_t err = cudaMalloc((void**)&d_memory, THREADS_MEM_SIZE);  // 在显存中分配内存
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
        cudaFree(threadsMemory[i]);
        cudaStreamSynchronize(streams[i]); // 等待流中的所有操作完成
        cudaStreamDestroy(streams[i]);
    }

    threadsMemory.clear();
}

// TODO: 这里要做可更改系数的
void ThreadPool::initPCcoefMatrix() {
//    double C = 3e8;
    double BandWidth = 15e6;
    double PulseWidth = 2e-6;
    double Fs = 31.25e6;
//    double PRT = 100e-6;

    numSamples = round(PulseWidth * Fs);
    NFFT = RANGE_NUM;

    auto LFM = generateLFM(BandWidth, PulseWidth, Fs);
    auto PCcoef = generatePCcoef(LFM);
    PCcoef = repmat(PCcoef, NUM_PULSE, 1);

    PCcoefMatrix = CudaMatrix(NUM_PULSE, numSamples, PCcoef);
    PCcoefMatrix.fft_N(NFFT);   // 提前做fft
}

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
    cufftComplex* pComplex;
    cudaMalloc(&pComplex, sizeof(cufftComplex) * WAVE_NUM * NUM_PULSE * RANGE_NUM);

    int* d_headPositions;
    cudaMalloc(&d_headPositions, 300 * sizeof(size_t));

    // 原始IQ
    vector<CudaMatrix> matrices;
    for (int i = 0; i < WAVE_NUM; i++) {
        matrices.emplace_back(NUM_PULSE, RANGE_NUM, pComplex + i * NUM_PULSE * RANGE_NUM, true);
    }

    // CFAR结果
    vector<CudaMatrix> CFAR_res(WAVE_NUM, CudaMatrix(NUM_PULSE, RANGE_NUM));
    // 选大结果
    vector<CudaMatrix> Max_res(WAVE_NUM, CudaMatrix(1, RANGE_NUM));

    while (!stop) {
        waitForProcessingSignal(threadID);

        if (stop) break; // 退出循环

        processData(threadID, pComplex, matrices, d_headPositions, CFAR_res, Max_res); // 处理 CUDA 内存中的数据

        // 处理完毕后重置标志
        {
            std::lock_guard<std::mutex> lock(mutexes[threadID]);
            processingFlags[threadID] = false;
        }
    }

    // 释放内存
    cudaFree(d_headPositions);
//    delete[] pComplex;
    cudaFree(pComplex);
}


// TwoChars2float 函数，将两个字符转换为 float 类型
__device__ float TwoChars2float(const char* startAddr) {
    return static_cast<float>(  static_cast<uint8_t>(startAddr[0]) << 8
                                | static_cast<uint8_t>(startAddr[1]));
}

// 核函数：unpackDatabuf2CudaMatrices
// 该核函数用于将数据从字符数组转换为 cufftComplex 数组
__global__ void unpackDatabuf2CudaMatrices(const char* data, const int* headPositions,
                                           int numHeads, int rangeNum, cufftComplex* pComplex) {
    int idx = threadIdx.x;

    if (idx < numHeads) {
        auto blockIQstartAddr = data + headPositions[idx] + 33 * 4; // 计算数据起始地址

        // 遍历 rangeNum 和 WAVE_NUM，按块处理数据
        for (int i = 0; i < rangeNum; i++) {
            for (int j = 0; j < WAVE_NUM; j++) {
                int blockOffset = i * WAVE_NUM * 4 + j * 4; // 每个数据块的偏移
                auto newindex = j * NUM_PULSE * RANGE_NUM + idx * RANGE_NUM + i;  // 计算新的索引位置

//                // 检查是否越界访问
//                if (idx < numHeads - 1 && blockOffset + 4 > headPositions[idx + 1] - headPositions[idx] - 19 * 4) {
//                    printf("Out of range access at thread %d, i=%d, j=%d\n", idx, i, j);
//                }
//
//                if (newindex >= numHeads * rangeNum) {
//                    printf("Out of range access at thread %d, i=%d, j=%d\n", idx, i, j);
//                }

                // 将转换后的数据存入 pComplex 数组
                pComplex[newindex].x = TwoChars2float(blockIQstartAddr + blockOffset);
                pComplex[newindex].y = TwoChars2float(blockIQstartAddr + blockOffset + 2);
            }
        }
    }
}


__global__ void processKernel(char* threadsMemory, cufftComplex* pComplex,
                              const int* headPositions, int numHeads, int rangeNum) {
    // 获取线程和网格索引
    int headIdx = blockIdx.z;  // 每个block.z处理一个头位置
    int rangeIdx = blockIdx.y * blockDim.x + threadIdx.x; // 每个线程处理一个距离单元
    int beamIdx = blockIdx.x;  // 每个block.x处理一个波束

    // 检查索引是否越界
    if (headIdx < numHeads && rangeIdx < rangeNum && beamIdx < WAVE_NUM) {
        // 计算头位置的起始地址
        int headOffset = headPositions[headIdx];
        char* blockIQstartAddr = threadsMemory + headOffset + 33 * 4;

        // 计算当前数据块的偏移和新索引
        int blockOffset = rangeIdx * WAVE_NUM * 4 + beamIdx * 4;
        int newIndex = beamIdx * NUM_PULSE * RANGE_NUM + headIdx * RANGE_NUM + rangeIdx;

        // 提取IQ数据并存储到结果数组
        pComplex[newIndex].x = TwoChars2float(blockIQstartAddr + blockOffset);
        pComplex[newIndex].y = TwoChars2float(blockIQstartAddr + blockOffset + 2);
    }
}

// 线程池中的数据处理函数
// #define WAVE_NUM 32    // 波束数
// #define NUM_PULSE 256     // 一个脉组中的脉冲数
// #define RANGE_NUM 8192   // 一个脉冲中的距离单元数
void ThreadPool::processData(int threadID, cufftComplex *pComplex, vector<CudaMatrix> &matrices,
                             int *d_headPositions, vector<CudaMatrix> &CFAR_res, vector<CudaMatrix> &Max_res) {

    cout << "thread " << threadID << " start" << endl;

    int numHeads = headPositions[threadID].size();       // 256
    int headLength = headPositions[threadID][1] - headPositions[threadID][0];
    int rangeNum = floor((headLength - 33 * 4) / WAVE_NUM / 4.0);

    cudaMemcpyAsync(d_headPositions, headPositions[threadID].data(), numHeads * sizeof(int), cudaMemcpyHostToDevice, streams[threadID]);

//    unpackDatabuf2CudaMatrices<<<1, numHeads, 0, streams[threadID]>>>(threadsMemory[threadID], d_headPositions, numHeads, rangeNum, pComplex);

    // 定义 blockDim 的大小
    const int threadsPerBlock = 256; // 每个block的线程数，根据硬件性能选择

    // 计算 gridDim 的大小
    dim3 gridDim1(WAVE_NUM, (rangeNum + threadsPerBlock - 1) / threadsPerBlock, numHeads);

    processKernel<<<gridDim1, threadsPerBlock, 0, streams[threadID]>>>(threadsMemory[threadID], pComplex, d_headPositions, numHeads, rangeNum);

//    cudaStreamSynchronize(streams[threadID]);
//    if (!threadID) {
//        for (int i = 0; i < WAVE_NUM; i++) {
//            matrices[i].print(i, i+1);
//        }
//
//    }

    processPulseGroupData(threadID, matrices, CFAR_res, Max_res, rangeNum);

    cout << "thread " << threadID << " process finished" << endl;
}


void ThreadPool::processPulseGroupData(int threadID, vector<CudaMatrix> &matrices, vector<CudaMatrix> &CFAR_res,
                                       vector<CudaMatrix> &Max_res, int rangeNum) {
    for (int i = 0; i < CAL_WAVE_NUM; i++) {

        /*Pulse Compression*/
        matrices[i].fft(streams[threadID]);
        matrices[i].elementWiseMul(PCcoefMatrix, streams[threadID]);
        matrices[i].ifft(streams[threadID]);
        auto& PCres_Segment = matrices[i];


        /*coherent integration*/
        PCres_Segment.fft_by_col(streams[threadID]);

        /*cfar*/
        double Pfa = 1e-6;
        int numGuardCells = 4;
        int numRefCells = 20;
        PCres_Segment.cfar(CFAR_res[i], streams[threadID], Pfa, numGuardCells, numRefCells, numSamples-1, numSamples-1+rangeNum);
        CFAR_res[i].max(Max_res[i], streams[threadID], 1);
//        res_cfar.printLargerThan0();
    }
}


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

void ThreadPool::copyToThreadMemory() {
    int block_index = sharedQueue->read_index;
//    std::cout << "Block index: " << block_index << std::endl << std::endl;

    unsigned int seqNum;

    unsigned int indexValue; // 当前packet相对于1GB的起始地址
    unsigned long copyStartAddr = block_index * BLOCK_SIZE; // 当前Block相对于1GB的复制起始地址
    bool startFlag;

    for (int i = 0; i < 1024; i++) {
        size_t indexOffset = block_index * INDEX_SIZE + i * 4;
        indexValue = FourChars2Uint(sharedQueue->index_buffer + indexOffset);

        // Check pattern match
        if (*(uint64_t *) (sharedQueue->buffer + indexValue) == uint64Pattern) {
            seqNum = FourChars2Uint(sharedQueue->buffer + indexValue + SEQ_OFFSET);
            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0) {
                inPacket = false;
                std::cerr << "Error! Sequence number not continuous!" << std::endl;
                std::time_t now = std::time(nullptr);
                std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                logFile << "[" << timebuf << "] " << "Error! Sequence number not continuous!" << std::endl;
            }
            prevSeqNum = seqNum;

            startFlag = static_cast<uint8_t>(sharedQueue->buffer[indexValue + 23]) & 0x02;

            if (startFlag) {

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
                if (block_index == 0 && i == 0){
                    currentPos[cur_thread_id] += (indexValue + QUEUE_SIZE * BLOCK_SIZE - prevIndexValue) % (QUEUE_SIZE * BLOCK_SIZE);
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

void ThreadPool::memcpyDataToThread(unsigned int startAddr, unsigned int endAddr){
    size_t copyLength = endAddr - startAddr;
//    std::cout << "Copying " << copyLength / 1024 / 1024 << " MB from address " << startAddr << " to thread memory at " << currentAddrOffset << std::endl;

    if ((currentAddrOffset + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds
//        memcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
//               sharedQueue->buffer + startAddr,
//               copyLength);

//        // 内存拷贝到显存
        cudaMemcpyAsync(threadsMemory[cur_thread_id] + currentAddrOffset,
                   sharedQueue->buffer + startAddr,
                   copyLength,
                   cudaMemcpyHostToDevice,
                   streams[cur_thread_id]
                   );

        currentAddrOffset += copyLength;
    } else {
//        cout << (currentAddrOffset + copyLength) / 1024 / 1024 << " MB !!!!" << endl;
        std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
    }
}


unsigned int ThreadPool::FourChars2Uint(const char* startAddr){
    return   static_cast<uint8_t>(startAddr[0]) << 24
           | static_cast<uint8_t>(startAddr[1]) << 16
           | static_cast<uint8_t>(startAddr[2]) << 8
           | static_cast<uint8_t>(startAddr[3]);
}

float ThreadPool::TwoChars2float(const char* startAddr) {
    return static_cast<float>(static_cast<uint8_t>(startAddr[0]) << 8
                              | static_cast<uint8_t>(startAddr[1]));
}

// 检查 CUDA 错误的函数
void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(result));
    }
}

void printHex(const char *data, size_t dataSize) {
    for (int i = 0; i < dataSize; ++i) {
        if (i%4 == 0)  {
            cout << endl;
            printf("%d: ", i / 4);
        }

        std::cout << std::hex << std::setw(2) << std::setfill('0') << (static_cast<unsigned int>(data[i]) & (0xFF))
                  << " ";
    }
    std::cout << std::dec;
    std::cout << std::endl;
}



// 线程池中的数据处理函数
//void ThreadPool::processData(int threadID, cufftComplex* pComplex, vector<CudaMatrix>& matrices) {
////    if (threadID) return;
//    // 开始计时
//    auto startTime = high_resolution_clock::now();
//    cout << "thread " << threadID << " start" << endl;
//
//    // 计算 rangeNum
//    int numHeads = headPositions[threadID].size();
//    int headLength = headPositions[threadID][1] - headPositions[threadID][0];
//    int rangeNum = floor((headLength - 33 * 4) / WAVE_NUM / 4.0);
//
//    // 数据填充
//
//    auto dataFillStart = high_resolution_clock::now();
//
//    for (int idx = 0; idx < numHeads; ++idx) {
//        auto blockIQstartAddr = threadsMemory[threadID] + headPositions[threadID][idx] + 33 * 4;
//        for (int i = 0; i < rangeNum; i++) {
//            for (int j = 0; j < WAVE_NUM; j++) {
//                int blockOffset = i * WAVE_NUM * 4 + j * 4;
//                auto newindex = j * NUM_PULSE * RANGE_NUM + idx * RANGE_NUM + i;
//                pComplex[newindex].x = TwoChars2float(blockIQstartAddr + blockOffset);
//                pComplex[newindex].y = TwoChars2float(blockIQstartAddr + blockOffset + 2);
//            }
//        }
//    }
//    auto dataFillEnd = high_resolution_clock::now();
//
//    // 数据复制到设备
//    auto copyToDeviceStart = high_resolution_clock::now();
//    for (int i = 0; i < WAVE_NUM; i++) {
//        matrices[i].copyFromHost(NUM_PULSE, RANGE_NUM, pComplex + i * NUM_PULSE * RANGE_NUM);
//    }
//    auto copyToDeviceEnd = high_resolution_clock::now();
//
//    // 数据处理
//    auto processPulseStart = high_resolution_clock::now();
//    processPulseGroupData(matrices);
//    auto processPulseEnd = high_resolution_clock::now();
//
//    // 记录结束时间
//    auto endTime = high_resolution_clock::now();
//
//    // 输出时间统计
////    cout << "thread " << threadID << " timing details:" << endl;
////    cout << "  Data fill: "
////         << duration_cast<milliseconds>(dataFillEnd - dataFillStart).count() << " ms" << endl;
////    cout << "  Copy to device: "
////         << duration_cast<milliseconds>(copyToDeviceEnd - copyToDeviceStart).count() << " ms" << endl;
////    cout << "  Pulse group processing: "
////         << duration_cast<milliseconds>(processPulseEnd - processPulseStart).count() << " ms" << endl;
////    cout << "  Total: "
////         << duration_cast<milliseconds>(endTime - startTime).count() << " ms" << endl;
//
//    // 清理 headPositions 数据和设备内存
//    // headPositions[threadID].clear();
//    cout << "thread " << threadID << " process finished" << "  Total: "
//                                                         << duration_cast<milliseconds>(endTime - startTime).count() << " ms" << endl;;
//}




//void ThreadPool::processPulseGroupData(vector<CudaMatrix>& matrices) {
//    using namespace std::chrono;
//
//    auto startTime = high_resolution_clock::now();
////    cout << "Processing pulse group data..." << endl;
//
//    for (int i = 0; i < WAVE_NUM; i++) {
//        auto waveStartTime = high_resolution_clock::now();
//
//        /* Pulse Compression */
//        auto pulseCompressionStart = high_resolution_clock::now();
//        matrices[i].fft_N(NFFT);
//        matrices[i].elementWiseMul(PCcoefMatrix);
//        matrices[i].ifft();
//        auto PCres_Segment = matrices[i].extractSegment(numSamples - 2, RANGE_NUM);
//        auto pulseCompressionEnd = high_resolution_clock::now();
//
//        /* Coherent Integration */
//        auto coherentIntegrationStart = high_resolution_clock::now();
//        PCres_Segment.fft_by_col();
//        auto coherentIntegrationEnd = high_resolution_clock::now();
//
//        /* CFAR */
//        auto cfarStart = high_resolution_clock::now();
//        double Pfa = 1e-6;
//        int numGuardCells = 4;
//        int numRefCells = 20;
//        auto cfar_signal = PCres_Segment.cfar(Pfa, numGuardCells, numRefCells);
//        auto res_cfar = cfar_signal.max();
//        auto cfarEnd = high_resolution_clock::now();
//
//        auto waveEndTime = high_resolution_clock::now();
//
//        if (i == 0) {
//            // 输出单波形处理的时间统计
//            cout << "Wave " << i << " timing details:" << endl;
//            cout << "  Pulse Compression: "
//                 << duration_cast<microseconds >(pulseCompressionEnd - pulseCompressionStart).count() << " us" << endl;
//            cout << "  Coherent Integration: "
//                 << duration_cast<microseconds>(coherentIntegrationEnd - coherentIntegrationStart).count() << " us" << endl;
//            cout << "  CFAR: "
//                 << duration_cast<microseconds>(cfarEnd - cfarStart).count() << " us" << endl;
//            cout << "  Total: "
//                 << duration_cast<microseconds>(waveEndTime - waveStartTime).count() << " us" << endl;
//        }
//    }
//
////    auto endTime = high_resolution_clock::now();
////
////    // 输出总处理时间
////    cout << "Total processing time for pulse group data: "
////         << duration_cast<microseconds>(endTime - startTime).count() << " us" << endl;
//}


