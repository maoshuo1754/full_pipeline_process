#include "myThreadPool.h"
#include <cuda_runtime.h>
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads, SharedQueue *sharedQueue)
        : stop(false), sharedQueue(sharedQueue), processingFlags(numThreads, false),
          conditionVariables(numThreads), mutexes(numThreads),
          headPositions(numThreads, std::vector<size_t>()), currentPos(numThreads, 0),
          currentAddrOffset(numThreads, 0), numThreads(numThreads), inPacket(false),
          cur_thread_id(0), prevSeqNum(0) { // 初始化 conditionVariables 和 mutexes
    // 创建并初始化线程

    logFile = ofstream("error_log.txt", ios_base::app);
    uint64Pattern = *(uint64_t *) pattern;
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
        threadsMemory.emplace_back(new char[THREADS_MEM_SIZE]);
    }

    initPCcoefMatrix();

//    allocateThreadMemory();
    cout << "Initial Finished" << endl;
}

ThreadPool::~ThreadPool() {
    stop = true;
    for (auto &cv: conditionVariables) cv.notify_all(); // 通知所有线程退出
    for (std::thread &thread: threads) {
        if (thread.joinable()) thread.join();
    }

    for (auto ptr: threadsMemory) {
        delete[] ptr;
    }

    logFile.close();
//    freeThreadMemory();
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
        cudaStreamCreate(&stream);
        streams.push_back(stream);
    }
}

void ThreadPool::freeThreadMemory() {
    for (size_t i = 0; i < numThreads; ++i) {
        cudaFree(threadsMemory[i]);
        cudaStreamDestroy(streams[i]);
    }
    threadsMemory.clear();
    streams.clear();
}

void ThreadPool::initPCcoefMatrix() {
//    double C = 3e8;
    double BandWidth = 15e6;
    double PulseWidth = 2e-6;
    double Fs = 31.25e6;
//    double PRT = 100e-6;

    numSamples = round(PulseWidth * Fs);
    NFFT = nextpow2(RANGE_NUM + numSamples - 1);

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
    cufftComplex* pComplex = new cufftComplex[WAVE_NUM * NUM_PULSE * RANGE_NUM];

//    cout << "buffer size:" << WAVE_NUM * NUM_PULSE * RANGE_NUM * sizeof (cufftComplex) << endl;

    // 提前在显存分配好内存，将上述脉组数据拷贝进来处理
    vector<CudaMatrix> matrices(WAVE_NUM, CudaMatrix(NUM_PULSE, RANGE_NUM));

    while (!stop) {
        waitForProcessingSignal(threadID);

        if (stop) break; // 退出循环

        processData(threadID, pComplex, matrices); // 处理 CUDA 内存中的数据

        // 处理完毕后重置标志
        {
            std::lock_guard<std::mutex> lock(mutexes[threadID]);
            processingFlags[threadID] = false;
        }
    }

    // 释放内存
    delete[] pComplex;
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

    for (int i = 0; i < 512; i++) {
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
                currentAddrOffset[cur_thread_id] = 0;
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
//                    std::cout << "Copying " << copyLength << " bytes from address " << copyStartAddr << " to thread memory at " << currentAddrOffset[cur_thread_id] << std::endl;

    if ((currentAddrOffset[cur_thread_id] + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds
        memcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
               sharedQueue->buffer + startAddr,
               copyLength);

//        // 内存拷贝到显存
//        cudaMemcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
//                   sharedQueue->buffer + startAddr,
//                   copyLength,
//                   cudaMemcpyHostToDevice);

        currentAddrOffset[cur_thread_id] += copyLength;
    } else {
        std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
    }
}


unsigned int ThreadPool::FourChars2Uint(const char* startAddr){
    return     static_cast<uint8_t>(startAddr[0]) << 24
               | static_cast<uint8_t>(startAddr[1]) << 16
               | static_cast<uint8_t>(startAddr[2]) << 8
               | static_cast<uint8_t>(startAddr[3]);
}

__device__ unsigned int FourChars2Uint(const char* startAddr) {
    return static_cast<uint8_t>(startAddr[0]) << 24
           | static_cast<uint8_t>(startAddr[1]) << 16
           | static_cast<uint8_t>(startAddr[2]) << 8
           | static_cast<uint8_t>(startAddr[3]);
}

float TwoChars2float(const char* startAddr) {
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
void ThreadPool::processData(int threadID, cufftComplex* pComplex, vector<CudaMatrix>& matrices) {
    cout << "thread " << threadID << " start" << endl;
    int numHeads = headPositions[threadID].size();
    int headLength = headPositions[threadID][1] - headPositions[threadID][0];
    int rangeNum = floor((headLength - 33 * 4) / WAVE_NUM / 4.0);
//    cout << numHeads << " " << numHeads << endl;
//    cout << sizeof(cufftComplex) << endl;

//    cout << headPositions[threadID][1] - headPositions[threadID][0] << endl;

    for (int idx = 0; idx < numHeads; ++idx) {
        auto blockIQstartAddr = threadsMemory[threadID] + headPositions[threadID][idx] + 33 * 4; // 计算数据起始地址
//        if (idx == 1 &&threadID == 1) printHex(blockIQstartAddr, 4*50);
        for (int i = 0; i < rangeNum; i++) {
            for (int j = 0; j < WAVE_NUM; j++) {
                int blockOffset = i * WAVE_NUM * 4 + j * 4; // 每个数据块的偏移
                auto newindex = j * NUM_PULSE * RANGE_NUM + idx * RANGE_NUM + i;  // 计算新的索引位置
                pComplex[newindex].x = TwoChars2float(blockIQstartAddr + blockOffset);
                pComplex[newindex].y = TwoChars2float(blockIQstartAddr + blockOffset + 2);
            }
        }
    }


    for (int i = 0; i < WAVE_NUM; i++) {
        matrices[i].copyFromHost(NUM_PULSE, RANGE_NUM, pComplex + i * NUM_PULSE * RANGE_NUM);
    }

//    processPulseGroupData(matrices);

    // 清理 headPositions 数据和设备内存
    // headPositions[threadID].clear();
    cout << "thread " << threadID << " process finished" << endl;
}

void ThreadPool::processPulseGroupData(vector<CudaMatrix>& matrices) {
    for (int i = 0; i < WAVE_NUM; i++) {

        /*Pulse Compression*/

//        PCcoefMatrixFFT.print(0);
        matrices[i].fft_N(NFFT);

        matrices[i].elementWiseMul(PCcoefMatrix);
        matrices[i].ifft();

        auto PCres_Segment = matrices[i].extractSegment(numSamples-2, RANGE_NUM);

        /*coherent integration*/
        PCres_Segment.fft_by_col();

        /*cfar*/
        double Pfa = 1e-6;
        int numGuardCells = 4;
        int numRefCells = 20;
        auto cfar_signal = PCres_Segment.cfar(Pfa, numGuardCells, numRefCells);
        auto res_cfar = cfar_signal.max();
//        res_cfar.printLargerThan0();
    }
}

