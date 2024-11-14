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
    uint64Pattern = *(uint64_t *) pattern;
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
//        threadsMemory.emplace_back(new char[THREADS_MEM_SIZE]);
    }
    allocateThreadMemory();
    cout << "Initial Finished" << endl;
}

ThreadPool::~ThreadPool() {
    stop = true;
    for (auto &cv: conditionVariables) cv.notify_all(); // 通知所有线程退出
    for (std::thread &thread: threads) {
        if (thread.joinable()) thread.join();
    }

//    for (auto ptr: threadsMemory) {
//        delete[] ptr;
//    }
    freeThreadMemory();
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



    while (!stop) {
        waitForProcessingSignal(threadID);

        if (stop) break; // 退出循环

        processData(threadID); // 处理 CUDA 内存中的数据

        // 处理完毕后重置标志
        {
            std::lock_guard<std::mutex> lock(mutexes[threadID]);
            processingFlags[threadID] = false;
        }
    }


}


void ThreadPool::notifyThread(int threadID) {
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

void ThreadPool::memcpyDataToThread(unsigned int startAddr, unsigned int endAddr){
    size_t copyLength = endAddr - startAddr;
//                    std::cout << "Copying " << copyLength << " bytes from address " << copyStartAddr << " to thread memory at " << currentAddrOffset[cur_thread_id] << std::endl;

    if ((currentAddrOffset[cur_thread_id] + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds
//        memcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
//               sharedQueue->buffer + copyStartAddr,
//               copyLength);

        // 内存拷贝到显存
        cudaMemcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
                   sharedQueue->buffer + startAddr,
                   copyLength,
                   cudaMemcpyHostToDevice);

        currentAddrOffset[cur_thread_id] += copyLength;
    } else {
        std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
    }
}

void ThreadPool::copyToThreadMemory() {
    int block_index = sharedQueue->read_index;
//    std::cout << "Block index: " << block_index << std::endl << std::endl;

    unsigned int seqNum;

    unsigned int indexValue; // 当前packet相对于1GB的起始地址
    unsigned long copyStartAddr = block_index * BLOCK_SIZE; // 当前Block相对于1GB的复制起始地址
    bool startFlag;

    for (int i = 0; i < 128; i++) {
        size_t indexOffset = block_index * INDEX_SIZE + i * 4;
        indexValue = FourChars2Uint(sharedQueue->index_buffer + indexOffset);

        // Check pattern match
        if (*(uint64_t *) (sharedQueue->buffer + indexValue) == uint64Pattern) {
            seqNum = FourChars2Uint(sharedQueue->buffer + indexValue + SEQ_OFFSET);
            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0) {
                inPacket = false;
                std::cerr << "Error! Sequence number not continuous!" << std::endl;
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



//float TwoChars2float(const char* startAddr) {
//    return static_cast<float>(static_cast<uint8_t>(startAddr[0]) << 8
//                              | static_cast<uint8_t>(startAddr[1]));
//}

__global__ void verifySequenceNumbers(const char* data, const size_t* headPositions, int numHeads, bool* result) {
    int idx = threadIdx.x;

    if (idx < numHeads - 1) {
        // 获取当前和下一个包头的序列号
        unsigned int currentSeq = FourChars2Uint(data + headPositions[idx] + 16);
        unsigned int nextSeq = FourChars2Uint(data + headPositions[idx + 1] + 16);

//        printf("%d %d\n", currentSeq, nextSeq);
        // 检查序列号连续性
        if (nextSeq - currentSeq != 1) {
            *result = false;
        }
    }
}

// TwoChars2float 函数，将两个字符转换为 float 类型
__device__ float TwoChars2float(const char* startAddr) {
    return static_cast<float>(  static_cast<uint8_t>(startAddr[0]) << 8
                              | static_cast<uint8_t>(startAddr[1]));
}

// 核函数：unpackDatabuf2CudaMatrices
// 该核函数用于将数据从字符数组转换为 cufftComplex 数组
__global__ void unpackDatabuf2CudaMatrices(const char* data, const size_t* headPositions,
                                           int* d_numHeads, int* d_rangeNum, cufftComplex* pComplex) {

    int idx = threadIdx.x;
    int numHeads = *d_numHeads;  // 获取头的数量

    if (idx < numHeads) {
        int rangeNum = *d_rangeNum;  // 获取 rangeNum 的值
        auto blockIQstartAddr = data + headPositions[idx] + 19 * 4; // 计算数据起始地址

        // 遍历 rangeNum 和 WAVE_NUM，按块处理数据
        for (int i = 0; i < rangeNum; i++) {
            for (int j = 0; j < WAVE_NUM; j++) {
                int blockOffset = i * WAVE_NUM * 4 + j * 4; // 每个数据块的偏移
                auto newindex = j * numHeads * rangeNum + idx * rangeNum + i;  // 计算新的索引位置

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

// 检查 CUDA 错误的函数
void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(result));
    }
}

// 线程池中的数据处理函数
void ThreadPool::processData(int threadID) {
    size_t* d_headPositions;
    int numHeads = headPositions[threadID].size();

    // 分配并复制 headPositions 数据到设备内存
    cudaMalloc(&d_headPositions, 300 * sizeof(size_t));
    cudaMemcpy(d_headPositions, headPositions[threadID].data(), numHeads * sizeof(size_t), cudaMemcpyHostToDevice);

    cout << "thread " << threadID << " start processing " << numHeads << endl;

    // 计算 headLength 和 rangeNum
    int headLength = headPositions[threadID][1] - headPositions[threadID][0];
    int rangeNum = floor((headLength - 19 * 4) / 32.0 / 4.0);

    // 将 rangeNum 和 numHeads 复制到设备内存
    int* d_rangeNum;
    cudaMalloc(&d_rangeNum, sizeof(int));
    cudaMemcpy(d_rangeNum, &rangeNum, sizeof(int), cudaMemcpyHostToDevice);

    int* d_numHeads;
    cudaMalloc(&d_numHeads, sizeof(int));
    cudaMemcpy(d_numHeads, &numHeads, sizeof(int), cudaMemcpyHostToDevice);

    // 创建 CudaMatrix 对象和 pComplex 指针
    vector<CudaMatrix> matrices(WAVE_NUM);
    cufftComplex* pComplex;

    // 分配内存给 pComplex
    checkCudaErrors(cudaMalloc(&pComplex, WAVE_NUM * sizeof(cufftComplex) * numHeads * rangeNum));

    // 启动 CUDA 核函数进行数据处理
    unpackDatabuf2CudaMatrices<<<1, numHeads>>>(threadsMemory[threadID], d_headPositions, d_numHeads, d_rangeNum, pComplex);

    // 同步 CUDA 设备
    checkCudaErrors(cudaDeviceSynchronize());

    // 释放内存
    checkCudaErrors(cudaFree(pComplex));

    // 清理 headPositions 数据和设备内存
    headPositions[threadID].clear();
    cudaFree(d_headPositions);
    cudaFree(d_rangeNum);

    cout << "thread " << threadID << " process finished" << endl;
}

