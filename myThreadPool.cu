#include "myThreadPool.h"
#include <cuda_runtime.h>
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads, SharedQueue *sharedQueue)
        : stop(false), sharedQueue(sharedQueue), processingFlags(numThreads, false),
          conditionVariables(numThreads), mutexes(numThreads),
          headPositions(numThreads, std::vector<size_t>()), currentPos(numThreads, 0),
          currentAddrOffset(numThreads, 0), numThreads(numThreads), inPacket(false),
          cur_thread_id(0), prevSeqNum(0), unEnd(false) { // 初始化 conditionVariables 和 mutexes
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
    size_t* d_headPositions;
    bool* d_result;
    cudaMalloc(&d_headPositions, 300 * sizeof(size_t));
    cudaMalloc(&d_result, sizeof(bool));

    while (!stop) {
        waitForProcessingSignal(threadID);

        if (stop) break; // 退出循环

        processData(threadID, d_headPositions, d_result); // 处理 CUDA 内存中的数据

        // 处理完毕后重置标志
        {
            std::lock_guard<std::mutex> lock(mutexes[threadID]);
            processingFlags[threadID] = false;
        }
    }

    cudaFree(d_headPositions);
    cudaFree(d_result);
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

void ThreadPool::copyToThreadMemory() {
    int block_index = sharedQueue->read_index;
//    std::cout << "Block index: " << block_index << std::endl << std::endl;

    unsigned int seqNum;

    unsigned int indexValue, nextIndexValue, packetLength;
    unsigned long copyStartAddr = block_index * BLOCK_SIZE; // 相对于1GB的复制起始地址
    bool startFlag;

    for (int i = 0; i < 128; i++) {
        size_t indexOffset = block_index * INDEX_SIZE + i * 4;
        indexValue     = FourChars2Uint(sharedQueue->index_buffer + indexOffset);
        nextIndexValue = FourChars2Uint(sharedQueue->index_buffer + indexOffset + 4);

        // 当前包是这个BLOCK的最后一个包，并且到BLOCK末尾都没结束
        if (*(uint64_t *) (sharedQueue->buffer + nextIndexValue) != uint64Pattern){
            packetLength = indexValue - FourChars2Uint(sharedQueue->index_buffer + indexOffset - 4);
        }
        else{
            packetLength = nextIndexValue - indexValue;
        }

        // Check pattern match
        if (*(uint64_t *) (sharedQueue->buffer + indexValue) == uint64Pattern) {
            seqNum = FourChars2Uint(sharedQueue->buffer + indexValue + SEQ_OFFSET);
//            std::cout << "Sequence Number: " << seqNum << std::endl << "Index offset: " << indexOffset << ", Index value: " << indexValue << std::endl;;

            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0) {
                currentAddrOffset[cur_thread_id] = 0;
                inPacket = false;
                std::cerr << "Error! Sequence number not continuous!" << std::endl;
            }
            prevSeqNum = seqNum;

            startFlag = static_cast<uint8_t>(sharedQueue->buffer[indexValue + 23]) & 0x02;

            if (startFlag) {
                // 发送上一个脉组的数据
                if (inPacket) {
                    size_t copyLength = indexValue - copyStartAddr;
//                    std::cout << "Copying " << copyLength << " bytes from address " << copyStartAddr << " to thread memory at " << currentAddrOffset[cur_thread_id] << std::endl;

                    if ((currentAddrOffset[cur_thread_id] + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds
//                        memcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
//                               sharedQueue->buffer + copyStartAddr,
//                               copyLength);
                        cudaMemcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
                                   sharedQueue->buffer + copyStartAddr,
                                   copyLength,
                                   cudaMemcpyHostToDevice);

                        currentAddrOffset[cur_thread_id] += copyLength;
                    } else {
                        std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
                    }

                    notifyThread(cur_thread_id);
                    cur_thread_id = (cur_thread_id + 1) % numThreads;
                }

                // 初始化当前脉冲的参数
                inPacket = true;
                currentPos[cur_thread_id] = 0;
                currentAddrOffset[cur_thread_id] = 0;
                headPositions[cur_thread_id].clear();
                copyStartAddr = indexValue;
//                std::cout << "Start flag detected, starting at address: " << copyStartAddr << std::endl;
            }

            if (inPacket) {
                headPositions[cur_thread_id].push_back(currentPos[cur_thread_id]);
                currentPos[cur_thread_id] += packetLength;
            }

        } else {
            if (inPacket) {
                size_t copyLength = (block_index + 1) * BLOCK_SIZE - copyStartAddr;
//                std::cout << "Copying " << copyLength << " bytes at end of block. to thread memory at " << currentAddrOffset[cur_thread_id] << std::endl;

                if ((currentAddrOffset[cur_thread_id] + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds
//                    memcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
//                           sharedQueue->buffer + copyStartAddr,
//                           copyLength);

                    cudaMemcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
                               sharedQueue->buffer + copyStartAddr,
                               copyLength,
                               cudaMemcpyHostToDevice
                               //streams[cur_thread_id]
                               );



                    currentAddrOffset[cur_thread_id] += copyLength;
                } else {
                    std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
                }
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

void ThreadPool::processData(int threadID, size_t* d_headPositions, bool* d_result) {
    int numHeads = headPositions[threadID].size();

    cout << "thread " << threadID << " start processing " << endl;
    cout << headPositions[threadID][0] << endl;
    cout << headPositions[threadID][1] << endl;



//    bool result = true;
//    cout << "thread " << threadID << " start processing ";
//    cout << "headPositions[threadID].size()：" << numHeads << endl;
//
//    cudaMemcpy(d_headPositions, headPositions[threadID].data(), numHeads * sizeof(size_t), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_result, &result, sizeof(bool), cudaMemcpyHostToDevice);
//
//    verifySequenceNumbers<<<1, numHeads-1>>>(threadsMemory[threadID], d_headPositions, numHeads, d_result);
//    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
//
//    if (result) {
////        cout << "seq num verify success!" << endl;
//        char tmp[4];
//        unsigned int startSeq, endSeq;
//        cudaMemcpy(tmp, threadsMemory[threadID] + headPositions[threadID][0] + 16, sizeof (unsigned int), cudaMemcpyDeviceToHost);
//        startSeq = FourChars2Uint(tmp);
//
//        cudaMemcpy(tmp, threadsMemory[threadID] + headPositions[threadID][numHeads-1] + 16, sizeof (unsigned int), cudaMemcpyDeviceToHost);
//        endSeq = FourChars2Uint(tmp);
//
//        cout << "seq num verify success! From " << startSeq << " to " << endSeq << endl;
//
//    } else {
//        cerr << "seq num verify failed!" << endl;
//    }


//    char* data = threadsMemory[threadID];
//    bool success = true;
//    size_t head, prevHead;
//    unsigned int startSeq, endSeq;
//    for (int i = 0; i < headPositions[threadID].size(); i++) {
//
//        head = headPositions[threadID][i];
//
//        if (i == 0){
//            startSeq = FourChars2Uint(data + head + 16);
//        }
//        if (i == headPositions[threadID].size() - 1){
//            endSeq = FourChars2Uint(data + head + 16);
//        }
//        if (i > 0){
//            if (FourChars2Uint(data + head + 16) - FourChars2Uint(data + prevHead + 16) != 1){
//                success = false;
//                break;
//            }
//        }
//        prevHead = head;
//    }
//
//    if (!success){
//        cerr << "seq num verify failed!" << endl;
//    }
//    else{
//        cout << "seq num verify success! From " << startSeq << " to " << endSeq << endl;
//    }
    headPositions[threadID].clear();
}


