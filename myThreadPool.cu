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
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
        threadsMemory.emplace_back(new char[THREADS_MEM_SIZE]);
    }
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

void ThreadPool::copyToThreadMemory() {
    int block_index = sharedQueue->read_index;
    std::cout << "Block index: " << block_index << std::endl << std::endl;

    unsigned int seqNum;

    unsigned int indexValue, nextIndexValue;
    unsigned long copyStartAddr = block_index * BLOCK_SIZE; // 相对于1GB的复制起始地址
    bool startFlag, endFlag;

    for (int i = 0; i < 128; i++) {
        size_t indexOffset = block_index * INDEX_SIZE + i * 4;
        indexValue     = FourChars2Uint(sharedQueue->index_buffer + indexOffset);
        nextIndexValue = FourChars2Uint(sharedQueue->index_buffer + indexOffset + 4);
        auto packetLength =  nextIndexValue - indexValue;

        // 当前包是这个BLOCK的最后一个包，并且到BLOCK末尾都没结束
        if (*(uint64_t *) (sharedQueue->buffer + nextIndexValue) != *(uint64_t *) pattern){
            packetLength = indexValue - FourChars2Uint(sharedQueue->index_buffer + indexOffset - 4);
        }

            // Check pattern match
        if (*(uint64_t *) (sharedQueue->buffer + indexValue) == *(uint64_t *) pattern) {
            seqNum = FourChars2Uint(sharedQueue->buffer + indexValue + SEQ_OFFSET);
//            std::cout << "Sequence Number: " << seqNum << std::endl << "Index offset: " << indexOffset << ", Index value: " << indexValue << std::endl;;

            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0) {
                currentAddrOffset[cur_thread_id] = 0;
                headPositions[cur_thread_id].clear();
                inPacket = false;
                std::cerr << "Error! Sequence number not continuous!" << std::endl;
            }
            prevSeqNum = seqNum;

            uint8_t flagByte = static_cast<uint8_t>(sharedQueue->buffer[indexValue + 23]);
            startFlag = flagByte & 0x02;
            endFlag = flagByte & 0x01;

            if (startFlag) {
                inPacket = true;
                currentPos[cur_thread_id] = 0;
                currentAddrOffset[cur_thread_id] = 0;
                copyStartAddr = indexValue;
                std::cout << "Start flag detected, starting at address: " << copyStartAddr << std::endl;
            }

            if (inPacket) {
                headPositions[cur_thread_id].push_back(currentPos[cur_thread_id]);
                currentPos[cur_thread_id] += packetLength;
            }

            if (endFlag && inPacket) {
                size_t copyLength = indexValue + packetLength - copyStartAddr;
                std::cout << "Copying " << copyLength << " bytes from address " << copyStartAddr << " to thread memory at " << currentAddrOffset[cur_thread_id] << std::endl;

                if ((currentAddrOffset[cur_thread_id] + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds
                    memcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
                           sharedQueue->buffer + copyStartAddr,
                           copyLength);
                    currentAddrOffset[cur_thread_id] += copyLength;
                } else {
                    std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
                }

                notifyThread(cur_thread_id);
                cur_thread_id = (cur_thread_id + 1) % numThreads;
                inPacket = false;
            }
        } else {
            if (inPacket) {
                size_t copyLength = (block_index + 1) * BLOCK_SIZE - copyStartAddr;
                std::cout << "Copying " << copyLength << " bytes at end of block. to thread memory at " << currentAddrOffset[cur_thread_id] << std::endl;

                if ((currentAddrOffset[cur_thread_id] + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds
                    memcpy(threadsMemory[cur_thread_id] + currentAddrOffset[cur_thread_id],
                           sharedQueue->buffer + copyStartAddr,
                           copyLength);
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

void ThreadPool::processData(int threadID) {
    cout << "thread " << threadID << " start processing" << endl;
    cout << "headPositions[threadID].size()：" << headPositions[threadID].size() << endl;
    char* data = threadsMemory[threadID];
    bool success = true;
    size_t head, prevHead;
    for (int i = 0; i < headPositions[threadID].size(); i++) {
        head = headPositions[threadID][i];
        if (i > 0){
            if (FourChars2Uint(data + head + 16) - FourChars2Uint(data + prevHead + 16) != 1){
                success = false;
                break;
            }
        }
        prevHead = head;
    }
    if (!success){
        cerr << "seq num verify failed!" << endl;
    }
    else{
        cout << "seq num verify success!" << endl;
    }
    headPositions[threadID].clear();
}


