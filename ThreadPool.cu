#include "ThreadPool.h"
#include "Config.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads, SharedQueue *sharedQueue) :
        stop(false),
        sharedQueue(sharedQueue),
        processingFlags(numThreads, false),
        conditionVariables(numThreads), mutexes(numThreads),
        headPositions(numThreads, std::vector<int>()),
        currentPos(numThreads, 0),
        numThreads(numThreads),
        inPacket(false),
        cur_thread_id(0),
        prevSeqNum(0) {
    // 初始化 conditionVariables 和 mutexes
    // 创建并初始化线程

    uint64Pattern = *(uint64_t *) pattern;

    for (int i = 0; i < numThreads; i++) {
        waveGroupProcessors.push_back(std::make_unique<WaveGroupProcessor>(WAVE_NUM, PULSE_NUM, NFFT));
    }

    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
    }
    threads.emplace_back(&ThreadPool::senderThread, this);

    if (debug_mode) {
        string input_file_name = dataPath.substr(dataPath.rfind('/') + 1, dataPath.rfind('.') - dataPath.rfind('/') - 1);

        debug_folder_path =  input_file_name +
        "_frame_" + to_string(start_frame) + "_" + to_string(end_frame) +
        "_wave_" + to_string(start_wave) + "_" + to_string(end_wave) +
        "_" + to_string(PULSE_NUM) + "x" + to_string(NFFT);

        if (debug_mode == 1) {
            debugFile.open(debug_folder_path, std::ios::binary);
        }
        else if (debug_mode == 2 && !std::filesystem::exists(debug_folder_path)) {
            std::filesystem::create_directories(debug_folder_path);
        }
    }
    // logFile = ofstream("error_log.txt", ios_base::app);
    logFile = ofstream("error_log.txt");
    cout << "Initial Finished" << endl;
}

ThreadPool::~ThreadPool() {
    stop = true;
    for (auto &cv: conditionVariables) cv.notify_all(); // 通知所有线程退出
    for (std::thread &thread: threads) {
        if (thread.joinable()) thread.join();
    }
    freeThreadMemory();
    debugFile.close();
    logFile.close();
}


void ThreadPool::freeThreadMemory() {
    for (size_t i = 0; i < numThreads; ++i) {
        checkCudaErrors(cudaFree(threadsMemory[i]));
        cudaStreamSynchronize(streams[i]); // 等待流中的所有操作完成
        cudaStreamDestroy(streams[i]);
    }
    threadsMemory.clear();
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
    auto& waveGroupProcessor_ = waveGroupProcessors[threadID];

    while (!stop) {
        waitForProcessingSignal(threadID);

        if (stop) break; // 退出循环

        // 调用处理函数，传递资源对象
        processData(waveGroupProcessor_, threadID);

        // 处理完毕后重置标志
        {
            std::lock_guard<std::mutex> lock(mutexes[threadID]);
            processingFlags[threadID] = false;
        }
    }
}


// 线程池中的数据处理函数
void ThreadPool::processData(std::unique_ptr<WaveGroupProcessor>& waveGroupProcessor, int threadID) {

    waveGroupProcessor->resetAddr();
    cout << "thread " << threadID << " start" << endl;
    static int taskCounter = 0;
    taskCounter++;
    int taskId = taskCounter;
    cout << "count:" << taskId << endl;

    // if (taskId != 1)
    // {
    //     return;
    // }
    int numHeads = headPositions[threadID].size();       // 2048
    int headLength = headPositions[threadID][1] - headPositions[threadID][0];
    int rangeNum = floor((headLength - DATA_OFFSET) / WAVE_NUM / 4.0);

    if (numHeads != PULSE_NUM  || rangeNum != RANGE_NUM) {
        cout << "numHeads:" << numHeads << " PULSE_NUM:" << PULSE_NUM << endl;
        cout << "rangeNum:" << rangeNum << " RANGE_NUM:" << RANGE_NUM << endl;
        throw std::runtime_error("The calculated range num is different from that is set");
    }

    waveGroupProcessor->getRadarParams();
    waveGroupProcessor->unpackData(headPositions[threadID].data());
    if (debug_mode == 1) {
        waveGroupProcessor->saveToDebugFile(taskId, debugFile);
    }
    else if (debug_mode == 2) {
        waveGroupProcessor->saveToDebugFile_new(taskId, debug_folder_path);
    }

    waveGroupProcessor->fullPipelineProcess();
    waveGroupProcessor->getResult();

    if (azi_densify_enable) {
        waveGroupProcessor->processAziDensify();
    }

    // 存储结果到共享 map，而不是直接发送
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        resultMap[taskId] = waveGroupProcessor->getParams();
        sender_cv_.notify_all();  // 通知发送线程
    }
    cout << "thread " << threadID << " process finished" << endl;
}

void ThreadPool::senderThread() {
    int nextToSend = 1;
    while (!stop) {
        std::unique_lock<std::mutex> lock(mapMutex);
        // 等待直到下一个预期结果可用
        sender_cv_.wait(lock, [&]{ return resultMap.find(nextToSend) != resultMap.end(); });

        // 发送结果
        sender.send(resultMap[nextToSend]);
        resultMap.erase(nextToSend);  // 删除已发送的结果
        nextToSend++;  // 更新下一个待发送的序号
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
    dataRateTracker.dataArrived();
    // std::cout << "Block index: " << block_index << std::endl << std::endl;

    unsigned int seqNum;

    static unsigned int prevIndexValue1;
    unsigned int indexValue; // 当前packet相对于1GB的起始地址
    unsigned long copyStartAddr = block_index * BLOCK_SIZE; // 当前Block相对于1GB的复制起始地址
    bool startFlag;
    for (int i = 0; i < 1024; i++) {
        size_t indexOffset = block_index * INDEX_SIZE + i * 4;
        if(dataSource_type == 0) {
            indexValue = *(unsigned int *) (sharedQueue->index_buffer + indexOffset);
        }
        else {
            indexValue = FourChars2Uint(reinterpret_cast<char*>(sharedQueue->index_buffer + indexOffset));
        }

        if (indexValue == 0)
        {
            logFile << "block_index:" << block_index << " indexValue:" << indexValue << endl;
            if (inPacket) {
                unsigned int copyEndAddr = (block_index + 1) * BLOCK_SIZE;
                memcpyDataToThread(copyStartAddr, copyEndAddr);
            }
            break;
        }

        // // 检查索引值是否在当前块范围内
        if (indexValue < block_index * BLOCK_SIZE || indexValue >= (block_index + 1) * BLOCK_SIZE) {
            cerr << "Error: Index value " << indexValue << " is out of bounds for block " << block_index
                 << " (valid range: [" << (block_index * BLOCK_SIZE) << ", " << ((block_index + 1) * BLOCK_SIZE - 1)
                 << "])" << endl;
            logFile  << "Error: Index value " << indexValue << " is out of bounds for block " << block_index
                     << " (valid range: [" << (block_index * BLOCK_SIZE) << ", " << ((block_index + 1) * BLOCK_SIZE - 1)
                     << "])" << endl;
            indexValue %= (BLOCK_SIZE * QUEUE_SIZE);
            // inPacket = false;
            // break;
        }

        if (indexValue + 20 >= (block_index + 1) * BLOCK_SIZE) {
            cerr << "Error: startFlag Index value " << indexValue << " exceed block " << block_index << " (valid range: <" <<
                  (block_index + 1) * BLOCK_SIZE - 1 << endl;
            logFile << "Error: startFlag Index value " << indexValue << " exceed block " << block_index << " (valid range: <" <<
                  (block_index + 1) * BLOCK_SIZE - 1 << endl;
        }

        // cout << "index:" << indexValue << endl;
        // Check pattern match
        uint64_t header = *(uint64_t *)(sharedQueue->buffer + indexValue);
        if (indexValue >= block_index * BLOCK_SIZE && header == uint64Pattern) {
            seqNum = *(uint32_t *) (sharedQueue->buffer + indexValue + SEQ_OFFSET);
            auto time_ms = *(uint32_t *) (sharedQueue->buffer + indexValue + 6 * 4);
            logFile << "seqNum: " << seqNum << " time:" << time_ms << endl;


            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0) {
                inPacket = false;
                std::cerr << "Error! Sequence number not continuous!" << prevIndexValue1 << ":" << prevSeqNum << " " << indexValue << ":" << seqNum << std::endl;
                std::time_t now = std::time(nullptr);
                std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                logFile << "[" << timebuf << "] " << "Error! Sequence number not continuous!" << prevIndexValue1 << ":" << prevSeqNum << " " << indexValue << ":" << seqNum << std::endl;
            }
            prevSeqNum = seqNum;
            prevIndexValue1 = indexValue;

            startFlag = *(uint8_t *) (sharedQueue->buffer + indexValue + 20) & 0x02;

            if (startFlag) {
                logFile << "========================" << endl;
                logFile << "startFlag: " << startFlag << endl;
                logFile << "Packege start. seqNum:" << seqNum << endl;
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
        }
    }
    sharedQueue->read_index = (sharedQueue->read_index + 1) % QUEUE_SIZE;
}

// 从startAddr到endAddr的数据拷贝给线程的独立空间，Addr是相对于共享内存的起始地址
void ThreadPool::memcpyDataToThread(unsigned int startAddr, unsigned int endAddr) {
    size_t data_size = endAddr - startAddr;
//    std::cout << "Copying " << copyLength / 1024 / 1024 << " MB to thread " << cur_thread_id << " :totally "
//              << (currentAddrOffset + copyLength) / 1024 / 1024 << " MB" << endl;

    if (data_size > BLOCK_SIZE) {
        std::cerr << "Error: Invalid copy range, startAddr: " << startAddr << ", data_size: " << data_size << std::endl;
        inPacket = false;
        return;
    }

    if (waveGroupProcessors[cur_thread_id]->copyRawData(sharedQueue->buffer + startAddr, data_size) != 0) {  // Ensure within buffer bounds
        std::cerr << "Copying " << data_size / 1024 / 1024 << " MB to thread " << cur_thread_id << endl;
        std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
        inPacket = false;
    }
}
