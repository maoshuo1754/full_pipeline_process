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
    radar_params_ = new RadarParams();

    for (int i = 0; i < numThreads; i++) {
        waveGroupProcessors.push_back(std::make_unique<WaveGroupProcessor>(WAVE_NUM, PULSE_NUM, NFFT));
    }

    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
    }

    if (debug_mode) {
        string input_file_name = dataPath.substr(dataPath.rfind('/') + 1, dataPath.rfind('.') - dataPath.rfind('/') - 1);

        string debug_file_path =  input_file_name +
        "_frame_" + to_string(start_frame) + "_" + to_string(end_frame) +
        "_pulse_" + to_string(start_wave) + "_" + to_string(end_wave) +
        "_" + to_string(PULSE_NUM) + "x" + to_string(NFFT);

        debugFile.open(debug_file_path, std::ios::binary);
    }

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
    static int count = 0;
    count++;
    int thisCount = count;
    cout << "count:" << thisCount << endl;

    // if (thisCount > 15)
    // {
    //     return;
    // }
    int numHeads = headPositions[threadID].size();       // 2048
    int headLength = headPositions[threadID][1] - headPositions[threadID][0];
    int rangeNum = floor((headLength - DATA_OFFSET) / WAVE_NUM / 4.0);

    // cout << "numHeads: " << numHeads << endl;
    // cout << "headLength: " << headLength << endl;
    if (numHeads != PULSE_NUM  || rangeNum != RANGE_NUM) {
        cout << "numHeads:" << numHeads << " PULSE_NUM:" << PULSE_NUM << endl;
        cout << "rangeNum:" << rangeNum << " RANGE_NUM:" << RANGE_NUM << endl;
        throw std::runtime_error("The calculated range num is different from that is set");
    }

    waveGroupProcessor->unpackData(headPositions[threadID].data());

    getRadarParams(waveGroupProcessor, thisCount);

    waveGroupProcessor->processPulseCompression(radar_params_->numSamples);
    waveGroupProcessor->processCoherentIntegration(radar_params_->scale);
    waveGroupProcessor->processCFAR();
    // waveGroupProcessor->cfar(radar_params_->numSamples);
    // waveGroupProcessor->cfar_by_col();
    waveGroupProcessor->processMaxSelection();

    waveGroupProcessor->getResult(radar_params_->h_max_results_, radar_params_->h_speed_channels_);
    sender.send(radar_params_);


    cout << "thread " << threadID << " process finished" << endl;
}

void ThreadPool::getRadarParams(std::unique_ptr<WaveGroupProcessor>& waveGroupProcessor, int frame) {
    static bool isInit = false;
    waveGroupProcessor->getPackegeHeader(radar_params_->rawMessage, DATA_OFFSET);

    if (debug_mode && frame >= start_frame && frame < end_frame)
    {
        writeToDebugFile(radar_params_->rawMessage, waveGroupProcessor->getData());
    }

    if (!isInit) {
        isInit = true;
        auto* packageArr = (uint32_t *)(radar_params_->rawMessage);

        auto freqPoint = packageArr[11] & 0x000000ff;
        radar_params_->lambda = c_speed / ((freqPoint * 10 + initCarryFreq) * 1e6);
        radar_params_->pulseWidth = (packageArr[13] & 0xfffff) / Fs_system; //5e-6
        radar_params_->PRT = packageArr[14] / Fs_system;  //120e-6
        auto fLFMStartWord = packageArr[16];
        radar_params_->bandWidth = (Fs_system - fLFMStartWord / pow(2.0f, 32) * Fs_system) * 2.0;

        double delta_v = radar_params_->lambda / radar_params_->PRT / PULSE_NUM / 2.0f; //两个滤波器之间的速度间隔
        double blind_v = radar_params_->lambda / radar_params_->PRT / 2.0f;

        for(int i = 0; i < PULSE_NUM; ++i){
            int v;
            if (i < PULSE_NUM / 2){
                v = static_cast<int>(std::round(delta_v * i * 100)) ;
            }
            else{
                v = static_cast<int>(std::round((delta_v * i - blind_v) * 100)) ;
            }
            radar_params_->chnSpeeds.push_back(v);
        }
        radar_params_->scale = 1.0f / sqrt(radar_params_->bandWidth * radar_params_->pulseWidth) / PULSE_NUM;
        radar_params_->getCoef();
    }

    waveGroupProcessor->getCoef(radar_params_->pcCoef, radar_params_->cfarCoef);
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

    unsigned int indexValue; // 当前packet相对于1GB的起始地址
    unsigned long copyStartAddr = block_index * BLOCK_SIZE; // 当前Block相对于1GB的复制起始地址
    bool startFlag;
    for (int i = 0; i < INDEX_SIZE / sizeof(unsigned int); i++) {
        size_t indexOffset = block_index * INDEX_SIZE + i * 4;
        if(dataSource_type == 0) {
            indexValue = *(unsigned int *) (sharedQueue->index_buffer + indexOffset);
        }
        else {
            indexValue = FourChars2Uint(reinterpret_cast<char*>(sharedQueue->index_buffer + indexOffset));
        }

        // cout << "index:" << indexValue << endl;
        // Check pattern match
        if (indexValue >= block_index * BLOCK_SIZE &&
            indexValue < (block_index + 1) * BLOCK_SIZE &&
            *(uint64_t *) (sharedQueue->buffer + indexValue) == uint64Pattern) {
            seqNum = *(uint32_t *) (sharedQueue->buffer + indexValue + SEQ_OFFSET);
            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0) {
                inPacket = false;
                std::cerr << "Error! Sequence number not continuous!" << prevIndexValue << " " << seqNum << std::endl;
        }
            prevSeqNum = seqNum;

            startFlag = *(uint8_t *) (sharedQueue->buffer + indexValue + 20) & 0x02;

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
    size_t data_size = endAddr - startAddr;
//    std::cout << "Copying " << copyLength / 1024 / 1024 << " MB to thread " << cur_thread_id << " :totally "
//              << (currentAddrOffset + copyLength) / 1024 / 1024 << " MB" << endl;

    if (waveGroupProcessors[cur_thread_id]->copyRawData(sharedQueue->buffer + startAddr, data_size) != 0) {  // Ensure within buffer bounds
        std::cerr << "Copying " << data_size / 1024 / 1024 << " MB to thread " << cur_thread_id << endl;
        std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
        inPacket = false;
    }
}

void ThreadPool::writeToDebugFile(unsigned char *rawMessage, const cufftComplex* d_data) {
    int oneWaveSize = PULSE_NUM * NFFT;
    int waveNum = end_wave - start_wave;

    auto* startAddr = d_data + start_wave * oneWaveSize;
    size_t size = waveNum * oneWaveSize;

    auto* h_data = new cufftComplex[size]; // 在主机上分配内存

    // 将数据从显存复制到主机内存
    cudaMemcpy(h_data, startAddr, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // 打开文件并以二进制方式写入
    auto rawMsg = reinterpret_cast<uint32*>(rawMessage);
    auto time = rawMsg[6] / 10 + 8*60*60*1000; // FPGA时间 //0.1ms->1ms + 8h

    debugFile.write(reinterpret_cast<char*>(&time), 4);
    debugFile.write(reinterpret_cast<char*>(h_data), size * sizeof(cufftComplex));

    delete[] h_data; // 释放主机内存
}

