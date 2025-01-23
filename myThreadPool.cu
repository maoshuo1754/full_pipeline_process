#include "myThreadPool.h"
#include "Config.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

ThreadPool::ThreadPool(size_t numThreads, SharedQueue *sharedQueue) :
        stop(false),
        sharedQueue(sharedQueue),
        processingFlags(numThreads, false),
        conditionVariables(numThreads), mutexes(numThreads),
        headPositions(numThreads, std::vector<int>()),
        currentPos(numThreads, 0),
        currentAddrOffset(0),
        numThreads(numThreads),
        inPacket(false),
        cur_thread_id(0),
        prevSeqNum(0),
        PcCoefMatrix(1, NFFT),
        sender() {
    // 初始化 conditionVariables 和 mutexes
    // 创建并初始化线程

    logFile = ofstream("error_log.txt", ios_base::app);

    allocateThreadMemory();

    uint64Pattern = *(uint64_t *) pattern;

    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::threadLoop, this, i);
    }

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
        unsigned char *d_memory = nullptr;
        checkCudaErrors(cudaMalloc((void **) &d_memory, THREADS_MEM_SIZE));  // 在显存中分配内存
        threadsMemory.emplace_back(d_memory);

        cudaStream_t stream;
        checkCudaErrors(cudaStreamCreate(&stream));
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
    ThreadPoolResources resources(threadID, streams[threadID]); // 创建资源对象

    while (!stop) {
        waitForProcessingSignal(threadID);

        if (stop) break; // 退出循环

        // 调用处理函数，传递资源对象
        processData(resources);

        // 处理完毕后重置标志
        {
            std::lock_guard<std::mutex> lock(mutexes[threadID]);
            processingFlags[threadID] = false;
        }
    }
}

__global__ void processKernel(unsigned char *threadsMemory, cufftComplex *pComplex,
                              const int *headPositions, int numHeads, int rangeNum) {
    // 获取线程和网格索引
    int headIdx = blockIdx.z;  // 每个block.z处理一个头位置
    int rangeIdx = blockIdx.y * blockDim.x + threadIdx.x; // 每个线程处理一个距离单元
    int beamIdx = blockIdx.x;  // 每个block.x处理一个波束

    // 检查索引是否越界
    if (headIdx < numHeads && rangeIdx < rangeNum && beamIdx < WAVE_NUM) {
        // 计算头位置的起始地址
        int headOffset = headPositions[headIdx];
        unsigned char *blockIQstartAddr = threadsMemory + headOffset + DATA_OFFSET;
        // 计算当前数据块的偏移和新索引
        int blockOffset = rangeIdx * WAVE_NUM * 4 + beamIdx * 4;
        int newIndex = beamIdx * NUM_PULSE * NFFT + headIdx * NFFT + rangeIdx;

        // 提取IQ数据并存储到结果数组
        pComplex[newIndex].x = *(int16_t *) (blockIQstartAddr + blockOffset + 2);
        pComplex[newIndex].y = *(int16_t *) (blockIQstartAddr + blockOffset);
    }
}


void ThreadPool::generatePCcoefMatrix(unsigned char *rawMessage, cufftHandle &pcPlan, cudaStream_t _stream) {
    static double prevPRT = -1;
    static double prevPulseWidth = -1;
    static double prevfLFMStartWord = -1;

    if (*(uint64_t *) (rawMessage) != uint64Pattern) {
        cerr << "data error" << endl;
        return;
    }
//    double BandWidth = 6e6;
//    double PulseWidth = 5e-6;

    auto PRT = *(uint32_t *) (rawMessage + 14 * 4) / Fs_system; //120e-6
    pulseWidth = ((*(uint32_t *) (rawMessage + 13 * 4)) & 0xfffff) / Fs_system; //5e-6

    auto fLFMStartWord = *(uint32_t *) (rawMessage + 16 * 4);

    if (!isEqual(PRT, prevPRT) || !isEqual(pulseWidth, prevPulseWidth) || !isEqual(fLFMStartWord, prevfLFMStartWord)) {
        cout << "Param changed, regenerate pulse compress coefficient." << endl;
        numSamples = round(pulseWidth * Fs);  // 31.25e6
        Bandwidth = (Fs_system - fLFMStartWord / pow(2.0f, 32) * Fs_system) * 2.0;
        int freqPoint = (*(uint32_t *)(rawMessage + 12 * 4) & 0x00000fff);
        freqPoint = 3;
        double lambda = c_speed / ((freqPoint * 10 + 9600) * 1e6);

        cout << "Bandwidth:" << Bandwidth << endl;
        cout << "PulseWidth:" << pulseWidth << endl;
        cout << "lambda:" << lambda << endl;


        chnSpeeds.clear();
        double delta_v = lambda / PRT / NUM_PULSE / 2.0f; //两个滤波器之间的速度间隔
        double blind_v = lambda / PRT / 2.0f;

        for(int i = 0; i < NUM_PULSE; ++i){
            int v;
            if (i < NUM_PULSE / 2){
                v = static_cast<int>(std::round(delta_v * i * 100)) ;
            }
            else{
                v = static_cast<int>(std::round((delta_v * i - blind_v) * 100)) ;
            }
            chnSpeeds.push_back(v);
        }

//        for(int i = 1930; i < 1940; i++) {
//                cout << "speed channel" << i << " " << chnSpeeds[i] << endl;
//        }


        vector<cufftComplex> PcCoef = PCcoef(Bandwidth, pulseWidth, Fs, NFFT);
        PcCoefMatrix.copyFromHost(_stream, 1, NFFT, PcCoef.data());
        PcCoefMatrix.fft(pcPlan);
        prevPRT = PRT;
        prevfLFMStartWord = fLFMStartWord;
        prevPulseWidth = pulseWidth;
    }
}

// 线程池中的数据处理函数
void ThreadPool::processData(ThreadPoolResources &resources) {
    int threadID = resources.threadID;
    cout << "thread " << threadID << " start" << endl;

    checkCudaErrors(cudaMemsetAsync(resources.pComplex_d, 0, sizeof(cufftComplex) * WAVE_NUM * NUM_PULSE * NFFT,
                                    resources.stream));
    int numHeads = headPositions[threadID].size();       // 2048
    int headLength = headPositions[threadID][1] - headPositions[threadID][0];
    int rangeNum = floor((headLength - DATA_OFFSET) / WAVE_NUM / 4.0);

//        cout << "numHeads: " << numHeads << endl;
//        cout << "headLength: " << headLength << endl;
    if (rangeNum != RANGE_NUM) {
        throw std::runtime_error("The calculated range num is different from that is set");
    }

    // 头的位置拷贝到显存
    checkCudaErrors(cudaMemcpyAsync(resources.pHeadPositions_d, headPositions[threadID].data(), numHeads * sizeof(int),
                                    cudaMemcpyHostToDevice,
                                    streams[threadID]));

    // 包头信息拷贝回内存(仅第一个包头)
    checkCudaErrors(cudaMemcpyAsync(resources.rawMessage, threadsMemory[threadID], sizeof(resources.rawMessage),
                                    cudaMemcpyDeviceToHost,
                                    streams[threadID]));

    generatePCcoefMatrix(resources.rawMessage, resources.pcPlan, streams[threadID]);

    // 计算 gridDim 的大小
    dim3 gridDim1(WAVE_NUM, (rangeNum + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, numHeads);

    // 在gpu解包
    processKernel<<<gridDim1, CUDA_BLOCK_SIZE, 0, streams[threadID]>>>(threadsMemory[threadID], resources.pComplex_d,
                                                                       resources.pHeadPositions_d, numHeads, rangeNum);

    processPulseGroupData(resources, rangeNum);
    // 选大结果拷贝回内存
    checkCudaErrors(cudaMemcpyAsync(resources.pMaxRes_h, resources.pMaxRes_d, sizeof(cufftComplex) * CAL_WAVE_NUM * NFFT,
                            cudaMemcpyDeviceToHost,
                            streams[threadID]));

    // 速度通道拷贝回内存
    checkCudaErrors(cudaMemcpyAsync(resources.pSpeed_h, resources.pSpeed_d, sizeof(int) * CAL_WAVE_NUM * NFFT,
                                    cudaMemcpyDeviceToHost,
                                    streams[threadID]));


    cudaStreamSynchronize(streams[threadID]); // 等待流中的拷贝操作完成

    sender.send(resources.rawMessage, resources.pMaxRes_h, chnSpeeds, resources.pSpeed_h, numSamples, rangeNum);

    cout << "thread " << threadID << " process finished" << endl;
}


// 在GPU处理一个脉组的所有波束的数据，全流程处理，包括脉压、积累、CFAR、选大。
void ThreadPool::processPulseGroupData(ThreadPoolResources &resources, int rangeNum) {
    static int count = 0;
    count++;
    cout << "count:" << count << endl;
    int threadID = resources.threadID;
    auto &matrices = resources.IQmatrices;
    auto &CFAR_res = resources.CFAR_res;
    auto &Max_res = resources.Max_res;

    float scale = 1.0f / sqrt(Bandwidth * pulseWidth) / NUM_PULSE / RANGE_NUM;
    // for (int i = 0; i < CAL_WAVE_NUM; i++) {
    for (int i = 18; i < 21; i++) {
//        string filename = "data" + to_string(i) + "_max.txt";
        /*Pulse Compression*/
        matrices[i].fft(resources.rowPlan);

        matrices[i].rowWiseMul(PcCoefMatrix, streams[threadID]);

        matrices[i].ifft(streams[threadID], resources.rowPlan);

        // 归一化，同时连ifft的归一化一起做了
        matrices[i].scale(streams[threadID], scale);

//        IQmatrices[i].MTI(streams[threadID], 3);

        /*coherent integration*/
        for (int j = 0; j < INTEGRATION_TIMES; j++) {
            matrices[i].fft_by_col(resources.colPlan);
        }

        cudaStreamSynchronize(streams[threadID]); // 等待流中的拷贝操作完成
        if (count == 1) {
            string name = "pulse_" + to_string(count) + "_wave_" + to_string(i) +  + ".txt";
            matrices[i].writeMatTxt(name);
            cout << name << " write finished" << endl;
        }
        
        /*cfar*/
        matrices[i].abs(streams[threadID]);
        matrices[i].cfar(CFAR_res[i], streams[threadID], Pfa, numGuardCells, numRefCells, numSamples - 1,
                         numSamples + rangeNum);

        auto* pSpeedChannels = resources.pSpeed_d + i * NFFT;



        CFAR_res[i].max(Max_res[i], pSpeedChannels, streams[threadID]);
        Max_res[i].scale(streams[threadID], 1.0f / normFactor * 255);
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
    // std::cout << "Block index: " << block_index << std::endl << std::endl;

    unsigned int seqNum;

    unsigned int indexValue; // 当前packet相对于1GB的起始地址
    unsigned long copyStartAddr = block_index * BLOCK_SIZE; // 当前Block相对于1GB的复制起始地址
    bool startFlag;
    for (int i = 0; i < INDEX_SIZE / sizeof(unsigned int); i++) {
        size_t indexOffset = block_index * INDEX_SIZE + i * 4;
        indexValue = *(unsigned int *) (sharedQueue->index_buffer + indexOffset);
        // indexValue = FourChars2Uint(reinterpret_cast<char*>(sharedQueue->index_buffer + indexOffset));
        // cout << "index:" << indexValue << endl;
        // Check pattern match
        if (indexValue >= block_index * BLOCK_SIZE &&
            indexValue < (block_index + 1) * BLOCK_SIZE &&
            *(uint64_t *) (sharedQueue->buffer + indexValue) == uint64Pattern) {
            seqNum = *(uint32_t *) (sharedQueue->buffer + indexValue + SEQ_OFFSET);
            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0) {
                inPacket = false;
                currentAddrOffset = 0;
                std::cerr << "Error! Sequence number not continuous!" << prevIndexValue << " " << seqNum << std::endl;
                std::time_t now = std::time(nullptr);
                std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                logFile << "[" << timebuf << "] " << "Error! Sequence number not continuous!" << std::endl;
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
//    std::cout << "Copying " << copyLength / 1024 / 1024 << " MB to thread " << cur_thread_id << " :totally "
//              << (currentAddrOffset + copyLength) / 1024 / 1024 << " MB" << endl;

    if ((currentAddrOffset + copyLength) <= THREADS_MEM_SIZE) {  // Ensure within buffer bounds

//        // 内存拷贝到显存
        checkCudaErrors(cudaMemcpyAsync(threadsMemory[cur_thread_id] + currentAddrOffset,
                                        sharedQueue->buffer + startAddr,
                                        copyLength,
                                        cudaMemcpyHostToDevice,
                                        streams[cur_thread_id]
        ));
        currentAddrOffset += copyLength;
    } else {
//        cout << (currentAddrOffset + copyLength) / 1024 / 1024 << " MB !!!!" << endl;
        std::cerr << "Copying " << copyLength / 1024 / 1024 << " MB to thread " << cur_thread_id << " :totally "
                  << (currentAddrOffset + copyLength) / 1024 / 1024 << " MB" << endl;
        std::cerr << "Error: Copy exceeds buffer bounds!" << std::endl;
        inPacket = false;
        currentAddrOffset = 0;
    }
}