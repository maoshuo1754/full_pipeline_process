#include <iostream>

#include "reader.h"

__global__ void kmpSearchKernel(const char* data, const unsigned char* pattern, int dataSize, int patternSize, bool* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保不越界
    if (idx + patternSize <= dataSize) {
        result[idx] = false;
        bool match = true;
        for (int j = 0; j < patternSize; ++j) {
            if ((unsigned char)data[idx + j] != pattern[j]) {
                match = false;
                break;
            }
        }

//        if (*(uint64_t*)(data + idx) == *(uint64_t*)pattern) {
        if (match) {
            // 找到匹配，记录匹配的位置

            result[idx] = true;  // 或存储其他信息
        }
    }
}

Reader::Reader() :patternSize(sizeof(pattern)){

    int shmid = shmget(SHM_KEY, sizeof(SharedQueue), 0666 | IPC_CREAT);
    if (shmid == -1) throw std::runtime_error("Failed to create shared memory");

    sharedQueue = (SharedQueue*)shmat(shmid, nullptr, 0);

    // 初始化信号量和指针
    sem_init(&sharedQueue->mutex, 1, 1);
    sem_init(&sharedQueue->slots_available, 1, QUEUE_SIZE);
    sem_init(&sharedQueue->items_available, 1, 0);
    sharedQueue->read_index = 0;
    sharedQueue->write_index = 0;


    cudaMalloc((void**)&d_pattern, patternSize);
    cudaMalloc((void**)&d_data, BLOCK_SIZE*sizeof(char));

    cudaMemcpy(d_pattern, pattern, patternSize, cudaMemcpyHostToDevice);
    threadsPerBlock = 256;
    blocksPerGrid = (BLOCK_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    prevSeqNum = 0;
}

Reader::~Reader() {
    cudaFree(d_data);
    cudaFree(d_pattern);
}

void Reader::run() {
    while (true) {
        sem_wait(&sharedQueue->items_available); // 等待可用数据
        sem_wait(&sharedQueue->mutex); // 锁住共享资源

//        char data = sharedQueue->buffer[sharedQueue->read_index * BLOCK_SIZE];
//        std::cout << sharedQueue->read_index << std::endl;

        kmpSearch(sharedQueue->read_index);
//        cout << "Data Recieved!" <<endl;
//        cudaMemcpy(d_data, &sharedQueue->buffer[sharedQueue->read_index * BLOCK_SIZE], dataSize*sizeof(char), cudaMemcpyHostToDevice);
//        kmpSearchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_pattern, dataSize, patternSize, d_result);
//        cudaDeviceSynchronize();
////        memset(result, 0, sizeof(result));
//        cudaMemcpy(result, d_result, dataSize * sizeof(bool), cudaMemcpyDeviceToHost);
//
//        char* data = &(sharedQueue->buffer[sharedQueue->read_index * BLOCK_SIZE]);
//        for (int i = 0; i < BLOCK_SIZE; i++){
//            if (result[i]){
//                cout << extractSeqNum(data + i + 16) << endl;
//                unsigned int cur_count =  static_cast<uint8_t>(data[i+16]) << 24
//                                          | static_cast<uint8_t>(data[i+17]) << 16
//                                          | static_cast<uint8_t>(data[i+18]) << 8
//                                          | static_cast<uint8_t>(data[i+19]);
////                cout << cur_count << endl;
//
//            }
//        }
        sharedQueue->read_index = (sharedQueue->read_index + 1) % QUEUE_SIZE;

        sem_post(&sharedQueue->mutex); // 解锁
        sem_post(&sharedQueue->slots_available); // 增加空槽位
    }
}

unsigned int Reader::FourChars2Uint(char* startAddr){
    return   static_cast<uint8_t>(startAddr[0]) << 24
           | static_cast<uint8_t>(startAddr[1]) << 16
           | static_cast<uint8_t>(startAddr[2]) << 8
           | static_cast<uint8_t>(startAddr[3]);
}

void printHex(const char *data, size_t dataSize) {
    for (size_t i = 0; i < dataSize; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (static_cast<unsigned int>(data[i]) & (0xFF))
                  << " ";
    }
    std::cout << std::dec;
    std::cout << std::endl;
}


void Reader::kmpSearch(int block_index) {
//    char* data = &(sharedQueue->buffer[block_index * BLOCK_SIZE]);
//    char* index = &(sharedQueue->index_buffer[block_index * INDEX_SIZE]);
//
//    auto count = 0;
//
//    for (size_t i = 0; i < dataSize; ++i) { // 主串索引
//        // find package head
//        if (*(uint64_t*)(data + i) == *(uint64_t*)pattern) {
////            if (i + block_index *  BLOCK_SIZE != FourChars2Uint(index + count)){
//            auto index_offset = FourChars2Uint(index + count);
//            if (i + block_index *  BLOCK_SIZE != index_offset){
//                cout << "BOLCK index:" << block_index ;//<< endl;
//                cout << " real:" << i + block_index *  BLOCK_SIZE ;//<< endl;
//                cout << " read:" << index_offset << endl;
////                cout << " offset:" << i + block_index * BLOCK_SIZE - FourChars2Uint(index + count)  << endl;
//                printHex(sharedQueue->buffer+index_offset, 4);
//            }
//
//            count += 4;
////            unsigned int cur_count =  static_cast<uint8_t>(data[i+16]) << 24
////                                    | static_cast<uint8_t>(data[i+17]) << 16
////                                    | static_cast<uint8_t>(data[i+18]) << 8
////                                    | static_cast<uint8_t>(data[i+19]);
//            std::cout << " seq Num:" << FourChars2Uint(data + i + 16) << std::endl;
////            cout << endl;
//        }
//    }

    cudaMemcpy(d_data, sharedQueue->buffer + block_index * BLOCK_SIZE, BLOCK_SIZE, cudaMemcpyHostToDevice);
    unsigned int indexValue, dataValue, seqNum, startFlag = 0, endFlag = 0;
    for (int i = 0; i < 128; i++){
        indexValue = FourChars2Uint(sharedQueue->index_buffer + block_index * INDEX_SIZE + i * 4);

        if (*(uint64_t*)(sharedQueue->buffer + indexValue) == *(uint64_t*)pattern){
            seqNum =     FourChars2Uint(sharedQueue->buffer + indexValue + 16);
            if (seqNum != prevSeqNum + 1 && prevSeqNum != 0){
                std::cout << "Error! seq num not continuous~" << std::endl;
            }
            else{

                startFlag = static_cast<uint8_t>(sharedQueue->buffer[indexValue + 23]) & 0x02;
                endFlag = static_cast<uint8_t>(sharedQueue->buffer[indexValue + 23]) & 0x01;
//                if (startFlag){
//                    cout << "Start flag at seqNum:" << seqNum << endl;
//                }
//                if (endFlag){
//                    cout << "End flag at seqNum:" << seqNum << endl;
//                }
            }
            prevSeqNum = seqNum;
        }
        else{
//            std::cout << "end of head index" << std::endl;
            break;
        }
    }
}

void Reader::kmpSearchOnGPU(){

}

unsigned int Reader::extractSeqNum(const char *data) {
    return static_cast<uint8_t>(data[0]) << 24
           | static_cast<uint8_t>(data[1]) << 16
           | static_cast<uint8_t>(data[2]) << 8
           | static_cast<uint8_t>(data[3]);
}


int main() {
    Reader reader;
    reader.run();
}









