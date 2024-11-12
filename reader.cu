#include <iostream>

#include "reader.h"

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


//    cudaMalloc((void**)&d_pattern, patternSize);
//    cudaMalloc((void**)&d_data, BLOCK_SIZE*sizeof(char));
//
//    cudaMemcpy(d_pattern, pattern, patternSize, cudaMemcpyHostToDevice);
    d_data = new char [BLOCK_SIZE];

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

        kmpSearch(sharedQueue->read_index);

        sharedQueue->read_index = (sharedQueue->read_index + 1) % QUEUE_SIZE;

        sem_post(&sharedQueue->mutex); // 解锁
        sem_post(&sharedQueue->slots_available); // 增加空槽位
    }
}

void printHex(const char *data, size_t dataSize) {
    for (size_t i = 0; i < dataSize; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (static_cast<unsigned int>(data[i]) & (0xFF))
                  << " ";
    }
    std::cout << std::dec;
    std::cout << std::endl;
}

unsigned int Reader::FourChars2Uint(const char* startAddr){
    return     static_cast<uint8_t>(startAddr[0]) << 24
               | static_cast<uint8_t>(startAddr[1]) << 16
               | static_cast<uint8_t>(startAddr[2]) << 8
               | static_cast<uint8_t>(startAddr[3]);
}


void Reader::kmpSearch(int block_index) {
//    cudaMemcpy(d_data, sharedQueue->buffer + block_index * BLOCK_SIZE, BLOCK_SIZE, cudaMemcpyHostToDevice);
//    memcpy(d_data, sharedQueue->buffer + block_index * BLOCK_SIZE, BLOCK_SIZE);
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
                if (startFlag){
                    cout << "Start flag at: " << seqNum << endl;
                }
                if (endFlag){
                    cout << "End flag at: " << seqNum << endl;
                }
            }
            prevSeqNum = seqNum;
        }
        else{
//            cout << i << endl;
//            std::cout << "end of head index" << std::endl;
            break;
        }
    }
}

int main() {
    Reader reader;
    reader.run();
}









