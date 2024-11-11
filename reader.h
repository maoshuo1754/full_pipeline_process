//
// Created by csic724 on 2024/11/6.
//

#ifndef READER_READER_H
#define READER_READER_H

#include "queue.h"
#include <vector>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <thread>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <iomanip>

using namespace std;

// 模式串：要匹配的字节数组
const unsigned char pattern[] = {
        0x07, 0x24, 0x95, 0xbc,
        0x00, 0x09, 0x00, 0x09,
};

class Reader{
public:
    Reader();
    ~Reader();
    void run();
    void buildPrefixTable();
    void kmpSearch(int block_index);
    unsigned int extractSeqNum(const char* data);

private:
    SharedQueue* sharedQueue;
    size_t patternSize;

    unsigned char *d_pattern;
    char *d_data;
    unsigned int prevSeqNum;

    void kmpSearchOnGPU();
    int threadsPerBlock;
    int blocksPerGrid;
    unsigned int FourChars2Uint(char* startAddr);
};

#endif //READER_READER_H
