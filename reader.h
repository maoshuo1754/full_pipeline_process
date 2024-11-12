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



class Reader{
public:
    Reader();
    ~Reader();
    void run();
    void kmpSearch(int block_index);

private:
    SharedQueue* sharedQueue;
    size_t patternSize;

    unsigned char *d_pattern;
    char *d_data;
    unsigned int prevSeqNum;

    static unsigned int FourChars2Uint(const char* startAddr);
};

#endif //READER_READER_H
