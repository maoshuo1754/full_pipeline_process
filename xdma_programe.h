#ifndef XDMA_PROGRAME_H
#define XDMA_PROGRAME_H

#include <iostream>
#include "queue.h"
using namespace std;

class xdma_programe
{

public:
    xdma_programe();
    void readXDMAData(int blockIdx);
    void readXDMAUser(int index_addr,int readSize);
    void writeXDMAUserByte(int index_addr,char data);
    void run();

private:
    int dev_fd;
    int dev_fd_user;
    int dev_fd_events[4];

    void *map_user;//user map=

    int operateSize;
    char *pBufferData;
    char *pBufferAddr;

    SharedQueue* sharedQueue;
};

#endif // XDMA_PROGRAME_H
