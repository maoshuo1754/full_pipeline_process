#ifndef XDMA_PROGRAME_H
#define XDMA_PROGRAME_H

#include <iostream>
#include "queue.h"
using namespace std;

class xdma_program
{
public:
    xdma_program();
    ~xdma_program();
    void readXDMAData(int blockIdx);
    void readXDMAUser(int index_addr,int readSize);
    void writeXDMAUserByte(int index_addr,char data);
    void run();

private:
    int dev_fd;
    int dev_fd_user;
    int dev_fd_events[4];

    void *map_user;

    char *pBufferData;
    char *pBufferAddr;

    SharedQueue* sharedQueue;
};

#endif // XDMA_PROGRAME_H
