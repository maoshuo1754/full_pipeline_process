#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

#include <atomic>
#include "SharedQueue.h"
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <sys/mman.h>
#include <fstream>

#define DEVICE_NAME_READ "/dev/xdma0_c2h_0"
#define DEVICE_NAME_USER "/dev/xdma0_user"
#define DEVICE_NAME_EVENT0 "/dev/xdma0_events_0"
#define DEVICE_NAME_EVENT1 "/dev/xdma0_events_1"
#define DEVICE_NAME_EVENT2 "/dev/xdma0_events_2"
#define DEVICE_NAME_EVENT3 "/dev/xdma0_events_3"
#define IRQ_REG_OFFSET 32768
using namespace std;

class DataSource {
protected:
    SharedQueue* sharedQueue;
    std::atomic<bool>& monitorWriterRunning;

public:
    explicit DataSource(std::atomic<bool>& running, SharedQueue* sharedQueue);
    virtual ~DataSource() = default;
    virtual void run() = 0;

protected:
    void acquireSlot();
    void releaseSlot();
};


class FileDataSource : public DataSource {
private:
    string dataPath;

public:
    FileDataSource(const string& path, std::atomic<bool>& running, SharedQueue* sharedQueue);
    void run() override;

private:
    void processBlock();
};

class XDMADataSource : public DataSource {
private:
    int dev_fd;
    int dev_fd_user;
    int dev_fd_events[4];
    void* map_user;
    char* pBufferData;
    char* pBufferAddr;
    char* pBufferUser;

public:
    XDMADataSource(std::atomic<bool>& running, SharedQueue* sharedQueue);
    ~XDMADataSource();
    void run() override;

private:
    void initializeDevices();
    void initializeBuffers();
    void setupEvents();
    void closeDevices();
    const char* getEventDeviceName(int index);
    void readXDMAUser(int index_addr, int readSize);
    void writeXDMAUserByte(int index_addr, char data);
    void readXDMAData(int blockIdx);
    void writeXDMAUserReset(int blockIdx);
    void readXDMAIRQReg(int index_addr, int readSize);
    void waitForIRQRegChange(int blockIdx);
};


#endif // DATA_SOURCE_H