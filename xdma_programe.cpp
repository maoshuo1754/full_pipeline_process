#include "xdma_programe.h"

#include <unistd.h>
#include <fcntl.h>
#include <cstring>

#include <sys/mman.h>

#define DEVICE_NAME_READ "/dev/xdma0_c2h_0"
#define DEVICE_NAME_USER "/dev/xdma0_user"
#define DEVICE_NAME_EVENT0 "/dev/xdma0_events_0"
#define DEVICE_NAME_EVENT1 "/dev/xdma0_events_1"
#define DEVICE_NAME_EVENT2 "/dev/xdma0_events_2"
#define DEVICE_NAME_EVENT3 "/dev/xdma0_events_3"

xdma_programe::xdma_programe() {
    dev_fd = open(DEVICE_NAME_READ, O_RDWR | O_NONBLOCK);
    dev_fd_user = open(DEVICE_NAME_USER, O_RDWR | O_SYNC);
    dev_fd_events[0] = open(DEVICE_NAME_EVENT0, O_RDWR | O_SYNC);
    dev_fd_events[1] = open(DEVICE_NAME_EVENT1, O_RDWR | O_SYNC);
    dev_fd_events[2] = open(DEVICE_NAME_EVENT2, O_RDWR | O_SYNC);
    dev_fd_events[3] = open(DEVICE_NAME_EVENT3, O_RDWR | O_SYNC);

    sharedQueue = initSharedMemery(false);

    pBufferData = reinterpret_cast<char*>(sharedQueue->buffer);
    pBufferAddr = reinterpret_cast<char*>(sharedQueue->index_buffer);

    memset(pBufferAddr, 0, INDEX_SIZE * 4);

    for (int i = 0; i < 4; i++) {
        lseek(dev_fd_events[i], 0, SEEK_SET);
    }

    writeXDMAUserByte(0x00008080, 0x02); // trigger ready signal
}


void xdma_programe::run() {
    int eventVal;
    while (monitorWriterRunning.load()) {
        for (int blockIdx = 0; blockIdx < 4; blockIdx++) {
            auto res = read(dev_fd_events[blockIdx], &eventVal, 4);
            if (eventVal == 1) {
                sem_wait(&sharedQueue->slots_available); // 等待空槽位
                sem_wait(&sharedQueue->mutex); // 锁住共享资源

                readXDMAData(blockIdx);

                sem_post(&sharedQueue->mutex); // 解锁
                sem_post(&sharedQueue->items_available); // 增加可用数据量
            }
        }
    }
}


void xdma_programe::readXDMAUser(int index_addr, int readSize) {
    off_t pgsz, target_aligned, offset;
    off_t target = index_addr;
    pgsz = sysconf(_SC_PAGESIZE);
    offset = target & (pgsz - 1);
    target_aligned = target & (~(pgsz - 1));
    map_user = mmap(NULL, offset + 4, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd_user, target_aligned);

    memcpy(&pBufferAddr[index_addr], map_user, readSize);
    if (munmap(map_user, offset + 4) == -1) {
        printf("Memory 0x%lx mapped failed: %s.\n",
               target, strerror(errno));
    }
}

void xdma_programe::writeXDMAUserByte(int index_addr, char data) {
    //define some parameters
    off_t target;
    off_t pgsz, target_aligned, offset;
    char* map_user_write;
    target = index_addr;
    pgsz = sysconf(_SC_PAGESIZE);
    offset = target & (pgsz - 1);
    target_aligned = target & (~(pgsz - 1));

    // read/write value
    char ReadySignal[1] = {0};

    // create map
    map_user_write = static_cast<char*>(mmap(nullptr, offset + 4, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd_user, target_aligned));

    //add offset
    auto* map_user_write2 = map_user_write + offset;
    //read a byte
    memcpy(ReadySignal, map_user_write2, 1);
    //    ReadySignal[0] = ReadySignal[0] | 0b00000010;

    //write a byte
    ReadySignal[0] = data;
    *map_user_write2 = ReadySignal[0];

    //unmap
    if (munmap(map_user_write, offset + 4) == -1) {
        printf("Memory 0x%lx mapped failed: %s.\n",
               target, strerror(errno));
    }
}

void xdma_programe::readXDMAData(int blockIdx) {
    int read_addr = blockIdx * BLOCK_SIZE;
    int index_addr = blockIdx * INDEX_SIZE;
    lseek(dev_fd, read_addr, SEEK_SET);
    auto res = read(dev_fd, &pBufferData[read_addr], BLOCK_SIZE);
    readXDMAUser(index_addr,INDEX_SIZE);
}

