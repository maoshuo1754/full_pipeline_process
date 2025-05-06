 //
// Created by csic724 on 2025/2/19.
//

#include "DataSource.h"
#include "Config.h"
#include <chrono>
#include <thread>

DataSource::DataSource(std::atomic<bool>& running, SharedQueue* sharedQueue):
    monitorWriterRunning(running), sharedQueue(sharedQueue) {

}

void DataSource::acquireSlot() {
    sem_wait(&sharedQueue->slots_available);
    sem_wait(&sharedQueue->mutex);
}

void DataSource::releaseSlot() {
    sem_post(&sharedQueue->mutex);
    sem_post(&sharedQueue->items_available);
}

FileDataSource::FileDataSource(const string& path, std::atomic<bool>& running, SharedQueue* sharedQueue)
    : DataSource(running, sharedQueue), dataPath(path) {}

void FileDataSource::run() {
    ifstream file(dataPath, ios::binary | ios::ate);
    if(!file.is_open()) {
        cerr << "Can't open file " << dataPath << endl;
        return;
    }

    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    cout << "fileSize:" << fileSize / 1024 / 1024 / 1024 << " GB" << endl;

    while(!file.eof() && monitorWriterRunning.load()) {
        acquireSlot();

        memset(sharedQueue->index_buffer + sharedQueue->write_index * INDEX_SIZE, 0, INDEX_SIZE);
        file.read((char*)(sharedQueue->buffer + sharedQueue->write_index * BLOCK_SIZE), BLOCK_SIZE);

        processBlock();

        sharedQueue->write_index = (sharedQueue->write_index + 1) % QUEUE_SIZE;
        releaseSlot();
        this_thread::sleep_for(chrono::milliseconds(file_data_delay));
    }
}

void FileDataSource::processBlock() {
    int index = 0;
    for (unsigned int i = sharedQueue->write_index * BLOCK_SIZE;
         i < sharedQueue->write_index * BLOCK_SIZE + BLOCK_SIZE; i+=4) {
        if (*(uint64_t *)(sharedQueue->buffer + i) == *(uint64_t *)pattern) {
            auto ind_ptr = sharedQueue->index_buffer + sharedQueue->write_index * INDEX_SIZE + index * 4;
            memcpy(ind_ptr, &i, sizeof(i));
            index++;
        }
    }
}


XDMADataSource::XDMADataSource(std::atomic<bool>& running, SharedQueue* sharedQueue) : DataSource(running, sharedQueue) {
    initializeDevices();
    initializeBuffers();
    setupEvents();
    writeXDMAUserByte(0x00008080, 0x02); // trigger ready signal
}

XDMADataSource::~XDMADataSource() {
    closeDevices();
}

void XDMADataSource::run() {
    int eventVal;
    while (monitorWriterRunning.load()) {
        for (int blockIdx = 0; blockIdx < 4; blockIdx++) {
            auto res = read(dev_fd_events[blockIdx], &eventVal, 4);
            if (eventVal == 1) {
                acquireSlot();
                readXDMAData(blockIdx);
                writeXDMAUserReset(blockIdx);
                releaseSlot();
            }
        }
    }
}

void XDMADataSource::initializeDevices() {
    dev_fd = open(DEVICE_NAME_READ, O_RDWR | O_NONBLOCK);
    if (dev_fd < 0) {
        cerr << "Can't open device " << DEVICE_NAME_READ << endl;
    }
    dev_fd_user = open(DEVICE_NAME_USER, O_RDWR | O_SYNC);
    if (dev_fd_user < 0) {
        cerr << "Can't open device " << DEVICE_NAME_USER << endl;
    }
    for (int i = 0; i < 4; i++) {
        dev_fd_events[i] = open(getEventDeviceName(i), O_RDWR | O_SYNC);
        if (dev_fd_events[i] < 0) {
            cerr << "Can't open device " << getEventDeviceName(i) << endl;
        }
    }
}

void XDMADataSource::initializeBuffers() {
    pBufferData = reinterpret_cast<char*>(sharedQueue->buffer);
    pBufferAddr = reinterpret_cast<char*>(sharedQueue->index_buffer);
    memset(pBufferAddr, 0, INDEX_SIZE * 4);
}

void XDMADataSource::setupEvents() {
    for (int i = 0; i < 4; i++) {
        lseek(dev_fd_events[i], 0, SEEK_SET);
    }
}

void XDMADataSource::closeDevices() {
    close(dev_fd_user);
    close(dev_fd);
    for (int i = 0; i < 4; i++) {
        close(dev_fd_events[i]);
    }
}

const char* XDMADataSource::getEventDeviceName(int index) {
    static const char* eventNames[] = {
        DEVICE_NAME_EVENT0,
        DEVICE_NAME_EVENT1,
        DEVICE_NAME_EVENT2,
        DEVICE_NAME_EVENT3
    };
    return eventNames[index];
}

void XDMADataSource::readXDMAUser(int index_addr, int readSize) {
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

void XDMADataSource::writeXDMAUserByte(int index_addr, char data) {
    off_t target;
    off_t pgsz, target_aligned, offset;
    char* map_user_write;
    target = index_addr;
    pgsz = sysconf(_SC_PAGESIZE);
    offset = target & (pgsz - 1);
    target_aligned = target & (~(pgsz - 1));

    char ReadySignal[1] = {0};

    map_user_write = static_cast<char*>(mmap(nullptr, offset + 4, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd_user, target_aligned));

    auto* map_user_write2 = map_user_write + offset;
    memcpy(ReadySignal, map_user_write2, 1);

    ReadySignal[0] = data;
    *map_user_write2 = ReadySignal[0];

    if (munmap(map_user_write, offset + 4) == -1) {
        printf("Memory 0x%lx mapped failed: %s.\n",
               target, strerror(errno));
    }
}

void XDMADataSource::writeXDMAUserReset(int blockIdx) {
    int index_addr = blockIdx * INDEX_SIZE;

    off_t target;
    off_t pgsz, target_aligned, offset;
    char* map_user_write;
    target = index_addr;
    pgsz = sysconf(_SC_PAGESIZE);
    offset = target & (pgsz - 1);
    target_aligned = target & (~(pgsz - 1));

    map_user_write = static_cast<char*>(mmap(nullptr, offset + 4, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd_user, target_aligned));
    auto* map_user_write2 = map_user_write + offset;

    memset(map_user_write2, 0, INDEX_SIZE);

    if (munmap(map_user_write, offset + 4) == -1) {
        printf("Memory 0x%lx mapped failed: %s.\n",
               target, strerror(errno));
    }
}



void XDMADataSource::readXDMAData(int blockIdx) {
    int read_addr = blockIdx * BLOCK_SIZE;
    int index_addr = blockIdx * INDEX_SIZE;
    lseek(dev_fd, read_addr, SEEK_SET);
    auto res = read(dev_fd, &pBufferData[read_addr], BLOCK_SIZE);
    readXDMAUser(index_addr,INDEX_SIZE);
}
