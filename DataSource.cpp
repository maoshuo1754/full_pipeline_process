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
    delete[] pBufferUser;
    closeDevices();
}

void XDMADataSource::run() {
    int eventVal;
    auto prev_time = std::chrono::high_resolution_clock::now();
    while (monitorWriterRunning.load()) {
        for (int blockIdx = 0; blockIdx < 4; blockIdx++) {
            auto start = std::chrono::high_resolution_clock::now();

            // auto res = read(dev_fd_events[blockIdx], &eventVal, 4);
            waitForIRQRegChange(blockIdx);

            // Time each function
            auto read_fd_events_end = std::chrono::high_resolution_clock::now();
            acquireSlot();
            auto acquire_end = std::chrono::high_resolution_clock::now();
            readXDMAData(blockIdx);
            auto read_end = std::chrono::high_resolution_clock::now();
            writeXDMAUserReset(blockIdx);
            auto write_end = std::chrono::high_resolution_clock::now();
            releaseSlot();
            auto curr_time = std::chrono::high_resolution_clock::now();
            auto durant = curr_time - start;
            auto durant_ms = std::chrono::duration_cast<std::chrono::milliseconds>(durant).count();
            if (durant_ms > 100 ) {
                // Calculate durations in microseconds for better precision
                auto read_fd_events_dur = std::chrono::duration_cast<std::chrono::milliseconds>(read_fd_events_end - start).count();
                auto acquire_dur = std::chrono::duration_cast<std::chrono::milliseconds>(acquire_end - read_fd_events_end).count();
                auto read_dur = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - acquire_end).count();
                auto write_dur = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - read_end).count();
                auto release_dur = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - write_end).count();

                std::cerr << "durant: " << durant_ms << "ms" << std::endl;
                std::cerr << "read_fd_events_dur: " << read_fd_events_dur << "ms" << std::endl;
                std::cerr << "acquireSlot: " << acquire_dur << "ms" << std::endl;
                std::cerr << "readXDMAData: " << read_dur << "ms" << std::endl;
                std::cerr << "writeXDMAUserReset: " << write_dur << "ms" << std::endl;
                std::cerr << "releaseSlot: " << release_dur << "ms" << std::endl;


                prev_time = curr_time;
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
    pBufferUser = new char[4];
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
    map_user = mmap(nullptr, offset+4, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd_user, target_aligned);
    memcpy(&pBufferAddr[index_addr], map_user, readSize);
    memset(map_user,0x00,readSize);
    if (munmap(map_user, offset+4) == -1) {
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

void XDMADataSource::writeXDMAUserReset(int blockIdx) {// 没有使用这个函数 而是在读取后立刻复位了
    off_t target;
    void * map_user_write;
    int index_addr = blockIdx * INDEX_SIZE;
    target = index_addr;

    off_t pgsz, target_aligned;
    pgsz = sysconf(_SC_PAGESIZE);
    //offset = target & (pgsz - 1);
    target_aligned = target & (~(pgsz - 1));

    map_user_write = mmap(nullptr, INDEX_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd_user, target_aligned);

    memset(map_user_write, 0, INDEX_SIZE);

    if (munmap(map_user_write, INDEX_SIZE) == -1) {
        printf("Memory 0x%lx mapped failed: %s.\n",
               target, strerror(errno));
    }
}

void XDMADataSource::readXDMAIRQReg(int index_addr, int readSize) {
    off_t pgsz, target_aligned, offset;
    off_t target = index_addr;
    pgsz = sysconf(_SC_PAGESIZE);
    offset = target & (pgsz - 1);
    target_aligned = target & (~(pgsz - 1));
    map_user = mmap(nullptr, offset+4, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd_user, target_aligned);
    memcpy(pBufferUser, map_user, readSize);
    //     memset(map_user,0x00,readSize);
    if (munmap(map_user, offset+4) == -1) {
        printf("Memory 0x%lx mapped failed: %s.\n",
               target, strerror(errno));
    }
}

// 阻塞，一直到寄存器的值发生变化
void XDMADataSource::waitForIRQRegChange(int blockIdx) {
    uint32_t expect_value = 1 << blockIdx;
    uint32_t IRQ_REG_Value;

    while (true) {
        // 读取寄存器函数
        readXDMAIRQReg(IRQ_REG_OFFSET, 4);
        IRQ_REG_Value = (uint32_t)pBufferUser[3];

        // 检查寄存器值是否发生变化
        if (IRQ_REG_Value == expect_value) {
            // cout << expect_value << endl;
            return;
        }
        // 短暂休眠避免过度占用 CPU
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
}



void XDMADataSource::readXDMAData(int blockIdx) {
    int read_addr = blockIdx * BLOCK_SIZE;
    int index_addr = blockIdx * INDEX_SIZE;
    lseek(dev_fd, read_addr, SEEK_SET);
    auto res = read(dev_fd, &pBufferData[read_addr], BLOCK_SIZE);
    readXDMAUser(index_addr,INDEX_SIZE);
}
