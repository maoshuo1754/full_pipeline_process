//
// Created by csic724 on 2024/11/11.
//

#include "ThreadPool.h"
#include "Config.h"
#include "main.h"


int main(){
    string configFilePath = "../config.json";
    loadConfig(configFilePath);
    std::thread configMonitor(monitorConfig, configFilePath, loadConfig);

    auto* sharedQueue = initSharedMemery(true);
    cout << "Working Threads Num:" << num_threads << endl;
    ThreadPool thread_pool(num_threads, sharedQueue);

    if(dataSource == 0) {
        std::thread dataSourceThread(readDataFromFile, dataPath);
        dataSourceThread.detach();
    }

    thread_pool.run();

    monitorConfigRunning = false;
    monitorWriterRunning = false;
    configMonitor.join();
}


SharedQueue* initSharedMemery(bool initPara) {
    int shmid = shmget(SHM_KEY, sizeof(SharedQueue), 0666 | IPC_CREAT);
    if (shmid == -1) throw std::runtime_error("Failed to create shared memory");

    auto* sharedQueue = static_cast<SharedQueue*>(shmat(shmid, nullptr, 0));

    if (initPara) {
        // 初始化信号量和指针
        sem_init(&sharedQueue->mutex, 1, 1);
        sem_init(&sharedQueue->slots_available, 1, QUEUE_SIZE);
        sem_init(&sharedQueue->items_available, 1, 0);
        sharedQueue->read_index = 0;
        sharedQueue->write_index = 0;
    }

    return sharedQueue;
}

void readDataFromFile(const string& dataPath) {

    ifstream file(dataPath, ios::binary | ios::ate);
    if(!file.is_open()) {
        cerr << "Can't open file " << dataPath << endl;
        return ;
    }

    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    cout << "fileSize:" << fileSize / 1024 / 1024 / 1024 << " GB" << endl;

    auto sharedQueue = initSharedMemery(false);

    while(!file.eof() && monitorWriterRunning.load())
    {
        sem_wait(&sharedQueue->slots_available); // 等待空槽位
        sem_wait(&sharedQueue->mutex); // 锁住共享资源

        memset(sharedQueue->index_buffer + sharedQueue->write_index * INDEX_SIZE, 0, INDEX_SIZE);
        file.read((char*)(sharedQueue->buffer + sharedQueue->write_index * BLOCK_SIZE), BLOCK_SIZE);
        int index = 0;

        for (unsigned int i = sharedQueue->write_index * BLOCK_SIZE; i < sharedQueue->write_index * BLOCK_SIZE + BLOCK_SIZE; i+=4) {

            if (*(uint64_t *) (sharedQueue->buffer + i) == *(uint64_t *)pattern) {
                auto ind_ptr = sharedQueue->index_buffer + sharedQueue->write_index * INDEX_SIZE + index * 4;
                memcpy(ind_ptr, &i, sizeof(i));
                index++;
            }
        }

        sharedQueue->write_index = (sharedQueue->write_index + 1) % QUEUE_SIZE;
        sem_post(&sharedQueue->mutex); // 解锁
        sem_post(&sharedQueue->items_available); // 增加可用数据量
    }

}
