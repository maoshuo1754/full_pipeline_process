//
// Created by csic724 on 2024/11/11.
//
#include "Config.h"
#include "main.h"
#include "xdma_program.h"

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
    else {
        auto* xdma_programe1 = new xdma_program();
        std::thread dataSourceThread(&xdma_program::run, xdma_programe1);
        dataSourceThread.detach();
    }

    thread_pool.run();

    monitorConfigRunning = false;
    monitorWriterRunning = false;
    configMonitor.join();
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

    auto* sharedQueue = initSharedMemery(false);

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
