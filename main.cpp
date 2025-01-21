//
// Created by csic724 on 2024/11/11.
//

#include "myThreadPool.h"
#include "Config.h"

void runCommandDelayed(const string& commond) {
    this_thread::sleep_for(chrono::milliseconds(600));
    system(commond.c_str());
}

int main(){
    string configFilePath = "/home/csic724/CLionProjects/PcieReader/config.json";
    loadConfig(configFilePath);
    std::thread configMonitor(monitorConfig, configFilePath, loadConfig);

//    cout << sizeof (unsigned int ) << endl;
    int shmid = shmget(SHM_KEY, sizeof(SharedQueue), 0666 | IPC_CREAT);
    if (shmid == -1) throw std::runtime_error("Failed to create shared memory");
    
    auto sharedQueue = (SharedQueue*)shmat(shmid, nullptr, 0);

    // 初始化信号量和指针
    sem_init(&sharedQueue->mutex, 1, 1);
    sem_init(&sharedQueue->slots_available, 1, QUEUE_SIZE);
    sem_init(&sharedQueue->items_available, 1, 0);
    sharedQueue->read_index = 0;
    sharedQueue->write_index = 0;

    cout << "Working Threads Num:" << num_threads << endl;
    ThreadPool thread_pool(num_threads, sharedQueue);

    thread t(runCommandDelayed, "/home/csic724/CLionProjects/writer/cmake-build-debug/writer");
    t.detach();

    thread_pool.run();

    monitorConfigRunning = false;
    configMonitor.join();
}
