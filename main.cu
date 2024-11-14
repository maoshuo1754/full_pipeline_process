//
// Created by csic724 on 2024/11/11.
//

#include "myThreadPool.h"

#define NUM_THREADS 8 // 线程数


int main(){
    int shmid = shmget(SHM_KEY, sizeof(SharedQueue), 0666 | IPC_CREAT);
    if (shmid == -1) throw std::runtime_error("Failed to create shared memory");

    auto sharedQueue = (SharedQueue*)shmat(shmid, nullptr, 0);

    // 初始化信号量和指针
    sem_init(&sharedQueue->mutex, 1, 1);
    sem_init(&sharedQueue->slots_available, 1, QUEUE_SIZE);
    sem_init(&sharedQueue->items_available, 1, 0);
    sharedQueue->read_index = 0;
    sharedQueue->write_index = 0;

    auto thread_pool = ThreadPool(NUM_THREADS, sharedQueue);
    thread_pool.run();
}
