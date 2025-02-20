//
// Created by csic724 on 2025/2/13.
//
#include "SharedQueue.h"

#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdexcept>


SharedQueue* initSharedMemery(bool initPara)
{
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