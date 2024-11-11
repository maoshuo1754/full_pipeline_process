#ifndef QUEUE_H
#define QUEUE_H

#include <sys/ipc.h>
#include <sys/shm.h>
#include <semaphore.h>
#include <stdexcept>

#define SHM_KEY 0x1234
#define QUEUE_SIZE 4                // 队列大小
#define BLOCK_SIZE (256*1024*1024)  // 256MB
#define INDEX_SIZE (1024)           // 1KB

struct SharedQueue {
    sem_t mutex;
    sem_t slots_available; // 表示空槽位
    sem_t items_available; // 表示可用数据
    int read_index;
    int write_index;
    char buffer[QUEUE_SIZE * BLOCK_SIZE];
    char index_buffer[QUEUE_SIZE * INDEX_SIZE];
};

#endif // QUEUE_H
