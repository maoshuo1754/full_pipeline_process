#ifndef QUEUE_H
#define QUEUE_H

#include <sys/ipc.h>
#include <sys/shm.h>
#include <semaphore.h>
#include <stdexcept>

#define SHM_KEY 0x1234
#define QUEUE_SIZE 4                // 队列大小
#define BLOCK_SIZE (256*1024*1024)  // 256MB
#define INDEX_SIZE (4096)           // 1KB
#define SEQ_OFFSET 16               // 序列号相对于包头的偏移是16个Bytes

struct SharedQueue {
    sem_t mutex;
    sem_t slots_available; // 表示空槽位
    sem_t items_available; // 表示可用数据
    int read_index;
    int write_index;
    char index_buffer[QUEUE_SIZE * INDEX_SIZE];
    char buffer[QUEUE_SIZE * BLOCK_SIZE];

};

// 模式串：要匹配的字节数组
const unsigned char pattern[] = {
        0x07, 0x24, 0x95, 0xbc,
        0x00, 0x09, 0x00, 0x09,
};

#endif // QUEUE_H
