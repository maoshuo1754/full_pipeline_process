#ifndef QUEUE_H
#define QUEUE_H

#include <semaphore.h>
#include "Config.h"

#define SHM_KEY 0x1234
#define QUEUE_SIZE 4                // 队列大小
#define BLOCK_SIZE (256*1024*1024)  // 256MB
#define INDEX_SIZE (4096)           // 4KB
#define SEQ_OFFSET 16               // 序列号相对于包头的偏移是16个Bytes
#define DATA_OFFSET (32 * 4)        // 数据相对于包头的偏移是 33 * 4 个 Bytes

struct SharedQueue {
    sem_t mutex;
    sem_t slots_available; // 表示空槽位
    sem_t items_available; // 表示可用数据
    int read_index;
    int write_index;
    unsigned char index_buffer[QUEUE_SIZE * INDEX_SIZE];
    unsigned char buffer[QUEUE_SIZE * BLOCK_SIZE];

};

// 模式串：要匹配的字节数组
const unsigned char pattern[] = {
        0xbc, 0x95, 0x24, 0x07,
        0x09, 0x00, 0x09, 0x00,
};

SharedQueue* initSharedMemery(bool);

#endif // QUEUE_H
