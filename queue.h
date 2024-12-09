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
#define DATA_OFFSET (33 * 4)              // 数据相对于包头的偏移是 33 * 4 个 Bytes

#define THREADS_MEM_SIZE  (300 * 1024 * 1024)  // 存放未解包数据
#define WAVE_NUM 32    // 波束数
#define CAL_WAVE_NUM 32 // 需要计算的波束数
#define INTEGRATION_TIMES 50 // 积累次数

#define NUM_PULSE 256     // 一个波束中的脉冲数
#define RANGE_NUM 8192      // 一个脉冲中的距离单元数 做fft的，计算方法为 RANGE_NUM = 2 ** nextpow2(REAL_RANGE_NUM + numSamples - 1)
#define REAL_RANGE_NUM  7498 // 一个脉冲的真实距离单元数

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
