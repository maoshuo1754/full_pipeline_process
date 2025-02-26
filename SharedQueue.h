#ifndef QUEUE_H
#define QUEUE_H

#include <semaphore.h>

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
    unsigned char* index_buffer;
    unsigned char* buffer;

    SharedQueue() {
        index_buffer = new unsigned char[QUEUE_SIZE * INDEX_SIZE];
        buffer = new unsigned char[QUEUE_SIZE * BLOCK_SIZE];
        sem_init(&this->mutex, 1, 1);
        sem_init(&this->slots_available, 1, QUEUE_SIZE);
        sem_init(&this->items_available, 1, 0);
        this->read_index = 0;
        this->write_index = 0;
    }

    ~SharedQueue() {
        delete[] index_buffer;
        delete[] buffer;
    }
};

// 模式串：要匹配的字节数组
const unsigned char pattern[] = {
        0xbc, 0x95, 0x24, 0x07,
        0x09, 0x00, 0x09, 0x00,
};

#include <iostream>
#include <chrono>  // 用于计时
#include <iomanip> // 用于格式化输出

class DataRateTracker {
private:
    const size_t DATA_SIZE_MB = BLOCK_SIZE / 1024 / 1024; // 每次调用的大小
    size_t call_count = 0; // 调用次数计数
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_reset_time; // 上次重置时间
    double interval_seconds; // 输出间隔（秒）

public:
    DataRateTracker(double interval = 1.0) : interval_seconds(interval) {
        last_reset_time = std::chrono::steady_clock::now(); // 初始化起始时间
        start_time = last_reset_time;
    }

    ~DataRateTracker() {
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        if (elapsed_seconds.count() > 0) {
            double total_data_MB = call_count * DATA_SIZE_MB;
            double total_rate_gbps = (total_data_MB / 1024.0 / elapsed_seconds.count());
            std::cout << std::fixed << std::setprecision(2)
                << "Total data rate: " << total_rate_gbps << " GB/s over "
                << elapsed_seconds.count() << " seconds" << std::endl;;
        }
    }

    // 每次有数据到来时调用此函数
    void dataArrived() {
        call_count++;

        // 计算当前时间与上次重置时间的时间差
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = current_time - last_reset_time;

        // 如果超过指定间隔，输出数据率并重置
        if (elapsed.count() >= interval_seconds) {
            double data_mb = call_count * DATA_SIZE_MB; // 总数据量（MB）
            double rate_gbps = (data_mb / 1024.0) / elapsed.count(); // GB/s

            std::cout << std::fixed << std::setprecision(2) // 保留两位小数
                << "Data rate: " << rate_gbps << " GB/s" << std::endl;

            // 重置计数器和时间
            call_count = 0;
            last_reset_time = current_time;
        }
    }
};

#endif // QUEUE_H
