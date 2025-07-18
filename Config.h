//
// Created by csic724 on 2024/12/17.
//

#ifndef READER_CONFIG_H
#define READER_CONFIG_H

#include <vector>
#include <string>
#include <atomic>

#define WAVE_NUM 32             // 波束数
#define PULSE_NUM 2000          // 一个波束中的脉冲数
#define NFFT 4096               // 一个脉冲中fft的点数，计算方法为 NFFT = 2 ** nextpow2(RANGE_NUM + numSamples - 1)
#define RANGE_NUM  3904         // 一个脉冲的距离单元数
#define CFAR_LENGTH 16          // 分段fft长度
#define CUDA_BLOCK_SIZE 256     // cuda运算分块长度
#define CLUTTER_QUEUE_SIZE 10
#define SPEED_CHANNELS 20

// 配置参数声明
extern std::string dataPath;

extern int num_threads;         // 工作线程数
extern int THREADS_MEM_SIZE;    // 每个线程独立显存大小
extern int CAL_WAVE_NUM;        // 计算的波束数
extern float normFactor;        // 归一化参数

extern std::vector<int> azi_table; // 方位表

extern double c_speed;          // 光速
extern double d;                // 距离单元距离

extern std::string local_video_ip;
extern int local_video_port;

extern std::string remote_video_ip;
extern int remote_video_port;

extern std::string remote_plot_ip;
extern int remote_plot_port;

extern double Fs;               // 采样率
extern double Fs_system;        // 雷达系统内部时钟频率
extern int system_delay;        // 系统处理延时, 低通滤波等系统延时
extern int initCarryFreq;       // 初始载频

extern int cfar_enable;
extern double Pfa_cfar;
extern int numGuardCells;
extern int numRefCells;
extern float cfar_db_offset;
extern int rasterize_manage_enable;

extern int clutter_map_enable;
extern double Pfa_clutter_map;
extern double forgetting_factor;
extern double clutter_map_range;
extern float clutter_db_offset;

extern int azi_densify_enable;
extern int azi_densify_plot_conv_method;
extern int azi_densify_crow_num;
extern int azi_densify_EstSample_num;
extern double azi_densify_range_start;
extern double azi_densify_range_end;
extern float azi_densify_invalid_num;
extern int azi_densify_wave_start;
extern int azi_densify_wave_end;
extern double azi_densify_d;

extern int velocityCoalescenceMethod;  // 0代表质心法，1代表选大
extern int dataSource_type;                 // 0表示文件，1表示Pcie
extern int hamming_window_enable;      // hamming窗
extern int MTI_enable;
extern int MTI_pulse_num;

extern int debug_mode;
extern int start_frame;
extern int end_frame;
extern int start_wave;
extern int end_wave;

extern int v1;
extern int v2;

extern int file_data_delay;
// 数据缓冲区声明

struct clutterRegion {
    int waveStartIdx;
    int waveEndIdx;
    float startRange;
    float endRange;
};

extern std::vector<clutterRegion> clutterRegions;
extern int range_correct;

void loadConfig(const std::string& filename); // 声明配置加载函数
double parseFraction(const std::string& fraction);

inline std::atomic<bool> monitorConfigRunning(true);
void monitorConfig(const std::string& filename, void (*loadConfig)(const std::string&)); // 参数实时更新

#endif //READER_CONFIG_H
