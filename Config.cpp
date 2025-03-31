//
// Created by csic724 on 2024/12/17.
//
#include "Config.h"
#include <filesystem>
#include "nlohmann/json.hpp" // 使用 JSON 库
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

using json = nlohmann::json;

// 变量定义
std::string dataPath;

int num_threads;
int THREADS_MEM_SIZE;
int CAL_WAVE_NUM;
float normFactor;

std::vector<int> azi_table;
double c_speed;
double d;

std::string local_video_ip;
int local_video_port;
std::string remote_video_ip;
int remote_video_port;
std::string remote_plot_ip;
int remote_plot_port;

double Fs;
double Fs_system;
int system_delay;
int initCarryFreq;

double Pfa_cfar;
double Pfa_clutter_map;
int numGuardCells;
int numRefCells;
int clutter_map_enable;
int cfar_enable;
double forgetting_factor;
double clutter_map_range;

int velocityCoalescenceMethod;
int dataSource_type;
int hamming_window_enable;
int MTI_enable;
int MTI_pulse_num;

int debug_mode;
int start_frame;
int end_frame;
int start_wave;
int end_wave;

int v1;
int v2;

int file_data_delay;

std::vector<clutterRegion> clutterRegions;
int range_correct;

// 配置加载函数实现
void loadConfig(const std::string& filename) {
    std::ifstream configFile(filename);
    if (!configFile.is_open()) {
        throw std::runtime_error("Unable to open configuration file");
    }

    json config;
    configFile >> config;

    dataPath = config["dataPath"];
    num_threads = config["num_threads"];
    THREADS_MEM_SIZE = config["THREADS_MEM_SIZE"];

    normFactor = config["normFactor"];

    azi_table = config["azi_table"].get<std::vector<int>>();
    c_speed = config["c_speed"];
    d = config["d"];

    local_video_ip = config["local_video_ip"].get<std::string>();
    local_video_port = config["local_video_port"];

    remote_video_ip = config["remote_video_ip"].get<std::string>();
    remote_video_port = config["remote_video_port"];

    remote_plot_ip = config["remote_plot_ip"].get<std::string>();
    remote_plot_port = config["remote_plot_port"];

    Fs = config["Fs"];
    Fs_system = config["Fs_system"];
    system_delay = config["system_delay"];
    initCarryFreq = config["initCarryFreq"];

    Pfa_cfar = config["Pfa_cfar"];
    Pfa_clutter_map = config["Pfa_clutter_map"];
    numRefCells = config["numRefCells"];
    numGuardCells = config["numGuardCells"];
    clutter_map_enable = config["clutter_map_enable"];
    cfar_enable = config["cfar_enable"];
    std::string forgetting_factor_str = config["forgetting_factor"];
    forgetting_factor = parseFraction(forgetting_factor_str);
    clutter_map_range = config["clutter_map_range"];

    velocityCoalescenceMethod = config["velocityCoalescenceMethod"];
    dataSource_type = config["dataSource_type"];
    hamming_window_enable = config["hamming_window_enable"];
    MTI_enable = config["MTI_enable"];
    MTI_pulse_num = config["MTI_pulse_num"];
    if (MTI_pulse_num != 2 && MTI_pulse_num != 3) {
        throw std::runtime_error("MTI 目前只支持2脉冲或三脉冲");
    }

    debug_mode = config["debug_mode"];
    start_frame = config["start_frame"];
    end_frame = config["end_frame"];
    start_wave = config["start_wave"];
    end_wave = config["end_wave"];

    CAL_WAVE_NUM = end_wave - start_wave;

    file_data_delay = config["file_data_delay"];

    for (const auto& item : config["clutter_processing_region"]) {
        clutterRegion e{};
        e.waveStartIdx = item[0];
        e.waveEndIdx = item[1];
        e.startRange = item[2];
        e.endRange = item[3];
        clutterRegions.push_back(e);
    }

    double v1_mps = config["v1"].get<double>(); // 假设单位是 m/s，例如 1.0
    double v2_mps = config["v2"].get<double>(); // 例如 5.0
    v1 = static_cast<int>(std::round(v1_mps * 100)); // 转换为 cm/s
    v2 = static_cast<int>(std::round(v2_mps * 100));

    if (v1 >= v2) {
        throw std::runtime_error("v1 必须小于 v2");
    }

    range_correct = config["range_correct"];

    std::cout << "Configuration loaded successfully.\n";
}



void monitorConfig(const std::string& filename, void (*loadConfig)(const std::string&)) {
    namespace fs = std::filesystem;
    auto lastWriteTime = fs::last_write_time(filename);

    while(monitorConfigRunning) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto currentWriteTime = fs::last_write_time(filename);
        if(lastWriteTime < currentWriteTime) {
            std::cout << "Config file changes! reloading..." << std::endl;
            lastWriteTime = currentWriteTime;

            try {
                loadConfig(filename);
            } catch(const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }
        }
    }
}

// 负责将遗忘因子分数字符串转double (15/16)
double parseFraction(const std::string& fraction) {
    std::stringstream ss(fraction);
    int numerator, denominator;
    char slash;
    ss >> numerator >> slash >> denominator;
    return static_cast<double>(numerator) / denominator;
}
