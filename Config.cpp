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
int INTEGRATION_TIMES;
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

double Pfa;
int numGuardCells;
int numRefCells;

int velocityCoalescenceMethod;
int dataSource;
int hamming_window_enable;

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
    CAL_WAVE_NUM = config["CAL_WAVE_NUM"];
    INTEGRATION_TIMES = config["INTEGRATION_TIMES"];
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

    Pfa = config["Pfa"];
    numRefCells = config["numRefCells"];
    numGuardCells = config["numGuardCells"];

    velocityCoalescenceMethod = config["velocityCoalescenceMethod"];
    dataSource = config["dataSource"];
    hamming_window_enable = config["hamming_window_enable"];

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