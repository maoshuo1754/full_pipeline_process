//
// Created by csic724 on 2024/12/17.
//
#include "Config.h"
#include <nlohmann/json.hpp> // 使用 JSON 库
#include <fstream>
#include <iostream>

using json = nlohmann::json;

// 变量定义

int num_threads;
int THREADS_MEM_SIZE;
int CAL_WAVE_NUM;
int INTEGRATION_TIMES;

std::vector<int> azi_table;
double c_speed;
double d;

std::string send_ip;
int send_port;
std::string multicast_ip;
int multicast_port;

double Fs;
double Fs_system;
int system_delay;

double Pfa;
int numGuardCells;
int numRefCells;

// 配置加载函数实现
void loadConfig(const std::string& filename) {
    std::ifstream configFile(filename);
    if (!configFile.is_open()) {
        throw std::runtime_error("Unable to open configuration file");
    }

    json config;
    configFile >> config;

    num_threads = config["num_threads"];
    THREADS_MEM_SIZE = config["THREADS_MEM_SIZE"];
    CAL_WAVE_NUM = config["CAL_WAVE_NUM"];
    INTEGRATION_TIMES = config["INTEGRATION_TIMES"];

    azi_table = config["azi_table"].get<std::vector<int>>();
    c_speed = config["c_speed"];
    d = config["d"];

    send_ip = config["send_ip"].get<std::string>();
    multicast_ip = config["multicast_ip"].get<std::string>();
    multicast_port = config["multicast_port"];
    send_port = config["send_port"];

    Fs = config["Fs"];
    Fs_system = config["Fs_system"];
    system_delay = config["system_delay"];

    Pfa = config["Pfa"];
    numRefCells = config["numRefCells"];
    numGuardCells = config["numGuardCells"];

    std::cout << "Configuration loaded successfully.\n";
}