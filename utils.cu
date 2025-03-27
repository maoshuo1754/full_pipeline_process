#include "utils.h"

using namespace std;
std::vector<cufftComplex> PCcoef(double BandWidth, double PulseWidth, double Fs, int _NFFT, int hamming_window_enable) {
//    std::cout << "BandWidth:" << BandWidth << " PulseWidth:" << PulseWidth << " Fs:" << Fs << std::endl;
    double Ts = 1 / Fs;

    int N = round(PulseWidth * Fs);
    double dT = (PulseWidth - Ts) / (N - 1); // t = linspace(-PulseWidth/2, PulseWidth/2-Ts, N);

    std::vector<cufftComplex> result(_NFFT, make_cuComplex(0, 0));

    // 生成线性调频信号
    for (int i = 0; i < N; ++i) {
        double t = -PulseWidth / 2 + i * dT;
        double phase = M_PI * BandWidth / PulseWidth * t * t;
        result[N-1-i] = cuConjf(make_cuComplex(cos(phase), sin(phase)));
    }

    if (hamming_window_enable) {
        auto window = hammingWindow(N);
        for (int i = 0; i < N; ++i) {
            result[i].x = result[i].x * window[i];
            result[i].y = result[i].y * window[i];
        }
    }

    return result;
}


unsigned int nextpow2(unsigned int x) {
    if (x == 0) return 1;
    return 1 << static_cast<unsigned int>(std::ceil(std::log2(x)));
}

bool isCudaAvailable() {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << "\n-> " << cudaGetErrorString(error_id) << std::endl;
        return false;
    }

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
        return false;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
        return true;
    }
}

long long calculateDuration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    // return duration time (us)
    using namespace std::chrono;
    auto duration = duration_cast<microseconds>(end - start);
    return duration.count();
}


void checkCufftErrors(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: ";
        switch (result) {
            case CUFFT_INVALID_PLAN:
                std::cerr << "CUFFT_INVALID_PLAN";
                break;
            case CUFFT_ALLOC_FAILED:
                std::cerr << "CUFFT_ALLOC_FAILED";
                break;
            case CUFFT_INVALID_TYPE:
                std::cerr << "CUFFT_INVALID_TYPE";
                break;
            case CUFFT_INVALID_VALUE:
                std::cerr << "CUFFT_INVALID_VALUE";
                break;
            case CUFFT_INTERNAL_ERROR:
                std::cerr << "CUFFT_INTERNAL_ERROR";
                break;
            case CUFFT_EXEC_FAILED:
                std::cerr << "CUFFT_EXEC_FAILED";
                break;
            case CUFFT_SETUP_FAILED:
                std::cerr << "CUFFT_SETUP_FAILED";
                break;
            case CUFFT_INVALID_SIZE:
                std::cerr << "CUFFT_INVALID_SIZE";
                break;
            default:
                std::cerr << "Unknown error";
        }
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "cudaError_t code: " << result << std::endl;
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

std::vector<double> hammingWindow(int N) {
    std::vector<double> window(N);
    for(int n = 0; n < N; n++) {
        window[n] = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / (N - 1));
    }
    return window;
}

bool isEqual(double a, double b, double epsilon) {
    return fabs(a - b) < epsilon;
}

void getCurrentTime(uint32_t& second, uint32_t& microsecond) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    second = static_cast<uint32_t>(tv.tv_sec % 86400);
    microsecond = static_cast<uint32_t>(tv.tv_usec);
}

unsigned int FourChars2Uint(char *startAddr) {
    return static_cast<uint8_t>(startAddr[0]) << 24
           | static_cast<uint8_t>(startAddr[1]) << 16
           | static_cast<uint8_t>(startAddr[2]) << 8
           | static_cast<uint8_t>(startAddr[3]);
}

void saveToBinaryFile(const cufftComplex* d_data, size_t size, const char* filename) {
    auto* h_data = new cufftComplex[size]; // 在主机上分配内存

    // 将数据从显存复制到主机内存
    cudaMemcpy(h_data, d_data, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // 打开文件并以二进制方式写入
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(h_data), size * sizeof(cufftComplex));
        file.close();
        std::cout << "Data saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for writing." << std::endl;
    }

    delete[] h_data; // 释放主机内存
}

void writeComplexToFile(cufftComplex* d_data_, int pulse_num_, int range_num_, const std::string& filename) {
    // 1. 分配主机内存
    cufftComplex* h_data_ = new cufftComplex[pulse_num_ * range_num_];

    // 2. 将数据从设备内存拷贝到主机内存
    cudaMemcpy(h_data_, d_data_, pulse_num_ * range_num_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // 3. 打开文件准备写入

    std::ofstream outfile(filename, std::ios::out); // 不使用a+模式打开文件
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        delete[] h_data_;
        return;
    }

    // 4. 将数据写入文件
    for (int i = 0; i < pulse_num_; ++i) {
        for (int j = 0; j < range_num_; ++j) {
            cufftComplex& complex_num = h_data_[i * range_num_ + j];
            float real_part = complex_num.x;
            float imag_part = complex_num.y;

            // 判断虚部的符号
            if (imag_part >= 0) {
                outfile << real_part << "+" << imag_part << "i ";
            } else {
                outfile << real_part << imag_part << "i ";
            }
        }
        outfile << std::endl; // 每行写入完毕后换行
    }

    // 5. 关闭文件
    outfile.close();

    // 6. 释放主机内存
    delete[] h_data_;

    cout << "================================  file write finished!" << endl;
}


void writeBoolToFile(bool* d_data_, int wave_num_, int range_num_, const std::string& filename) {
    // 1. 分配主机内存
    bool* h_data_ = new bool[wave_num_ * range_num_];

    // 2. 将数据从设备内存拷贝到主机内存
    cudaMemcpy(h_data_, d_data_, wave_num_ * range_num_ * sizeof(bool), cudaMemcpyDeviceToHost);

    // 3. 打开文件准备写入
    std::ofstream outfile(filename, std::ios::out); // 不使用追加模式
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        delete[] h_data_;
        return;
    }

    // 4. 将数据写入文件
    for (int i = 0; i < wave_num_; ++i) {
        for (int j = 0; j < range_num_; ++j) {
            bool value = h_data_[i * range_num_ + j];
            outfile << (value ? "1" : "0") << " ";
        }
        outfile << std::endl; // 每行写入完毕后换行
    }

    // 5. 关闭文件
    outfile.close();

    // 6. 释放主机内存
    delete[] h_data_;

    std::cout << "================================  file write finished!" << std::endl;
}



void writeFloatToFile(float* d_data_, int pulse_num_, int range_num_, const std::string& filename) {
    // 1. 分配主机内存
    float* h_data_ = new float[pulse_num_ * range_num_];

    // 2. 将数据从设备内存拷贝到主机内存
    cudaMemcpy(h_data_, d_data_, pulse_num_ * range_num_ * sizeof(float), cudaMemcpyDeviceToHost);

    // 3. 打开文件准备写入
    std::ofstream outfile(filename, std::ios::out); // 使用a+模式打开文件
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        delete[] h_data_;
        return;
    }

    // 4. 将数据写入文件
    for (int i = 0; i < pulse_num_; ++i) {
        for (int j = 0; j < range_num_; ++j) {
            outfile << h_data_[i * range_num_ + j] << " ";

        }
        outfile << std::endl; // 每行写入完毕后换行
    }

    // 5. 关闭文件
    outfile.close();

    // 6. 释放主机内存
    delete[] h_data_;

    cout << "================================  file write finished!" << endl;
}


std::vector<cufftComplex> readFilterFromFile(const string& filename) {
    std::ifstream file(filename);
    std::vector<cufftComplex> filter;
    if (file.is_open()) {
        float real;
        while (file >> real) {
            cufftComplex temp = {real, 0};
            filter.push_back(temp);
        }
    }
    else {
        std::cerr << "Failed to open file " << filename << " for reading." << std::endl;
        exit(EXIT_FAILURE);
    }
    file.close();
    return filter;
}


double getClutterMapAlpha(double q, double P_fa) {
    // 使用静态变量实现持久化存储
    static std::map<double, std::vector<std::pair<double, double>>> clutter_alpha_table;

    // 如果表为空，则初始化
    if (clutter_alpha_table.empty()) {
        // q = 3/4 的表
        clutter_alpha_table[3.0 / 4.0] = {
            {0.0001, 16.8945},
            {1e-05,  19.5117},
            {1e-06,  22.0703},
            {1e-07,  24.5898},
            {1e-08,  27.1094}
        };

        // q = 7/8 的表
        clutter_alpha_table[7.0 / 8.0] = {
            {0.0001, 14.2578},
            {1e-05,  15.8984},
            {1e-06,  17.4023},
            {1e-07,  18.8281},
            {1e-08,  20.1758}
        };

        // q = 15/16 的表
        clutter_alpha_table[15.0 / 16.0] = {
            {0.0001, 13.1445},
            {1e-05,  14.4141},
            {1e-06,  15.5273},
            {1e-07,  16.5405},
            {1e-08,  17.5195}
        };
    }

    // 检查 q 是否支持
    if (clutter_alpha_table.find(q) == clutter_alpha_table.end()) {
        char errorMsg[100];
        snprintf(errorMsg, sizeof(errorMsg),
                 "Unsupported q value: %.4f. Supported values are 3/4, 7/8, 15/16", q);
        throw std::invalid_argument(errorMsg);
    }

    // 获取当前 q 对应的表数据
    const auto& table_data = clutter_alpha_table[q];

    // 查找匹配的 P_fa
    double tolerance = 1e-10;
    for (const auto& entry : table_data) {
        if (std::abs(entry.first - P_fa) < tolerance) {
            return entry.second; // 返回对应的 alpha
        }
    }

    // 如果没有找到匹配的 P_fa，抛出异常
    char errorMsg[100];
    snprintf(errorMsg, sizeof(errorMsg),
             "No alpha value found for q=%.4f and P_fa=%.0e", q, P_fa);
    throw std::invalid_argument(errorMsg);
}
