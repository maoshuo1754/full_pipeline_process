#include "utils.h"

#include <complex>
#include <iomanip>

#include "Config.h"

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

IppStatus save_ipp32fc_to_txt(const Ipp32fc* data, int length, const std::string& filename) {
    // 打开文件以写入
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        return ippStsErr; // 返回错误状态，表示文件打开失败
    }

    // 设置输出格式：高精度，固定点表示
    out_file << std::fixed << std::setprecision(10);

    // 逐个写入复数的实部和虚部，用空格分隔
    for (int i = 0; i < length; ++i) {
        out_file << data[i].re;
        if (data[i].im >= 0) {
            out_file << "+";
        }
        out_file << data[i].im << "i\n";
    }

    // 检查写入是否成功
    if (out_file.fail()) {
        out_file.close();
        return ippStsErr; // 返回错误状态，表示写入失败
    }

    // 关闭文件
    out_file.close();
    return ippStsNoErr; // 返回成功状态
}

double asind(double x) {
    std::complex<double> z(x, 0.0);
    std::complex<double> result = std::asin(z) * 180.0 / M_PI;
    return result.real();
}

// wave_idx: 波束号，0 ~ WAVE_NUM-1
// 根据波束号和lambda获取当前波束方位
float getAzi(int wave_idx, double lambda) {
    int nAzmCode = (azi_table[31 - wave_idx] & 0xffff);

    if (nAzmCode > 32768)
        nAzmCode -= 65536;

    //rAzm = 153.4 + asin((nAzmCode * radar_params_->lambda) / (65536 * d)) / 3.1415926 * 180.0f;
    // rAzm = 183.4 + asin((nAzmCode * radar_params_->lambda) / (65536 * d)) / 3.1415926 * 180.0f;//-83
    float rAzm = 249.0633 + asin(nAzmCode * lambda / (65536 * d)) / 3.1415926 * 180.0f;//-13
    if (rAzm < 0)
        rAzm += 360.f;
    return rAzm;
}

// 函数：读取 CSV 文件并传输到显存
// 返回值：指向显存中数据的指针 (double*)
std::vector<double> readCSVToGPU(const std::string& filename, int& rows, int& cols) {
    // 1. 读取 CSV 文件到 CPU 内存
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::string message = "Could not open file " + filename;
        rows = 0;
        cols = 0;
        throw runtime_error(message);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    file.close();

    // 检查数据是否为空
    if (data.empty()) {
        std::string message = "Could not read file " + filename;
        rows = 0;
        cols = 0;
        throw runtime_error(message);
    }

    // 获取矩阵维度
    rows = data.size();
    cols = data[0].size();

    if (rows != WAVE_NUM || cols != NFFT) {
        throw std::invalid_argument("Invalid number of rows or columns");
    }

    // 2. 将二维向量展平为一维数组（显存中通常使用一维连续内存）
    std::vector<double> flat_data;
    flat_data.reserve(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat_data.push_back(data[i][j]);
        }
    }

    return flat_data;

    // // 3. 分配显存
    // double* d_data = nullptr;
    // size_t size = rows * cols * sizeof(double);
    // cudaError_t err = cudaMalloc((void**)&d_data, size);
    // checkCudaErrors(err);
    //
    // // 4. 将数据从 CPU 传输到显存
    // err = cudaMemcpy(d_data, flat_data.data(), size, cudaMemcpyHostToDevice);
    // checkCudaErrors(err);
    //
    // return d_data;
}
