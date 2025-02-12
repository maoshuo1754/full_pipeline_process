#include "utils.h"

using namespace std;
std::vector<cufftComplex> PCcoef(double BandWidth, double PulseWidth, double Fs, int _NFFT) {
//    std::cout << "BandWidth:" << BandWidth << " PulseWidth:" << PulseWidth << " Fs:" << Fs << std::endl;
    double Ts = 1 / Fs;

    int N = round(PulseWidth * Fs);
    double dT = (PulseWidth - Ts) / (N - 1); // t = linspace(-PulseWidth/2, PulseWidth/2-Ts, N);

    std::vector<cufftComplex> result(_NFFT, make_cuComplex(0, 0));
    auto window = hammingWindow(N);
    // 生成线性调频信号
    for (int i = 0; i < N; ++i) {
        double t = -PulseWidth / 2 + i * dT;
        double phase = M_PI * BandWidth / PulseWidth * t * t;
        result[N-1-i] = cuConjf(make_cuComplex(cos(phase), sin(phase)));
    }

    for (int i = 0; i < N; ++i) {
        result[i].x = result[i].x * window[i];
        result[i].y = result[i].y * window[i];
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
