#include "utils.h"

std::vector<std::vector<cufftComplex>> readMatTxt(const std::string &filePath) {
    std::ifstream infile(filePath);

    if (!infile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
    }

    int m, n;
    infile >> m >> n;

    std::vector<std::vector<cufftComplex>> data(m, std::vector<cufftComplex>(n));

    float real, imag;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; ++j) {
            infile >> real >> imag;
            data[i][j].x = real;
            data[i][j].y = imag;
        }
    }
    infile.close();
    return data;
}

std::vector<cufftComplex> generateLFM(double BandWidth, double PulseWidth, double Fs) {
    double Ts = 1 / Fs;

    int N = round(PulseWidth * Fs);
    double dT = (PulseWidth - Ts) / (N - 1); // t = linspace(-PulseWidth/2, PulseWidth/2-Ts, N);
    std::vector<cufftComplex> LFM(N);

    for (int i = 0; i < N; ++i) {
        double t = -PulseWidth / 2 + i * dT;
        double phase = M_PI * BandWidth / PulseWidth * t * t;
        LFM[i] = make_cuComplex(cos(phase), sin(phase));
    }

    return LFM;
}

std::vector<cufftComplex> generatePCcoef(const std::vector<cufftComplex>& LFM) {
    int N = LFM.size();
    std::vector<cufftComplex> PCcoef(N);

    for (int i = 0; i < N; ++i) {
        cufftComplex val = LFM[N - 1 - i];
        PCcoef[i] = cuConjf(val);
    }

    return PCcoef;
}

std::vector<cufftComplex> repmat(const std::vector<cufftComplex>& vec, int rows, int cols) {
    std::vector<cufftComplex> result(rows * cols * vec.size());

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::copy(vec.begin(), vec.end(), result.begin() + (r * cols + c) * vec.size());
        }
    }
    return result;
}

std::vector<cufftComplex> PCcoef(double BandWidth, double PulseWidth, double Fs, int _NFFT) {
//    std::cout << "BandWidth:" << BandWidth << " PulseWidth:" << PulseWidth << " Fs:" << Fs << std::endl;
    double Ts = 1 / Fs;

    int N = round(PulseWidth * Fs);
    double dT = (PulseWidth - Ts) / (N - 1); // t = linspace(-PulseWidth/2, PulseWidth/2-Ts, N);

    std::vector<cufftComplex> result(_NFFT);

    // 生成线性调频信号
    for (int i = 0; i < N; ++i) {
        double t = -PulseWidth / 2 + i * dT;
        double phase = M_PI * BandWidth / PulseWidth * t * t;
        result[N-1-i] = cuConjf(make_cuComplex(cos(phase), sin(phase)));
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