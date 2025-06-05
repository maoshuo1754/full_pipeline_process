//
// Created by csic724 on 24-7-11.
//

#ifndef CUDAPROJECT_UTILS_H
#define CUDAPROJECT_UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>
#include <cuComplex.h>
#include <chrono>
#include <ipp.h>
#include <sys/time.h>
#include <map>

std::vector<cufftComplex> PCcoef(double BandWidth, double PulseWidth, double Fs, int _NFFT, int hamming_window_enable);

unsigned int nextpow2(unsigned int x);

bool isCudaAvailable();

long long calculateDuration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);

void checkCufftErrors(cufftResult result);

void checkCudaErrors(cudaError_t result);

std::vector<double> hammingWindow(int N);

bool isEqual(double a, double b, double epsilon = 1e-6);

void getCurrentTime(uint32_t& second, uint32_t& microsecond);

unsigned int FourChars2Uint(char *startAddr);

void saveToBinaryFile(const cufftComplex* d_data, size_t size, const char* filename);

std::vector<cufftComplex> readFilterFromFile(const std::string& filename);

double getClutterMapAlpha(double q, double P_fa);

IppStatus save_ipp32fc_to_txt(const Ipp32fc* data, int length, const std::string& filename);

double asind(double x);

float getAzi(int wave_idx, double lambda);

// 枚举用于标识数据类型
enum class DataType : int {
    COMPLEX = 0,
    INT = 1,
    FLOAT = 2,
    BOOL = 3
};

// 模板函数：将设备端数组写入二进制文件
template <typename T>
void writeArrayToFile(T* d_data_, int rows_, int cols_, const std::string& filename) {
    T* h_data_ = new T[rows_ * cols_];
    cudaMemcpy(h_data_, d_data_, rows_ * cols_ * sizeof(T), cudaMemcpyDeviceToHost);

    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        delete[] h_data_;
        return;
    }

    outfile.write(reinterpret_cast<const char*>(&rows_), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&cols_), sizeof(int));

    DataType dtype;
    if (std::is_same<T, cufftComplex>::value || std::is_same<T, float2>::value) {
        dtype = DataType::COMPLEX;
    } else if (std::is_same<T, int>::value) {
        dtype = DataType::INT;
    } else if (std::is_same<T, float>::value) {
        dtype = DataType::FLOAT;
    } else if (std::is_same<T, bool>::value) {
        dtype = DataType::BOOL;
    } else {
        std::cerr << "Unsupported data type!" << std::endl;
        outfile.close();
        delete[] h_data_;
        return;
    }
    outfile.write(reinterpret_cast<const char*>(&dtype), sizeof(int));

    outfile.write(reinterpret_cast<const char*>(h_data_), rows_ * cols_ * sizeof(T));

    outfile.close();
    delete[] h_data_;
    std::cout << "================================ file write finished! (" << filename << ")" << std::endl;
}

#endif //CUDAPROJECT_UTILS_H
