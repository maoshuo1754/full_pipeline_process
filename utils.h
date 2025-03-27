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

void writeComplexToFile(cufftComplex* d_data_, int pulse_num_, int range_num_, const std::string& filename);

void writeFloatToFile(float* d_data_, int pulse_num_, int range_num_, const std::string& filename);

void writeBoolToFile(bool* d_data_, int wave_num_, int range_num_, const std::string& filename);

std::vector<cufftComplex> readFilterFromFile(const std::string& filename);

double getClutterMapAlpha(double q, double P_fa);

#endif //CUDAPROJECT_UTILS_H
