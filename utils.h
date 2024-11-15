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

std::vector<std::vector<cufftComplex>> readMatTxt(const std::string &filePath);

std::vector<cufftComplex> generateLFM(double BandWidth, double PulseWidth, double Fs);

std::vector<cufftComplex> generatePCcoef(const std::vector<cufftComplex>& LFM);

std::vector<cufftComplex> repmat(const std::vector<cufftComplex>& vec, int rows, int cols);

unsigned int nextpow2(unsigned int x);

bool isCudaAvailable();

long long calculateDuration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);

#endif //CUDAPROJECT_UTILS_H
