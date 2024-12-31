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

std::vector<cufftComplex> PCcoef(double BandWidth, double PulseWidth, double Fs, int _NFFT);

unsigned int nextpow2(unsigned int x);

bool isCudaAvailable();

long long calculateDuration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);

void checkCufftErrors(cufftResult result);

void checkCudaErrors(cudaError_t result);

unsigned int FourChars2Uint(const char *startAddr);

std::vector<double> hammingWindow(int N);

bool isEqual(double a, double b, double epsilon = 1e-9);

#endif //CUDAPROJECT_UTILS_H
