//
// Created by csic724 on 2025/3/6.
//

#ifndef KELNELS_CUH
#define KELNELS_CUH


#include <cufft.h>

__global__ void unpackKernel3D(unsigned char *threadsMemory, cufftComplex *pComplex,
                               const int *headPositions, int pulseNum, int rangeNum);

__global__ void rowWiseMulKernel(cufftComplex *d_a, cufftComplex *d_b, int nrows, int ncols);

__global__ void cmpKernel(cufftComplex *d_a, cufftComplex *d_b, int nrows, int ncols, int offset);

__global__ void moveAndZeroKernel(cufftComplex* data, int m, int n, int start, int end);

__global__ void maxKernel(cufftComplex *data, float *maxValues, int *speedChannels, bool* maskPtr, int nrows, int ncols);

struct ScaleFunctor {
    float scale;

    ScaleFunctor(float s) : scale(s) {}

    __device__ cufftComplex operator()(cufftComplex& c) const {
        return make_cuComplex(c.x * scale, c.y * scale);
    }
};

struct SquareFunctor {
    SquareFunctor() = default;

    __device__ cufftComplex operator()(cufftComplex& c) const {
        return make_cuComplex(c.x * c.x + c.y * c.y, 0);
    }
};


#endif //KELNELS_CUH
