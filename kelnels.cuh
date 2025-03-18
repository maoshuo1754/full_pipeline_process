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

__global__ void cfarKernel(const cufftComplex* data, cufftComplex* cfar_signal, int nrows, int ncols,
                           double alpha, int numGuardCells, int numRefCells, int leftBoundary, int rightBoundary);

__global__ void fftshift_columns_inplace_kernel(cufftComplex* d_data, int nrows, int ncols) ;

__global__ void cfar_col_kernel(const cufftComplex* data, cufftComplex* cfar_signal, int nrows, int ncols,
                                double alpha, int numGuardCells, int numRefCells);

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
