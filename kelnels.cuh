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

__global__ void maxKernel2D(cufftComplex *data, float *maxValues, int *speedChannels,
                           bool* maskPtr, int nrows, int ncols, int nwaves);


__global__ void cfarKernel(const cufftComplex* data, cufftComplex* cfar_signal, int nrows, int ncols,
                           double alpha, int numGuardCells, int numRefCells, int leftBoundary, int rightBoundary);

__global__ void fftshift_columns_inplace_kernel(cufftComplex* d_data, int nrows, int ncols) ;

__global__ void cfar_col_kernel(const cufftComplex* data, cufftComplex* cfar_signal, int nrows, int ncols,
                                double alpha, int numGuardCells, int numRefCells);

__global__ void update_queues_kernel(
    const cufftComplex* frame, cufftComplex* queues, cufftComplex* queues_speed, int* indices,
    int wave_num, int pulse_num, int range_num, int queue_size, int speed_channels
);

__global__ void compute_clutter_kernel(
    const cufftComplex* queues, const cufftComplex* queues_speed, const int* indices, bool* clutter,
    int wave_num, int range_num, int queue_size, int speed_channels
);


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
