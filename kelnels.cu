//
// Created by csic724 on 2025/3/6.
//

#include "kelnels.cuh"

#include <thrust/detail/type_traits/is_call_possible.h>

#include "Config.h"
#include "SharedQueue.h"


__global__ void unpackKernel3D(unsigned char *threadsMemory, cufftComplex *pComplex,
                               const int *headPositions, int pulseNum, int rangeNum) {
    // 获取线程和网格索引
    int pulseIdx = blockIdx.z;  // 每个block.z处理一个头位置
    int rangeIdx = blockIdx.y * blockDim.x + threadIdx.x; // 每个线程处理一个距离单元
    int waveIdx = blockIdx.x;  // 每个block.x处理一个波束

    // 检查索引是否越界
    if (pulseIdx < pulseNum && rangeIdx < rangeNum && waveIdx < WAVE_NUM) {
        // 计算头位置的起始地址
        int headOffset = headPositions[pulseIdx];
        unsigned char *blockIQstartAddr = threadsMemory + headOffset + DATA_OFFSET;
        // 计算当前数据块的偏移和新索引
        int blockOffset = rangeIdx * WAVE_NUM * 4 + waveIdx * 4;
        int newIndex = waveIdx * PULSE_NUM * NFFT + pulseIdx * NFFT + rangeIdx;

        // 提取IQ数据并存储到结果数组
        pComplex[newIndex].x = *(int16_t *) (blockIQstartAddr + blockOffset + 2);
        pComplex[newIndex].y = *(int16_t *) (blockIQstartAddr + blockOffset);
    }
}

// Kernel for row-wise multiplication
__global__ void rowWiseMulKernel(cufftComplex *d_a, cufftComplex *d_b, int nrows, int ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrows * ncols) {
        cufftComplex temp_a = d_a[idx];
        cufftComplex temp_b = d_b[idx % ncols];
        d_a[idx].x = temp_a.x * temp_b.x - temp_a.y * temp_b.y;
        d_a[idx].y = temp_a.x * temp_b.y + temp_a.y * temp_b.x;
    }
}

__global__ void cmpKernel(cufftComplex *d_a, cufftComplex *d_b, int nrows, int ncols) {
    // d_a 为原始数据
    // d_b 为CFAR计算出来的噪底
    // 逐元素对比，大于噪底的，取根号，小于的取0
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrows * ncols) {
        if (d_a[idx].x < d_b[idx].x) {
            d_a[idx].x = 0;
            d_a[idx].y = 0;
        } else {
            d_a[idx].x = sqrtf(d_a[idx].x);
        }
    }
}

// CUDA kernel
__global__ void moveAndZeroKernel(cufftComplex* data, int m, int n, int start, int end) {
    // 获取当前处理的行
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        // 计算当前行起始位置
        int rowStart = row * n;

        // 第一步：移动数据
        for (int i = 0; i < (end - start + 1); i++) {
            data[rowStart + i] = data[rowStart + start + i];
        }

        // 第二步：置零剩余部分
        for (int i = end - start + 1; i < n; i++) {
            data[rowStart + i].x = 0.0f; // real part
            data[rowStart + i].y = 0.0f; // imag part
        }
    }
}

__global__ void maxKernel(cufftComplex *data, float *maxValues, int *speedChannels, bool* maskPtr, int nrows, int ncols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ind;

    if (col < ncols) {
        float maxVal;

        maxVal = channel_0_enable ? data[col].x : -100;
        int maxChannel = 0;
        for (int row = 1; row < nrows; ++row) {
            ind = row * ncols + col;
            if (data[ind].x > maxVal) {
                maxVal = data[ind].x;
                maxChannel = row;
            }
        }
        maxValues[col] = maxVal;
        // if (maskPtr[col]) {
        //     maxValues[col] = 1000;
        // }
        speedChannels[col] = maxChannel;
    }
}

