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


__global__ void cmpKernel(cufftComplex *d_data, cufftComplex *thresholds, int nrows, int ncols, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrows * ncols) {
        if (idx % ncols < ncols - offset)
        {
            if (d_data[idx].x < thresholds[idx + offset].x) {
                d_data[idx].x = 0;
                d_data[idx].y = 0;
            } else {
                d_data[idx].x = sqrtf(d_data[idx].x);
            }
        }
        else
        {
            d_data[idx].x = 0;
            d_data[idx].y = 0;
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

        maxVal = channel_0_enable ? data[col].x : 0;
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


// CUDA kernel 函数 - 原地按列 fftshift // 仅限nrows为偶数的情况
__global__ void fftshift_columns_inplace_kernel(cufftComplex* d_data, int nrows, int ncols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int mid = nrows / 2;
    // 仅交换前mid次，避免重复交换
    for (int i = 0; i < mid; ++i) {
        int src = i * ncols + col;
        int dst = (i + mid + (nrows % 2)) * ncols + col; // 处理奇数情况
        // 交换元素
        cufftComplex temp = d_data[src];
        d_data[src] = d_data[dst];
        d_data[dst] = temp;
    }
}

__global__ void cfarKernel(const cufftComplex* data, cufftComplex* cfar_signal, int nrows, int ncols,
                           double alpha, int numGuardCells, int numRefCells, int leftBoundary, int rightBoundary) {
    /*
     * blockIdx - 块的索引
     * blockIdx.x - [0, gridDim.x-1]
     * blockIdx.y - [0, gridDim.y-1]
     * blockDim - 每个块的维度
     * threadIdx - 线程在块内的索引
     * threadIdx.x - [0, blockDim-1]
     */

    int row = blockIdx.y;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int totalTrainingCells = numGuardCells + numRefCells;
    int col_start = max(thread_id * CFAR_LENGTH, leftBoundary + totalTrainingCells);
    int col_end = min(col_start + CFAR_LENGTH, rightBoundary - totalTrainingCells);

    if (col_start >= ncols || row >= nrows) return;

    double noiseLevel_left = 0.0;
    double noiseLevel_right = 0.0;

    for (int i = col_start; i < col_end; ++i) {
        if (i == col_start) {
            for (int j = i - totalTrainingCells; j < i - numGuardCells; ++j) {
                noiseLevel_left += data[row * ncols + j].x;
            }
            for (int j = i + numGuardCells + 1; j <= i + totalTrainingCells; ++j) {
                noiseLevel_right += data[row * ncols + j].x;
            }
        }
        else {
            noiseLevel_left += data[row * ncols + i - numGuardCells - 1].x;
            noiseLevel_left -= data[row * ncols + (i - totalTrainingCells - 1)].x;
            noiseLevel_right += data[row * ncols + i + totalTrainingCells].x;
            noiseLevel_right -= data[row * ncols + i + numGuardCells].x;
        }

        double threshold = alpha * (noiseLevel_left + noiseLevel_right) / (2 * numRefCells);

        cfar_signal[row * ncols + i].x = (data[row * ncols + i].x > threshold) ? sqrtf(data[row * ncols + i].x) : 0.0f;
        cfar_signal[row * ncols + i].y = 0.0;
    }
}


__global__ void cfar_col_kernel(const cufftComplex* data, cufftComplex* cfar_signal, int nrows, int ncols,
                                double alpha, int numGuardCells, int numRefCells) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    // 定义检测范围
    int start_row = numGuardCells + numRefCells;
    int end_row = nrows - numGuardCells - numRefCells - 1;
    if (start_row > end_row) return;

    double sum_power = 0.0;
    int count = 0;

    int row = start_row;
    int ref_start1 = row - numRefCells - numGuardCells; // 前参考窗口起始
    int ref_end1 = row - numGuardCells - 1;             // 前参考窗口结束
    int ref_start2 = row + numGuardCells + 1;           // 后参考窗口起始
    int ref_end2 = row + numGuardCells + numRefCells;   // 后参考窗口结束

    // 初始计算参考窗口的功率和
    for (int r = ref_start1; r <= ref_end1; ++r) {
        int idx = r * ncols + col;
        sum_power += data[idx].x;  // 直接使用 .x 作为功率
        count++;
    }
    for (int r = ref_start2; r <= ref_end2; ++r) {
        int idx = r * ncols + col;
        sum_power += data[idx].x;  // 直接使用 .x 作为功率
        count++;
    }

    // 滑动窗口检测
    for (row = start_row; row <= end_row; ++row) {
        int cut_idx = row * ncols + col;
        float cut_power = data[cut_idx].x;  // 直接使用 .x 作为 CUT 的功率

        // 计算阈值
        double mean_power = sum_power / count;
        double threshold = alpha * mean_power;

        // 检测并标记结果
        if (cut_power > threshold) {
            cfar_signal[cut_idx].x = sqrt(cut_power);  // 标记为目标
            cfar_signal[cut_idx].y = 0.0f;
        } else {
            cfar_signal[cut_idx].x = 0.0f;  // 标记为非目标
            cfar_signal[cut_idx].y = 0.0f;
        }

        // 更新滑动窗口
        if (row < end_row) {
            // 离开的单元功率
            int leave_before_idx = ref_start1 * ncols + col;
            double leave_before_power = data[leave_before_idx].x;

            int leave_after_idx = ref_end2 * ncols + col;
            double leave_after_power = data[leave_after_idx].x;

            // 进入的单元功率
            int enter_before_idx = (ref_end1 + 1) * ncols + col;
            double enter_before_power = data[enter_before_idx].x;

            int enter_after_idx = (ref_start2 - 1) * ncols + col;
            double enter_after_power = data[enter_after_idx].x;

            // 更新功率和
            sum_power -= leave_before_power + leave_after_power;
            sum_power += enter_before_power + enter_after_power;

            // 更新参考窗口索引
            ref_start1++;
            ref_end1++;
            ref_start2++;
            ref_end2++;
        }
    }
}