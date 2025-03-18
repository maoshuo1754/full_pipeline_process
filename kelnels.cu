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


// CUDA 内核：更新队列（0速通道和20个速度通道）
__global__ void update_queues_kernel(
    const cufftComplex* frame, cufftComplex* queues, cufftComplex* queues_speed, int* indices,
    int wave_num, int pulse_num, int range_num, int queue_size, int speed_channels
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;  // 波数索引
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // 距离索引
    if (w < wave_num && r < range_num) {
        int idx = w * range_num + r;                // 全局索引
        int current_idx = indices[idx];             // 当前写入位置

        // 更新0速通道队列
        int zero_offset = w * pulse_num * range_num + 0 * range_num + r; // 0速通道位置
        int queue_base = w * range_num * queue_size + r * queue_size;
        int write_idx = queue_base + current_idx;
        queues[write_idx] = frame[zero_offset];     // 写入0速通道队列

        // 更新20个速度通道队列，从 (PULSE_NUM/2 - 10) 开始
        int speed_start = (pulse_num / 2 - 10);     // 速度通道起始索引
        for (int s = 0; s < speed_channels; ++s) {
            int speed_idx = speed_start + s;
            int speed_offset = w * pulse_num * range_num + speed_idx * range_num + r;
            int queue_speed_base = w * range_num * speed_channels * queue_size + r * speed_channels * queue_size + s * queue_size;
            int write_speed_idx = queue_speed_base + current_idx;
            queues_speed[write_speed_idx] = frame[speed_offset]; // 写入速度通道队列
        }

        indices[idx] = (current_idx + 1) % queue_size; // 更新索引（循环队列）
    }
}

// CUDA 内核：计算自卷积、标准差并判断杂波
__global__ void compute_clutter_kernel(
    const cufftComplex* queues, const cufftComplex* queues_speed, const int* indices, bool* clutter,
    int wave_num, int range_num, int queue_size, int speed_channels
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;  // 波数索引
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // 距离索引
    if (w < wave_num && r < range_num) {
        int idx = w * range_num + r;                // 全局索引
        int queue_base = w * range_num * queue_size + r * queue_size;

        // **自卷积计算**
        // 提取0速通道队列数据
        cufftComplex x[CLUTTER_QUEUE_SIZE];
        for (int i = 0; i < queue_size; ++i) {
            x[i] = queues[queue_base + i];
        }

        // 计算（共轭翻转）
        cufftComplex xt[CLUTTER_QUEUE_SIZE];
        for (int i = 0; i < queue_size; ++i) {
            xt[i].x = x[queue_size - 1 - i].x;      // 实部翻转
            xt[i].y = -x[queue_size - 1 - i].y;     // 虚部取反（共轭）
        }

        // 计算自卷积 conv(x, x^t)
        float conv_result[2 * CLUTTER_QUEUE_SIZE - 1] = {0};
        for (int n = 0; n < 2 * queue_size - 1; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < queue_size; ++k) {
                int idx_xt = n - k;
                if (idx_xt >= 0 && idx_xt < queue_size) {
                    sum += x[k].x * xt[idx_xt].x + x[k].y * xt[idx_xt].y; // 复数乘法实部
                }
            }
            conv_result[n] = sum;
        }

        // 计算归一化因子 sum(x(i) * x^t(i))
        float norm_sum = 0.0f;
        for (int i = 0; i < queue_size; ++i) {
            norm_sum += x[i].x * xt[i].x + x[i].y * xt[i].y; // 复数乘法实部
        }

        // 归一化并统计大于3dB的点数
        int count_above_3db = 0;
        const float threshold = 2.0f; // 3dB ≈ 10^(3/10) ≈ 2
        for (int n = 0; n < 2 * queue_size - 1; ++n) {
            float normalized = (norm_sum != 0) ? (conv_result[n] / norm_sum) : 0.0f;
            if (normalized > threshold) {
                count_above_3db++;
            }
        }

        // 自卷积条件：count_above_3db > 1
        bool conv_condition = (count_above_3db > 1);

        // **标准差计算**
        // 计算20个速度通道的标准差
        float std_dev_sum = 0.0f;
        for (int s = 0; s < speed_channels; ++s) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int queue_speed_base = w * range_num * speed_channels * queue_size + r * speed_channels * queue_size + s * queue_size;
            for (int i = 0; i < queue_size; ++i) {
                cufftComplex val = queues_speed[queue_speed_base + i];
                float magnitude = sqrtf(val.x * val.x + val.y * val.y); // 幅度
                sum += magnitude;
                sum_sq += magnitude * magnitude;
            }
            float mean = sum / queue_size;
            float variance = (sum_sq / queue_size) - (mean * mean);
            float std_dev = sqrtf(variance);
            std_dev_sum += std_dev;
        }
        float avg_std_dev = std_dev_sum / speed_channels; // 20个通道的标准差平均值

        // 计算0通道的幅度（最新一帧）
        cufftComplex latest_zero = x[(indices[idx] - 1 + queue_size) % queue_size]; // 最新写入的数据
        float zero_magnitude = sqrtf(latest_zero.x * latest_zero.x + latest_zero.y * latest_zero.y);

        // 标准差条件：zero_magnitude > 6 * avg_std_dev
        bool std_dev_condition = (zero_magnitude > 6.0f * avg_std_dev);

        // **杂波判断**：两个条件都满足
        clutter[idx] = (conv_condition && std_dev_condition);
    }
}