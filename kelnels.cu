//
// Created by csic724 on 2025/3/6.
//

#include "kelnels.cuh"

#include <cfloat>
#include <thrust/detail/type_traits/is_call_possible.h>
#include "utils.h"
#include "Config.h"
#include "SharedQueue.h"


__global__ void unpackKernel3D(unsigned char *threadsMemory, cufftComplex *pComplex,
                               const int *headPositions, int pulseNum, int rangeNum) {
    // 获取线程和网格索引
    int pulseIdx = blockIdx.z;  // 每个block.z处理一个头位置
    int rangeIdx = blockIdx.y * blockDim.x + threadIdx.x; // 每个线程处理一个距离单元
    int waveIdx = blockIdx.x;  // 每个block.x处理一个波束

    // 检查索引是否越界
    if (pulseIdx < pulseNum && rangeIdx < RANGE_NUM && waveIdx < WAVE_NUM) {
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


__global__ void cmpKernel(  cufftComplex *d_data, cufftComplex *thresholds,
                            bool *d_clutterMap_masked_, int nrows, int ncols,
                            int offset, int cfar_enable, double* cfar_db_offset
                            ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrows * ncols) {
        if (d_clutterMap_masked_[idx] && idx % ncols < ncols - offset)
        {
            if (d_data[idx].x < thresholds[idx + offset].x * powf(10, cfar_db_offset[idx % ncols]/10) && cfar_enable) { // 后面不在这改
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

// 对指定速度范围的实部选大，d_rows是要选的行号，num_rows是d_rows的大小
__global__ void maxRealByColumn(cufftComplex *data, float *maxValues, int* d_maxindex, int *d_rows, int num_rows, int nrows, int ncols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ind;

    if (col < ncols) {
        float maxVal = -FLT_MAX;
        for (int i = 0; i < num_rows; ++i) {
            int row = d_rows[i];
            ind = row * ncols + col;
            if (data[ind].x > maxVal) {
                maxVal = data[ind].x;
                d_maxindex[col] = row;
            }
        }
        maxValues[col] = maxVal;
    }
}

void launchMaxRealByColumn(cufftComplex* d_input, float* d_output, int* d_maxindex,  int *d_rows, int num_rows, int nrows, int ncols, cudaStream_t stream) {
    int threadsPerBlock = CUDA_BLOCK_SIZE;
    int blocks = (ncols + threadsPerBlock - 1) / threadsPerBlock;
    maxRealByColumn<<<blocks, threadsPerBlock, 0, stream>>>(d_input, d_output, d_maxindex, d_rows, num_rows, nrows, ncols);
}

// __global__ void maxKernel2D(cufftComplex *data, float *maxValues, int *speedChannels,
//                            bool* maskPtr, int nrows, int ncols, int nwaves) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;    // range dimension
//     int wave = blockIdx.y * blockDim.y + threadIdx.y;   // wave dimension
//
//     if (col < ncols && wave < nwaves) {
//         // 计算当前wave和col的全局索引
//         int base_idx = wave * nrows * ncols + col;
//         float maxVal = channel_0_enable ? data[base_idx].x : 0;
//         int maxChannel = 0;
//
//         // 在row维度上找最大值
//         for (int row = 1; row < nrows; ++row) {
//             int ind = wave * nrows * ncols + row * ncols + col;
//             if (data[ind].x > maxVal) {
//                 maxVal = data[ind].x;
//                 maxChannel = row;
//             }
//         }
//
//         // 输出结果的索引
//         int out_idx = wave * ncols + col;
//         maxValues[out_idx] = maxVal;
//         speedChannels[out_idx] = maxChannel;
//         // if (maskPtr[out_idx]) {
//         //     maxValues[out_idx] = 1000;
//         // }
//     }
// }

// 核函数：只遍历指定的 row 索引来找最大值
__global__ void maxKernel2D(cufftComplex *data, float *maxValues, int *speedChannels, int *d_chnSpeeds,
                           int *d_rows, int num_rows, int nrows, int ncols, int nwaves) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引（range 维度）
    int wave = blockIdx.y * blockDim.y + threadIdx.y; // 波索引（wave 维度）

    if (col < ncols && wave < nwaves) {
        float maxVal = -FLT_MAX;  // 初始化最大值为负无穷
        int maxChannel = -1;      // 初始化通道索引为无效值

        // 只遍历传入的 row 索引
        for (int i = 0; i < num_rows; ++i) {
            int row = d_rows[i];
            int ind = wave * nrows * ncols + row * ncols + col;
            if (data[ind].x > maxVal) {
                maxVal = data[ind].x;
                maxChannel = row;
            }
        }

        // 输出结果
        int out_idx = wave * ncols + col;
        maxValues[out_idx] = (maxChannel == -1) ? 0 : maxVal; // 如果没有有效值，返回 0
        speedChannels[out_idx] = d_chnSpeeds[maxChannel];                  // 记录对应的通道索引
    }
}

__global__ void maxKernel_rasterize(cufftComplex *data, float *maxValues, int *speedChannels, int *d_chnSpeeds,
                           double* min_speed_idx, double* max_speed_idx, int nrows, int ncols, int nwaves) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引（range 维度）
    int wave = blockIdx.y * blockDim.y + threadIdx.y; // 波索引（wave 维度）

    if (col < ncols && wave < nwaves) {
        float maxVal = -FLT_MAX;  // 初始化最大值为负无穷
        int maxChannel = -1;      // 初始化通道索引为无效值

        // 只遍历传入的 row 索引
        for (int row = int(min_speed_idx[col]); row < max_speed_idx[col]; ++row) {
            int ind = wave * nrows * ncols + row * ncols + col;
            if (data[ind].x > maxVal) {
                maxVal = data[ind].x;
                maxChannel = row;
            }
        }

        for (int row = int(nrows - max_speed_idx[col]); row < nrows - min_speed_idx[col]; ++row) {
            int ind = wave * nrows * ncols + row * ncols + col;
            if (data[ind].x > maxVal) {
                maxVal = data[ind].x;
                maxChannel = row;
            }
        }

        // 输出结果
        int out_idx = wave * ncols + col;
        maxValues[out_idx] = (maxChannel == -1) ? 0 : maxVal; // 如果没有有效值，返回 0
        speedChannels[out_idx] = d_chnSpeeds[maxChannel];                  // 记录对应的通道索引
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
        int dst = (i + mid) * ncols + col;
        // 交换元素
        cufftComplex temp = d_data[src];
        d_data[src] = d_data[dst];
        d_data[dst] = temp;
    }
}



// 核函数：对 cufftComplex 矩阵按行进行 fftshift 原地
__global__ void fftshift_rows_inplace(cufftComplex *data, int pulsenum, int rangenum) {
    // 声明动态共享内存
    extern __shared__ cufftComplex shared_row[];

    // 确定当前线程块处理的行
    int row = blockIdx.x;
    if (row >= pulsenum) return;

    int tid = threadIdx.x;      // 线程在块内的索引
    int stride = blockDim.x;    // 线程步幅

    // 将全局内存的数据加载到共享内存
    for (int i = tid; i < rangenum; i += stride) {
        shared_row[i] = data[row * rangenum + i];
    }
    __syncthreads();  // 确保所有线程加载完成

    // 计算移位量并写回全局内存
    int shift = (rangenum + 1) / 2;
    for (int i = tid; i < rangenum; i += stride) {
        int src_idx = (i + shift) % rangenum;
        data[row * rangenum + i] = shared_row[src_idx];
    }
}

// 核函数：对 cufftComplex 矩阵按列进行 fftshift 原地
__global__ void fftshift_cols_inplace(cufftComplex *data, int pulsenum, int rangenum) {
    // 声明动态共享内存（大小为pulsenum）
    extern __shared__ cufftComplex shared_col[];

    // 确定当前线程块处理的列
    int col = blockIdx.x;
    if (col >= rangenum) return;

    int tid = threadIdx.x;      // 线程在块内的索引
    int stride = blockDim.x;    // 线程步幅

    // 将全局内存的列数据加载到共享内存
    for (int i = tid; i < pulsenum; i += stride) {
        shared_col[i] = data[i * rangenum + col];
    }
    __syncthreads();  // 确保所有线程加载完成

    // 计算列方向的移位量并写回全局内存
    int shift = (pulsenum + 1) / 2;
    for (int i = tid; i < pulsenum; i += stride) {
        int src_idx = (i + shift) % pulsenum;
        data[i * rangenum + col] = shared_col[src_idx];
    }
}

// 外层函数：根据dim调用行或列fftshift，支持CUDA流
void fftshift(cufftComplex *data, int pulsenum, int rangenum, int dim, cudaStream_t stream) {
    // 设置线程块和网格
    dim3 block(256); // 每块256个线程，可根据GPU架构调整
    dim3 grid;
    size_t shared_mem_size;

    if (dim == 2) { // 行fftshift
        grid = dim3(pulsenum); // 每个线程块处理一行
        shared_mem_size = rangenum * sizeof(cufftComplex); // 共享内存大小为一行
        fftshift_rows_inplace<<<grid, block, shared_mem_size, stream>>>(data, pulsenum, rangenum);
    } else if (dim == 1) { // 列fftshift
        grid = dim3(rangenum); // 每个线程块处理一列
        shared_mem_size = pulsenum * sizeof(cufftComplex); // 共享内存大小为一列
        fftshift_cols_inplace<<<grid, block, shared_mem_size, stream>>>(data, pulsenum, rangenum);
    } else {
        fprintf(stderr, "Invalid dim parameter: %d. Must be 1 (rows) or 2 (cols).\n", dim);
        return;
    }

    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
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

__global__ void zeroEdgeRows(cufftComplex* data, int pulse_num, int range_num, int NoiseWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    if (col < range_num) {
        // 行优先存储：第 col 列的元素分散存储
        int offset = col;

        // 前 NoiseWidth + 1行置零
        for (int row = 0; row < NoiseWidth + 1; row++) {
            data[offset + row * range_num].x = 0.0f; // 实部
            data[offset + row * range_num].y = 0.0f; // 虚部
        }

        // 后 NoiseWidth 行置零
        for (int row = pulse_num - NoiseWidth; row < pulse_num; row++) {
            data[offset + row * range_num].x = 0.0f; // 实部
            data[offset + row * range_num].y = 0.0f; // 虚部
        }
    }
}


// CUDA 内核：更新队列（0速通道和20个速度通道）
__global__ void update_queues_kernel(
    const cufftComplex* frame, cufftComplex* queues, cufftComplex* queues_speed, int* indices,
    int pulse_num, int range_num, int queue_size, int speed_channels
) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < range_num) {
        int idx = r;
        int current_idx = indices[idx];
        int queue_base = r * queue_size;
        int write_idx = queue_base + current_idx;

        // Update zero-speed channel
        int zero_offset = r;
        queues[write_idx] = frame[zero_offset];

        // Update speed channels
        int speed_start = (pulse_num / 2 - speed_channels / 2);
        for (int s = 0; s < speed_channels; ++s) {
            int speed_idx = speed_start + s;
            int speed_offset = speed_idx * range_num + r;
            int queue_speed_base = r * speed_channels * queue_size + s * queue_size;
            int write_speed_idx = queue_speed_base + current_idx;
            queues_speed[write_speed_idx] = frame[speed_offset];
        }

        indices[idx] = (current_idx + 1) % queue_size;
    }
}

// CUDA 内核：计算自卷积、标准差并判断杂波
__global__ void compute_clutter_kernel(
    const cufftComplex* queues, const cufftComplex* queues_speed, const int* indices, bool* clutter,
    int range_num, int queue_size, int speed_channels
) {
    int r = blockIdx.y * blockDim.y + threadIdx.y; // 距离索引
    if (r < range_num) {
        int idx = r;
        int queue_base = r * queue_size;

        // **自卷积计算**
        // 提取0速通道队列数据
        cufftComplex x[CLUTTER_QUEUE_SIZE];
        for (int i = 0; i < queue_size; ++i) {
            x[i] = queues[queue_base + i];
        }

        // 计算共轭翻转
        cufftComplex xt[CLUTTER_QUEUE_SIZE];
        for (int i = 0; i < queue_size; ++i) {
            xt[i].x = x[queue_size - 1 - i].x; // 实部翻转
            xt[i].y = -x[queue_size - 1 - i].y; // 虚部取反（共轭）
        }

        // 计算自卷积 conv(x, xt)，并取模
        float conv_result[2 * CLUTTER_QUEUE_SIZE - 1] = {0};
        for (int n = 0; n < 2 * queue_size - 1; ++n) {
            float real_sum = 0.0f, imag_sum = 0.0f;
            for (int k = 0; k < queue_size; ++k) {
                int idx_xt = n - k;
                if (idx_xt >= 0 && idx_xt < queue_size) {
                    float real_xk = x[k].x, imag_xk = x[k].y;
                    float real_xt = xt[idx_xt].x, imag_xt = xt[idx_xt].y;
                    real_sum += real_xk * real_xt - imag_xk * imag_xt;
                    imag_sum += real_xk * imag_xt + imag_xk * real_xt;
                }
            }
            conv_result[n] = sqrtf(real_sum * real_sum + imag_sum * imag_sum);
        }

        // 计算归一化因子 sum(abs(x(i) .* xt(i)))
        float norm_sum = 0.0f;
        for (int i = 0; i < queue_size; ++i) {
            // x[i] * xt[i]
            float real = x[i].x * xt[i].x - x[i].y * xt[i].y; // 复数乘法实部
            float imag = x[i].x * xt[i].y + x[i].y * xt[i].x; // 复数乘法虚部
            norm_sum += sqrtf(real * real + imag * imag); // 计算模并累加
        }

        // 归一化并统计大于3dB的点数
        int count_above_3db = 0;
        const float threshold = powf(10, -3.0f / 20);
        for (int n = 0; n < 2 * queue_size - 1; ++n) {
            float normalized = (norm_sum != 0) ? (conv_result[n] / norm_sum) : 0.0f;
            if (normalized > threshold) count_above_3db++;
        }
        bool conv_condition = (count_above_3db > 1);

        // Standard deviation
        float sum = 0.0f, sum_sq = 0.0f;
        for (int s = 0; s < speed_channels; ++s) {
            int queue_speed_base = r * speed_channels * queue_size + s * queue_size;
            for (int i = 0; i < queue_size; ++i) {
                cufftComplex val = queues_speed[queue_speed_base + i];
                float magnitude = sqrtf(val.x * val.x + val.y * val.y);
                sum += magnitude;
                sum_sq += magnitude * magnitude;
            }
        }
        int total_points = speed_channels * queue_size;
        float mean = sum / total_points;
        float variance = (sum_sq / total_points) - (mean * mean);
        float std_dev = sqrtf(variance);

        // 计算0通道的幅度（最新一帧）
        cufftComplex latest_zero = x[(indices[idx] - 1 + queue_size) % queue_size]; // 最新写入的数据
        float zero_magnitude = sqrtf(latest_zero.x * latest_zero.x + latest_zero.y * latest_zero.y);

        // 标准差条件：zero_magnitude > 6 * std_dev
        bool std_dev_condition = (zero_magnitude > 6.0f * std_dev);

        clutter[idx] = std_dev_condition && conv_condition;
    }
}


// CUDA Kernel：计算对数并更新杂波图
__global__ void processClutterMapKernel(cufftComplex* d_data, float* d_clutter_map, bool* d_clutterMap_masked,
    size_t size, int range_num, float alpha, float forgetting_factor, float clutter_db_offset, double* d_rasterize_thresholds_wave

    ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx % NFFT <= range_num) {
        // 计算幅值的平方
        float magnitude_squared = d_data[idx].x * d_data[idx].x + d_data[idx].y * d_data[idx].y;
        // 计算对数幅值（与 Log10Functor 一致）
        float log_magnitude = 10 * log10f(magnitude_squared);

        // 计算阈值
        float threshold = alpha + d_clutter_map[idx] + clutter_db_offset + d_rasterize_thresholds_wave[idx % NFFT];
        if (log_magnitude > threshold) {
            d_clutterMap_masked[idx] = true;
        } else {
            d_clutterMap_masked[idx] = false;
        }
        d_clutter_map[idx] = forgetting_factor * d_clutter_map[idx] + (1 - forgetting_factor) * log_magnitude;
    }
}


__global__ void MTIkernel3(cufftComplex *data, int nrows, int ncols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int currentIndex, nextIndex, lastIndex;
    cufftComplex current, next, last;
    if (col < ncols) {
        for (int row = 0; row < nrows - 2; row++) {
            currentIndex = row * ncols + col;
            nextIndex = (row + 1) * ncols + col;
            lastIndex = (row + 2) * ncols + col;

            current = data[currentIndex];
            next = data[nextIndex];
            last = data[lastIndex];
            data[currentIndex].x = current.x + last.x - 2 * next.x;
            data[currentIndex].y = current.y + last.y - 2 * next.y;
        }
        nextIndex = (nrows - 1) * ncols + col;
        data[nextIndex].x = 0.0f;
        data[nextIndex].y = 0.0f;

        nextIndex = (nrows - 2) * ncols + col;
        data[nextIndex].x = 0.0f;
        data[nextIndex].y = 0.0f;
    }
}

__global__ void MTIkernel2(cufftComplex *data, int nrows, int ncols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int currentIndex, nextIndex;
    cufftComplex current, next;
    if (col < ncols) {
        for (int row = 0; row < nrows - 1; row++) {
            currentIndex = row * ncols + col;
            nextIndex = (row + 1) * ncols + col;

            current = data[currentIndex];
            next = data[nextIndex];
            data[currentIndex].x = next.x - current.x;
            data[currentIndex].y = next.y - current.y;
        }
        nextIndex = (nrows - 1) * ncols + col;
        data[nextIndex].x = 0.0f;
        data[nextIndex].y = 0.0f;
    }
}

// 核函数：计算gn和hn，并直接存储到复用缓冲区
__global__ void compute_gn_hn(cufftComplex *x, cufftComplex *gn, cufftComplex *hn,
                             float *alpha, int pulsenum, int rangenum) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index % rangenum; // rangenum 维度
    int j = index / rangenum; // pulsenum 或 gFFTNumber 维度

    float pi = 3.141592653589793f;
    float alpha_i = alpha[i];
    float A_phase = -pi * alpha_i;
    float W_phase = -2.0f * pi * alpha_i / (float)pulsenum;

    // 计算 gn
    if (j < pulsenum) {
        float idx = (float)j;
        // A^(-j) * W^(j^2/2)
        cufftComplex A_term = make_cuComplex(cosf(A_phase * -idx), sinf(A_phase * -idx));
        cufftComplex W_term = make_cuComplex(cosf(W_phase * idx * idx / 2.0f), sinf(W_phase * idx * idx / 2.0f));
        cufftComplex tmpTerm = cuCmulf(cuCmulf(x[j * rangenum + i], A_term), W_term);
        gn[j * rangenum + i] = tmpTerm;
    }


    // 计算 hn
    if (j < pulsenum) {
        float idx = (float)j;
        hn[j * rangenum + i] = make_cuComplex(cosf(W_phase * -(idx * idx) / 2.0f), sinf(W_phase * -(idx * idx) / 2.0f));
    } else if (j == pulsenum) {
        hn[j * rangenum + i] = make_cuComplex(0.0f, 0.0f);
    } else if (j < 2 * pulsenum) {
        float idx_m = (float)(2 * pulsenum - j);
        hn[j * rangenum + i] = make_cuComplex(cosf(W_phase * -(idx_m * idx_m) / 2.0f), sinf(W_phase * -(idx_m * idx_m) / 2.0f));
    }
}

// 核函数：计算yn并存储到xSlowCZT
__global__ void compute_yn(cufftComplex *YFFT, cufftComplex *xSlowCZT, float *alpha,
                          int pulsenum, int rangenum, int gFFTNumber) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index % rangenum; // rangenum 维度
    int j = index / rangenum; // pulsenum 或 gFFTNumber 维度
    if (i >= rangenum || j >= pulsenum) return;

    float pi = 3.141592653589793f;
    float W_phase = -2.0f * pi * alpha[i] / pulsenum;
    float idx = (float)j;
    cufftComplex W_term = make_cuComplex(cosf(W_phase * idx * idx / 2.0f), sinf(W_phase * idx * idx / 2.0f));
    xSlowCZT[j * rangenum + i] = cuCmulf(YFFT[j * rangenum + i], W_term);
}

// 计算两个矩阵的点乘
__global__ void elementWiseMulKernel(cufftComplex *d_a, cufftComplex *d_b, cufftComplex *d_c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        cufftComplex temp_a = d_a[idx];
        cufftComplex temp_b = d_b[idx];
        d_c[idx].x = temp_a.x * temp_b.x - temp_a.y * temp_b.y;
        d_c[idx].y = temp_a.x * temp_b.y + temp_a.y * temp_b.x;
    }
}


__global__ void generateDechirpRes(cufftComplex* coef, float PRT, float TargetAccVelMax, int PulseNumber, int accNum, float Lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 脉冲索引
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // 加速度索引
    if (idx < PulseNumber && idy < accNum) {
        float step = 2 * TargetAccVelMax / (accNum - 1);
        float phase = -2.0f * M_PI * (-TargetAccVelMax + idy * step) * (PRT * idx) * (PRT * idx) / Lambda;
        coef[idx * accNum + idy].x = cosf(phase);
        coef[idx * accNum + idy].y = sinf(phase);
    }
}

__global__ void generateDechirpCoef(cufftComplex* coef, float PRT, float accScanStart, float accScanStep, int PulseNumber, int accNum, float Lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 时间索引
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // 加速度索引
    if (idx < PulseNumber && idy < accNum) {
        float phase = -2.0f * M_PI * (accScanStart + idy * accScanStep) * (PRT * idx) * (PRT * idx) / Lambda;
        coef[idx * accNum + idy].x = cosf(phase);
        coef[idx * accNum + idy].y = sinf(phase);
    }
}

// 计算FirstDechirpRes， FirstDechirpRes(:, i) = x .* FirstDechirpCoef(:, i) .* WinCoef;
__global__ void FirstDechirpRes(cufftComplex *d_FirstDechirpCoef, cufftComplex* data, float* windowCoef, cufftComplex* d_FirstDechirpRes, int accNum, int PulseNumber, int RangeNum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 脉冲索引
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // 加速度索引
    if (idx < PulseNumber && idy < accNum) {
        // Compute the index in the matrices
        int matrix_idx = idx * accNum + idy;
        int data_idx = idx * RangeNum;

        // Complex multiplication: data[idx] * d_FirstDechirpCoef[matrix_idx]
        cufftComplex temp;
        temp.x = data[data_idx].x * d_FirstDechirpCoef[matrix_idx].x - data[data_idx].y * d_FirstDechirpCoef[matrix_idx].y; // Real part
        temp.y = data[data_idx].x * d_FirstDechirpCoef[matrix_idx].y + data[data_idx].y * d_FirstDechirpCoef[matrix_idx].x; // Imaginary part

        // Scale by windowCoef[idx] (real number)
        d_FirstDechirpRes[matrix_idx].x = temp.x * windowCoef[idx];
        d_FirstDechirpRes[matrix_idx].y = temp.y * windowCoef[idx];
    }
}


// CUDA 函数：对 nrows × ncols 矩阵按列执行 FFT
// 参数：
//   d_matrix - 显存中的输入/输出矩阵，按行优先存储，类型为 cufftComplex
//   nrows - 矩阵行数（每列的 FFT 长度）
//   ncols - 矩阵列数（批量 FFT 的数量）
//   stream - CUDA 流，用于异步执行
// 返回值：true 表示成功，false 表示失败
bool columnFFT(cufftComplex* d_matrix, int nrows, int ncols, int batch, cudaStream_t stream) {
    // 定义静态 map，存储 ncols 到 cufftHandle 的映射
    static std::map<int, cufftHandle> planCache;

    // 检查是否已存在对应 ncols 的计划
    cufftHandle plan;
    auto it = planCache.find(ncols);
    if (it != planCache.end()) {
        // 计划已存在，重用
        plan = it->second;
    } else {
        // 创建新计划
        int rank = 1; // 1D FFT
        int n[] = {nrows}; // FFT 长度（每列的长度）
        int inembed[] = {ncols}; // 输入矩阵的总元素数（行优先存储）
        int onembed[] = {ncols}; // 输出矩阵的总元素数（行优先存储）
        int istride = ncols; // 输入步长：跳过 ncols 个元素到下一行
        int ostride = ncols; // 输出步长：跳过 ncols 个元素到下一行
        int idist = 1; // 输入列之间的距离（连续列）
        int odist = 1; // 输出列之间的距离（连续列）

        checkCufftErrors(cufftPlanMany(&plan, rank, n,
                                       inembed, istride, idist,
                                       onembed, ostride, odist,
                                       CUFFT_C2C, batch));

        // 存储计划到 map
        planCache[ncols] = plan;
    }

    // 设置 CUDA 流
    if (stream != nullptr) {
        checkCufftErrors(cufftSetStream(plan, stream));
    }

    // 执行 FFT
    checkCufftErrors(cufftExecC2C(plan, d_matrix, d_matrix, CUFFT_FORWARD));

    // 注意：计划不在这里销毁，保持在 map 中重用
    return true;
}

// 核函数：矩阵每列与向量逐元素相乘
__global__ void column_vector_multiply(cufftComplex* matrix, float* vector, int nrows, int ncols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引

    if (row < nrows && col < ncols) {
        int idx = row * ncols + col; // 行优先存储的矩阵索引
        // cufftComplex 与 float 相乘：(a + bi) * c = (a * c) + (b * c)i
        float a = matrix[idx].x; // 矩阵的实部
        float b = matrix[idx].y; // 矩阵的虚部
        float c = vector[row];   // 向量的实数值
        matrix[idx].x = a * c;   // 实部结果
        matrix[idx].y = b * c;   // 虚部结果
    }
}

// 核函数：将矩阵的第 col 列复制到向量
__global__ void copy_column_to_vector_kernel(cufftComplex* matrix, cufftComplex* vector, int nrows, int ncols, int col) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // 行索引

    if (row < nrows) {
        int idx = row * ncols + col; // 行优先存储的矩阵索引
        vector[row] = matrix[idx];   // 复制矩阵第 col 列的元素到向量
    }
}

// 将矩阵的第 col 列复制到向量
void copyColumnToVector(cufftComplex* matrix, cufftComplex* vector, int nrows, int ncols, int col, cudaStream_t stream_) {
    // 验证输入参数
    if (col < 0 || col >= ncols || nrows <= 0 || ncols <= 0) {
        // 可以添加错误处理，例如抛出异常或打印错误
        return;
    }

    // 配置线程块和网格大小
    int threadsPerBlock = 256; // 每个块的线程数，可根据硬件调整
    int blocksPerGrid = (nrows + threadsPerBlock - 1) / threadsPerBlock; // 向上取整

    // 启动核函数
    copy_column_to_vector_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
        matrix, vector, nrows, ncols, col
    );

    // 可选择同步（调试时使用，生产环境通常依赖流）
    // cudaStreamSynchronize(stream_);
}