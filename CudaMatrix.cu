#include "CudaMatrix.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cuComplex.h>
#include "Config.h"

using namespace std;

// Helper functions to check errors
void CudaMatrix::checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(result));
    }
}

void CudaMatrix::checkCublasErrors(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw runtime_error("cuBLAS error");
    }
}

void CudaMatrix::checkCufftErrors(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: ";
        switch (result) {
            case CUFFT_INVALID_PLAN:
                std::cerr << "CUFFT_INVALID_PLAN";
                break;
            case CUFFT_ALLOC_FAILED:
                std::cerr << "CUFFT_ALLOC_FAILED";
                break;
            case CUFFT_INVALID_TYPE:
                std::cerr << "CUFFT_INVALID_TYPE";
                break;
            case CUFFT_INVALID_VALUE:
                std::cerr << "CUFFT_INVALID_VALUE";
                break;
            case CUFFT_INTERNAL_ERROR:
                std::cerr << "CUFFT_INTERNAL_ERROR";
                break;
            case CUFFT_EXEC_FAILED:
                std::cerr << "CUFFT_EXEC_FAILED";
                break;
            case CUFFT_SETUP_FAILED:
                std::cerr << "CUFFT_SETUP_FAILED";
                break;
            case CUFFT_INVALID_SIZE:
                std::cerr << "CUFFT_INVALID_SIZE";
                break;
            default:
                std::cerr << "Unknown error";
        }
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Constructor
CudaMatrix::CudaMatrix(int rows, int cols) : nrows(rows), ncols(cols), data(nullptr), initFromDevice(false) {
    allocateMemory();
}

// Constructor for creating a matrix from host data
CudaMatrix::CudaMatrix(const std::vector<std::vector<cufftComplex>>& hostData)
        : nrows(hostData.size()), ncols(hostData[0].size()), data(nullptr), initFromDevice(false) {
    allocateMemory();
    // Copy data from host to device
    std::vector<cufftComplex> flattenedData(nrows * ncols);
    for (int i = 0; i < nrows; ++i) {
        std::copy(hostData[i].begin(), hostData[i].end(), flattenedData.begin() + i * ncols);
    }
    checkCudaErrors(cudaMemcpy(data, flattenedData.data(), sizeof(cufftComplex) * nrows * ncols, cudaMemcpyHostToDevice));
}

CudaMatrix::CudaMatrix(int rows, int cols, std::vector<cufftComplex> hostData) : nrows(rows), ncols(cols), data(nullptr), initFromDevice(false) {
    if (hostData.size() != nrows * ncols) {
        throw std::runtime_error("Host data size does not match matrix dimensions.");
    }
    checkCudaErrors(cudaMalloc(&data, sizeof(cufftComplex) * nrows * ncols));
    checkCudaErrors(cudaMemcpy(data, hostData.data(), sizeof(cufftComplex) * nrows * ncols, cudaMemcpyHostToDevice));
}

CudaMatrix::CudaMatrix(int rows, int cols, cufftComplex* hostData) : nrows(rows), ncols(cols), data(nullptr), initFromDevice(false) {
    checkCudaErrors(cudaMalloc(&data, sizeof(cufftComplex) * nrows * ncols));
    checkCudaErrors(cudaMemcpy(data, hostData, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyHostToDevice));
}

// Copy constructor
CudaMatrix::CudaMatrix(const CudaMatrix& other) : nrows(other.nrows), ncols(other.ncols), data(nullptr), initFromDevice(false) {
//    cout <<"copy constructor" << endl;
    allocateMemory();
    checkCudaErrors(cudaMemcpy(data, other.data, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyDeviceToDevice));

}

// Move constructor
CudaMatrix::CudaMatrix(CudaMatrix&& other) noexcept : nrows(other.nrows), ncols(other.ncols), data(other.data), initFromDevice(false) {
    other.data = nullptr;
    other.nrows = 0;
    other.ncols = 0;
}

CudaMatrix::CudaMatrix(int rows, int cols, cufftComplex* hostData, bool deviceFlag) : nrows(rows), ncols(cols), data(hostData), initFromDevice(true)  {
}


// Destructor
CudaMatrix::~CudaMatrix() {
    deallocateMemory();
//    cout << "Destructor" << endl;
}

// Copy data from host to device
void CudaMatrix::copyFromHost(const vector<cufftComplex> &hostData) {
    if (hostData.size() != nrows * ncols) {
        throw runtime_error("Host data size does not match matrix size.");
    }
    checkCudaErrors(cudaMemcpy(data, hostData.data(), sizeof(cufftComplex) * nrows * ncols, cudaMemcpyHostToDevice));
}

void CudaMatrix::copyFromHost(cudaStream_t _stream, int rows, int cols, const cufftComplex *hostData) {
//    cout << rows << " " << nrows << "  " << cols << " " << ncols << endl;
    if (rows != nrows || cols != ncols) {
        deallocateMemory();
        nrows = rows;
        ncols = cols;
        allocateMemory();
        cout << "copyFromHost(): Copy size not match!" << endl;
    }
//    checkCudaErrors(cudaMemcpy(data, hostData, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyHostToDevice));
    // 异步拷贝
    checkCudaErrors(cudaMemcpyAsync(data, hostData, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyHostToDevice, _stream));
}

// Copy data from device to host
void CudaMatrix::copyToHost(vector<cufftComplex> &hostData) const {
    hostData.resize(nrows * ncols);
    checkCudaErrors(cudaMemcpy(hostData.data(), data, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyDeviceToHost));
}

// 将显存的数据拷贝回内存，hostData需要提前new好
void CudaMatrix::copyToHost(cufftComplex* hostData) const {
    checkCudaErrors(cudaMemcpy(hostData, data, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyDeviceToHost));
}

// Get matrix data as 2D vector
vector<vector<cufftComplex>> CudaMatrix::to2DVector() const {
    vector<cufftComplex> flatData;
    copyToHost(flatData);
    vector<vector<cufftComplex>> result(nrows, vector<cufftComplex>(ncols));

    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            result[i][j] = flatData[i * ncols + j];
        }
    }
    return result;
}

// Fill matrix with random values
void CudaMatrix::fillWithRandomValues() {
    vector<cufftComplex> hostData(nrows * ncols);
    for (int i = 0; i < nrows * ncols; ++i) {
        hostData[i].x = static_cast<float>(rand()) / RAND_MAX;
        hostData[i].y = static_cast<float>(rand()) / RAND_MAX;
    }
    checkCudaErrors(cudaMemcpy(data, hostData.data(), nrows * ncols * sizeof(cufftComplex), cudaMemcpyHostToDevice));
}

// Print matrix for debugging
void CudaMatrix::print() const {
    vector<cufftComplex> hostData;
    copyToHost(hostData);

    cout << "array([" << endl;
    for (int i = 0; i < nrows; ++i) {
        cout << "  [";
        for (int j = 0; j < ncols; ++j) {
            cout << std::fixed << std::setprecision(5) << std::setw(6)
                 << hostData[i * ncols + j].x << " + "
                 << std::fixed << std::setprecision(5) << std::setw(6)
                 << hostData[i * ncols + j].y << "j";
            if (j < ncols - 1) {
                cout << ",  ";
            }
        }
        cout << "]";
        if (i < nrows - 1) {
            cout << ",";
        }
        cout << endl;
    }
    cout << "])" << endl;
}

void CudaMatrix::print(int row, float des) const {
    if (row >= nrows || row < 0) {
        throw runtime_error("Index out of bounds.");
    }

    vector<cufftComplex> hostData;
    copyToHost(hostData);
    int i = row;
    int count = 0;
    for (int j = 0; j < ncols; ++j) {
        if (fabs(hostData[i * ncols + j].x - des) < 1e-5 && fabs(hostData[i * ncols + j].y - des - 4096.0) < 1e-5) {
            count ++;
        } else { break; }
    }

    cout << "count:" << count << endl;
}

void CudaMatrix::print(int row) const {
    if (row >= nrows || row < 0) {
        throw runtime_error("Index out of bounds.");
    }

    vector<cufftComplex> hostData;
    copyToHost(hostData);

    cout << "array of row" << row << "([" << endl;
    int i = row;
    cout << "  [";
    for (int j = 0; j < ncols; ++j) {
        cout << std::fixed << std::setprecision(5) << std::setw(6)
             << hostData[i * ncols + j].x << " + "
             << std::fixed << std::setprecision(5) << std::setw(6)
             << hostData[i * ncols + j].y << "j";
        if (j < ncols - 1) {
            cout << ",  ";
        }
    }
    cout << "]";
    if (i < nrows - 1) {
        cout << ",";
    }
    cout << endl;

    cout << "])" << endl;
}

void CudaMatrix::printLargerThan0() const {
    vector<cufftComplex> hostData;
    copyToHost(hostData);

    for (int i = 0; i < this->size(); i++) {
        if (hostData[i].x > 0) {
            cout << i << ":" << hostData[i].x << endl;
        }
    }
}

void CudaMatrix::printShape() const {
    cout << "CudaMatrix Shape: (" << nrows << ", " << ncols << ")" << endl;
}

void CudaMatrix::setElement(int x, int y, cufftComplex value) {
    if (x >= nrows || y >= ncols || x < 0 || y < 0) {
        throw runtime_error("Index out of bounds.");
    }
    cudaMemcpy(data + x * ncols + y, &value, sizeof(cufftComplex), cudaMemcpyHostToDevice);
}

// Kernel for element-wise multiplication
__global__ void elementWiseMulKernel(cufftComplex *d_a, cufftComplex *d_b, cufftComplex *d_c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        cufftComplex temp_a = d_a[idx];
        cufftComplex temp_b = d_b[idx];
        d_c[idx].x = temp_a.x * temp_b.x - temp_a.y * temp_b.y;
        d_c[idx].y = temp_a.x * temp_b.y + temp_a.y * temp_b.x;
    }
}

// Element-wise multiplication method
void CudaMatrix::elementWiseMul(const CudaMatrix &other, cudaStream_t _stream) const {
    if (nrows != other.nrows || ncols != other.ncols) {
        throw std::runtime_error("Matrix dimensions must match for element-wise multiplication.");
    }

    int size = nrows * ncols;
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    elementWiseMulKernel<<<gridSize, blockSize, 0, _stream>>>(data, other.data, data, size);
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

// 原地做行点乘，this矩阵是nxm的，other矩阵是1xm的，让this矩阵的每一行都点乘other矩阵
void CudaMatrix::rowWiseMul(const CudaMatrix &other, cudaStream_t _stream) {
    if (other.nrows != 1 || other.ncols != ncols) {
        throw std::runtime_error("The other matrix must be of dimensions 1 x m.");
    }

    int size = nrows * ncols;
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    rowWiseMulKernel<<<gridSize, blockSize, 0, _stream>>>(data, other.data, nrows, ncols);
}


// Kernel for transpose
__global__ void transposeKernel(cufftComplex* input, cufftComplex* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

CudaMatrix CudaMatrix::T(bool inplace) {
    // Non in-place transpose
    CudaMatrix result(ncols, nrows);

    dim3 blockSize(16, 16);
    dim3 gridSize((ncols + blockSize.x - 1) / blockSize.x, (nrows + blockSize.y - 1) / blockSize.y);
    transposeKernel<<<gridSize, blockSize>>>(data, result.data, nrows, ncols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    if(inplace){
        *this = std::move(result);
        return {};
    }
    return result;
}

// Kernel for matrix multiplication
__global__ void
matMulKernel(const cufftComplex *A, const cufftComplex *B, cufftComplex *C, int Arows, int Acols, int Bcols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Arows && col < Bcols) {
        cufftComplex sum = make_cuComplex(0.0f, 0.0f);
        for (int i = 0; i < Acols; ++i) {
            cufftComplex a = A[row * Acols + i];
            cufftComplex b = B[i * Bcols + col];
            sum.x += a.x * b.x - a.y * b.y;
            sum.y += a.x * b.y + a.y * b.x;
        }
        C[row * Bcols + col] = sum;
    }
}

// Matrix multiplication method using custom kernel
CudaMatrix CudaMatrix::operator*(const CudaMatrix &other) const {
    if (ncols != other.nrows) {
        throw runtime_error("Matrix dimensions must match for multiplication.");
    }

    CudaMatrix result(nrows, other.ncols);

    dim3 blockSize(16, 16);
    dim3 gridSize((other.ncols + blockSize.x - 1) / blockSize.x, (nrows + blockSize.y - 1) / blockSize.y);
    matMulKernel<<<gridSize, blockSize>>>(data, other.data, result.data, nrows, ncols, other.ncols);

    checkCudaErrors(cudaDeviceSynchronize());

    return result;
}

bool CudaMatrix::operator==(const CudaMatrix& other) const {
    if (nrows != other.nrows || ncols != other.ncols) {
        return false;
    }

    std::vector<cufftComplex> hostDataThis(nrows * ncols);
    std::vector<cufftComplex> hostDataOther(nrows * ncols);

    checkCudaErrors(cudaMemcpy(hostDataThis.data(), data, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hostDataOther.data(), other.data, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyDeviceToHost));

    const float epsilon = 1e-6;

    for (int i = 0; i < nrows * ncols; ++i) {
        if (std::fabs(hostDataThis[i].x - hostDataOther[i].x) > epsilon ||
            std::fabs(hostDataThis[i].y - hostDataOther[i].y) > epsilon) {
            return false;
        }
    }

    return true;
}

// Copy assignment operator
CudaMatrix& CudaMatrix::operator=(const CudaMatrix& other) {
    if (this != &other) {
        deallocateMemory();
        nrows = other.nrows;
        ncols = other.ncols;
        allocateMemory();
        checkCudaErrors(cudaMemcpy(data, other.data, sizeof(cufftComplex) * nrows * ncols, cudaMemcpyDeviceToDevice));
    }
    return *this;
}

// Move assignment operator
CudaMatrix& CudaMatrix::operator=(CudaMatrix&& other) noexcept {
    if (this != &other) {
        deallocateMemory();
        nrows = other.nrows;
        ncols = other.ncols;
        data = other.data;
        other.data = nullptr;
        other.nrows = 0;
        other.ncols = 0;
    }
    return *this;
}

void CudaMatrix::fft(cufftHandle &plan) const{
    checkCufftErrors(cufftExecC2C(plan, data, data, CUFFT_FORWARD));
}

void CudaMatrix::fft_by_col(cufftHandle &plan) {
    checkCufftErrors(cufftExecC2C(plan, data, data, CUFFT_FORWARD));
}



__global__ void zeroPadAndCopy(cufftComplex* idata, cufftComplex* odata, int nrows, int ncols, int nPoints) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        int rowOffsetSrc = row * ncols;
        int rowOffsetDst = row * nPoints;
        for (int col = 0; col < ncols; ++col) {
            odata[rowOffsetDst + col] = idata[rowOffsetSrc + col];
        }
        for (int col = ncols; col < nPoints; ++col) {
            odata[rowOffsetDst + col].x = 0.0f;
            odata[rowOffsetDst + col].y = 0.0f;
        }
    }
}

void CudaMatrix::fft_N(int nPoints) {
    if (nPoints < ncols) {
        throw std::runtime_error(
                "Number of FFT points must be greater than or equal to the number of columns in the matrix.");
    }

    // Create a result matrix with the same number of rows and the specified number of points
    CudaMatrix result(nrows, nPoints);

    // Set up CUDA kernel configuration
    int threadsPerBlock = 128;
    int blocksPerGrid = (nrows + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel to copy data and zero-pad
    zeroPadAndCopy<<<blocksPerGrid, threadsPerBlock>>>(data, result.data, nrows, ncols, nPoints);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Create a cufftHandle for the batched FFT plan
    cufftHandle plan;
    checkCufftErrors(cufftPlan1d(&plan, nPoints, CUFFT_C2C, nrows));

    // Execute the FFT plan
    checkCufftErrors(cufftExecC2C(plan, result.data, result.data, CUFFT_FORWARD));

    // Clean up
    checkCufftErrors(cufftDestroy(plan));

    *this = std::move(result);
}


struct ScaleFunctor {
    float scale;

    ScaleFunctor(float s) : scale(s) {}

    __device__ cufftComplex operator()(cufftComplex c) const {
        return make_cuComplex(c.x * scale, c.y * scale);
    }
};

void CudaMatrix::ifft(cudaStream_t _stream, cufftHandle &plan) const {
    checkCufftErrors(cufftSetStream(plan, _stream));
    checkCufftErrors(cufftExecC2C(plan, data, data, CUFFT_INVERSE));
//    float scale = 1.0f / ncols;
//    thrust::device_ptr<cufftComplex> thrust_data(data);
//    auto exec_policy = thrust::cuda::par.on(_stream);
//    thrust::transform(exec_policy, thrust_data, thrust_data + nrows * ncols, thrust_data, ScaleFunctor(scale));
}



// Kernel to extract segments from each row
__global__ void
extractSegmentKernel(cufftComplex *input, cufftComplex *output, int nrows, int ncols, int startInd, int rangeNumber) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        for (int j = 0; j < rangeNumber; ++j) {
            int inputIdx = row * ncols + startInd + j;
            int outputIdx = row * rangeNumber + j;
            output[outputIdx] = input[inputIdx];
        }
    }
}

CudaMatrix CudaMatrix::extractSegment(int startInd, int rangeNumber) const {
    if (startInd < 0 || startInd + rangeNumber > ncols) {
        throw std::runtime_error("Index out of bounds.");
    }

    CudaMatrix result(nrows, rangeNumber);

    // Launch kernel to extract segments
    int blockSize = 256;
    int gridSize = (nrows + blockSize - 1) / blockSize;
    extractSegmentKernel<<<gridSize, blockSize>>>(data, result.data, nrows, ncols, startInd, rangeNumber);

    // Check for errors
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return result;
}

void CudaMatrix::writeMatTxt(const std::string &filePath) const {
    std::ofstream outfile(filePath);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file " << filePath << " for writing!" << std::endl;
        return;
    }

    outfile << nrows << " " << ncols << std::endl;

        std::vector<cufftComplex> hostData(nrows * ncols);
    copyToHost(hostData);

    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            outfile << hostData[i * ncols + j].x << " " << hostData[i * ncols + j].y << endl;
        }
        outfile << std::endl;
    }

    outfile.close();
}

//__global__ void cfarKernel(const cufftComplex* data, cufftComplex* cfar_signal, int nrows, int ncols, double alpha,
//                           int numGuardCells, int numRefCells, int leftBoundary, int rightBoundary) {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (row < nrows && col < ncols && col >= leftBoundary && col < rightBoundary) {
//        int total_training_cells = numGuardCells + numRefCells;
//
//        double noise_level = 0.0;
//        int num_ref_cells;
//
//        if (col < total_training_cells) {
//            for (int i = col + numGuardCells + 1; i <= col + numGuardCells + numRefCells; ++i) {
//                if (i < ncols) {
//                    noise_level += data[row * ncols + i].x;
//                }
//            }
//            num_ref_cells = numRefCells;
//        } else if (col >= ncols - total_training_cells) {
//            for (int i = col - numRefCells - numGuardCells; i < col - numGuardCells; ++i) {
//                if (i >= 0) {
//                    noise_level += data[row * ncols + i].x;
//                }
//            }
//            num_ref_cells = numRefCells;
//        } else {
//            for (int i = col - total_training_cells; i < col - total_training_cells + numRefCells; ++i) {
//                noise_level += data[row * ncols + i].x;
//            }
//            for (int i = col + numGuardCells + 1; i <= col + numGuardCells + numRefCells; ++i) {
//                noise_level += data[row * ncols + i].x;
//            }
//            num_ref_cells = numRefCells * 2;
//        }
//
//        double threshold = alpha * noise_level / num_ref_cells;
//
//        if (data[row * ncols + col].x > threshold) {
//            cfar_signal[row * ncols + col].x = sqrt(data[row * ncols + col].x);
//        }
//    }
//}
//
//void CudaMatrix::cfar(CudaMatrix &output, cudaStream_t _stream, double Pfa, int numGuardCells, int numRefCells,
//                      int leftBoundary, int rightBoundary) const {
//    double alpha = (numRefCells * 2 * (pow(Pfa, -1.0 / (numRefCells * 2)) - 1));
//
//    // Compute the absolute values
//    this->abs(_stream);
//
//    // Compute the squared absolute values
//    this->elementWiseSquare(_stream);
//
//    // Configure the CUDA kernel launch parameters
//    dim3 blockDim(16, 16);
//    dim3 gridDim((ncols + blockDim.x - 1) / blockDim.x, (nrows + blockDim.y - 1) / blockDim.y);
//
//    // Launch the CFAR kernel
//    cfarKernel<<<gridDim, blockDim, 0, _stream>>>(data, output.data, nrows, ncols, alpha, numGuardCells, numRefCells, leftBoundary, rightBoundary);
//}

__global__ void cfarKernel(const cufftComplex* data, cufftComplex* cfar_signal, int nrows, int ncols,
                           double alpha, int numGuardCells, int numRefCells, int leftBoundary, int rightBoundary) {

    int row = blockIdx.y;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int totalTrainingCells = numGuardCells + numRefCells;
    int col_start = max(thread_id * CFAR_LENGTH, leftBoundary + totalTrainingCells);
    int col_end = min(col_start + CFAR_LENGTH, rightBoundary - totalTrainingCells);

    if (col_start >= ncols || row >= nrows) return;

    double noiseLevel = 0.0;
    for (int i = col_start; i < col_end; ++i) {
        if (i == col_start) {
            for (int j = i - totalTrainingCells; j < i - numGuardCells; ++j) {
                noiseLevel += data[row * ncols + j].x;
            }
            for (int j = i + numGuardCells + 1; j <= i + totalTrainingCells; ++j) {
                noiseLevel += data[row * ncols + j].x;
            }
        }
        else {
            noiseLevel += data[row * ncols + i + totalTrainingCells].x;
            noiseLevel += data[row * ncols + i - numGuardCells - 1].x;
            noiseLevel -= data[row * ncols + i + numGuardCells].x;
            noiseLevel -= data[row * ncols + (i - totalTrainingCells - 1)].x;
        }

        double threshold = alpha * noiseLevel / (2 * numRefCells);
        cfar_signal[row * ncols + i].x = (data[row * ncols + i].x > threshold) ? sqrt(data[row * ncols + i].x) : 0.0;
        cfar_signal[row * ncols + i].y = 0.0;
    }
}

void CudaMatrix::cfar(CudaMatrix &output, cudaStream_t _stream, double Pfa, int numGuardCells, int numRefCells,
                      int leftBoundary, int rightBoundary) const {
    double alpha = (numRefCells * 2 * (pow(Pfa, -1.0 / (numRefCells * 2)) - 1));

    // Compute the absolute values
    this->abs(_stream);

    // Compute the squared absolute values
    this->elementWiseSquare(_stream);

    // Configure the CUDA kernel launch parameters
    int colsPerThread = CFAR_LENGTH; // 每个线程处理的列数
    int threadsPerBlock = NFFT / colsPerThread; // 每个线程块中的线程数
    int blocksPerRow = (ncols + colsPerThread - 1) / colsPerThread / threadsPerBlock; // 每行的线程块数
    dim3 blockDim(threadsPerBlock, 1); // 线程块大小：1 行 x 32 列
    dim3 gridDim(blocksPerRow, nrows); // 网格大小：每行 block 数 x 总行数

    // Launch the CFAR kernel
    cfarKernel<<<gridDim, blockDim, 0, _stream>>>(data, output.data, nrows, ncols, alpha, numGuardCells, numRefCells, leftBoundary, rightBoundary);
}


// 现在是对实部选大，而不是abs
__global__ void maxKernelDim1(cufftComplex *data, cufftComplex *maxValues, int nrows, int ncols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ind;
    if (col < ncols) {
        float maxVal = data[col].x;
        for (int row = 1; row < nrows; ++row) {
            ind = row * ncols + col;
//            if (sqrt(val.x * val.x + val.y * val.y) > sqrt(maxVal.x * maxVal.x + maxVal.y * maxVal.y)) {
            if (data[ind].x > maxVal) {
                maxVal = data[ind].x;
            }
        }
        maxValues[col].x = maxVal;
        maxValues[col].y = 0;
    }
}

__global__ void maxKernelDim2(cufftComplex *data, cufftComplex *maxValues, int nrows, int ncols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        cufftComplex maxVal = data[row * ncols];
        for (int col = 1; col < ncols; ++col) {
            cufftComplex val = data[row * ncols + col];
            if (sqrt(val.x * val.x + val.y * val.y) > sqrt(maxVal.x * maxVal.x + maxVal.y * maxVal.y)) {
                maxVal = val;
            }
        }
        maxValues[row] = maxVal;
    }
}

void CudaMatrix::max(CudaMatrix &output, cudaStream_t _stream, int dim) {
    dim3 blockDim(256);
    dim3 gridDim((ncols + blockDim.x - 1) / blockDim.x);

    maxKernelDim1<<<gridDim, blockDim, 0, _stream>>>(data, output.data, nrows, ncols);
}

__global__ void elementWiseSquareKernel(cufftComplex *idata, cufftComplex *odata, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        cufftComplex value = idata[idx];
        odata[idx].x = value.x * value.x - value.y * value.y;
        odata[idx].y = 2 * value.x * value.y;
    }
}

void CudaMatrix::elementWiseSquare(cudaStream_t _stream) const {
    int blockSize = 256;
    int size = ncols * nrows;
    int gridSize = (size + blockSize - 1) / blockSize;

    elementWiseSquareKernel<<<gridSize, blockSize, 0, _stream>>>(data, data, size);
}

__global__ void absCufftComplexKernel(cufftComplex *idata, cufftComplex *odata, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        odata[idx].x = cuCabsf(idata[idx]);
        odata[idx].y = 0;
    }
}


void CudaMatrix::abs(cudaStream_t _stream) const {
    int blockSize = 256;
    int size = ncols * nrows;
    int gridSize = (size + blockSize - 1) / blockSize;
    absCufftComplexKernel<<<gridSize, blockSize, 0, _stream>>>(data, data, size);
}

// Allocate memory on the device
void CudaMatrix::allocateMemory() {
    if (nrows > 0 && ncols > 0) {
        checkCudaErrors(cudaMalloc(&data, sizeof(cufftComplex) * nrows * ncols));
    }
}

// Deallocate memory on the device
void CudaMatrix::deallocateMemory() {
    if (data && !initFromDevice) {
        checkCudaErrors(cudaFree(data));
        data = nullptr;
    }
}

void CudaMatrix::setSize(int _nrows, int _ncols) {
    if (data != nullptr){
        deallocateMemory();
        data = nullptr;
    }
    nrows = _nrows;
    ncols = _ncols;

    if (nrows > 0 && ncols > 0) {
        checkCudaErrors(cudaMalloc(&data, sizeof(cufftComplex) * nrows * ncols));
    }
}

