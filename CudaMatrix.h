#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <vector>
#include <complex>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <utility>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

class CudaMatrix {
private:
    int nrows, ncols;
    cufftComplex* data;

public:
    CudaMatrix() : nrows(0), ncols(0), data(nullptr) {}
    CudaMatrix(int rows, int cols);
    CudaMatrix(const std::vector<std::vector<cufftComplex>>& hostData);
    CudaMatrix(int rows, int cols, std::vector<cufftComplex> hostData);
    CudaMatrix(int rows, int cols, cufftComplex* hostData);
    CudaMatrix(const CudaMatrix& other); // Copy constructor
    CudaMatrix(CudaMatrix&& other) noexcept; // Move constructor
    ~CudaMatrix();

    size_t size() const { return nrows * ncols; }
    size_t getRows() const { return nrows; }
    size_t getCols() const { return ncols; }
    cufftComplex* getData() const { return data; }
    void printShape() const;
    void setSize(int _nrows, int _ncols);
    void setElement(int x, int y, cufftComplex value);

    void copyFromHost(const std::vector<cufftComplex>& hostData);
    void copyFromHost(int rows, int cols, const cufftComplex* hostData);
    void copyToHost(std::vector<cufftComplex>& hostData) const;
    std::vector<std::vector<cufftComplex>> to2DVector() const;
    void fillWithRandomValues();
    void print() const;
    void print(int row) const;
    void printLargerThan0() const;

    CudaMatrix max(int dim = 1);

    CudaMatrix T(bool inplace = true);
    CudaMatrix operator*(const CudaMatrix& other) const;
    CudaMatrix& operator=(const CudaMatrix& other);     // Copy assignment
    CudaMatrix& operator=(CudaMatrix&& other) noexcept; // Move assignment
    bool operator==(const CudaMatrix& other) const;
    CudaMatrix elementWiseMul(const CudaMatrix& other, bool inplace = true) const;
    CudaMatrix elementWiseSquare(bool inplace = true) const;
    CudaMatrix abs(bool inplace = true) const;

    void fft() const;
    void fft_by_col();
    void ifft() const;
    void fft_N(int nPoints);

    CudaMatrix extractSegment(int startInd, int rangeNumber) const;
    CudaMatrix cfar(double Pfa, int numGuardCells, int numRefCells) const;
    void writeMatTxt(const std::string &filePath) const;

private:
    void allocateMemory();
    void deallocateMemory();
    static void checkCudaErrors(cudaError_t result);
    static void checkCublasErrors(cublasStatus_t result);
    static void checkCufftErrors(cufftResult result);
};

#endif // CUDA_MATRIX_H
