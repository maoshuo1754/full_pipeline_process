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
    bool initFromDevice;

public:
    CudaMatrix() : nrows(0), ncols(0), data(nullptr) {}
    CudaMatrix(int rows, int cols);
    CudaMatrix(int rows, int cols, cudaStream_t& stream);
    CudaMatrix(const std::vector<std::vector<cufftComplex>>& hostData);
    CudaMatrix(int rows, int cols, std::vector<cufftComplex> hostData);
    CudaMatrix(int rows, int cols, cufftComplex* hostData);
    CudaMatrix(int rows, int cols, cufftComplex *hostData, bool deviceFlag);

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
    void copyFromHost(cudaStream_t _stream, int rows, int cols, const cufftComplex *hostData);
    void copyToHost(std::vector<cufftComplex>& hostData) const;
    void copyToHost(cufftComplex *hostData) const;
    std::vector<std::vector<cufftComplex>> to2DVector() const;
    void fillWithRandomValues();
    void print() const;
    void print(int row) const;
    void print(int row, float des) const;
    void printLargerThan0() const;

    void max(CudaMatrix &output, cudaStream_t _stream, int dim=1);

    CudaMatrix T(bool inplace = true);
    CudaMatrix operator*(const CudaMatrix& other) const;
    CudaMatrix& operator=(const CudaMatrix& other);     // Copy assignment
    CudaMatrix& operator=(CudaMatrix&& other) noexcept; // Move assignment
    bool operator==(const CudaMatrix& other) const;
    void elementWiseMul(const CudaMatrix &other, cudaStream_t _stream) const;
    void rowWiseMul(const CudaMatrix &other, cudaStream_t _stream);
    void elementWiseSquare(cudaStream_t _stream) const;
    void abs(cudaStream_t _stream) const;

    void fft(cufftHandle &plan) const;
    void fft_by_col(cufftHandle &plan);
    void ifft(cudaStream_t _stream, cufftHandle &plan) const;
    void fft_N(int nPoints);

    CudaMatrix extractSegment(int startInd, int rangeNumber) const;
    void cfar(CudaMatrix &output, cudaStream_t stream, double Pfa, int numGuardCells, int numRefCells,
              int leftBoundary, int rightBoundary) const;
    void writeMatTxt(const std::string &filePath) const;
    void scale(cudaStream_t _stream, float _scale);

    void MTI(cudaStream_t _stream, int numCancellerPulses);

private:
    void allocateMemory();
    void deallocateMemory();



};

#endif // CUDA_MATRIX_H
