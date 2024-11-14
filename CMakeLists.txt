cmake_minimum_required(VERSION 3.27)
project(reader CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 80)

set(SOURCE_FILES
        queue.h
        CudaMatrix.cu
        CudaMatrix.h
#        reader.cu
#        reader.h
        myThreadPool.h
        myThreadPool.cu
        main.cu
)

add_executable(reader ${SOURCE_FILES})

# Find and link CUDA Toolkit libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(reader
        pthread
        rt
        CUDA::cublas
        CUDA::cufft
)

# Include CUDA Toolkit headers
target_include_directories(reader PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

## Set CUDA separable compilation
#set_target_properties(reader PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON
#)

