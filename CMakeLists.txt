cmake_minimum_required(VERSION 3.27)
project(reader CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS_DEBUG "-G -g")

# Enable CUDA language
enable_language(CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 80)

set(SOURCE_FILES
        queue.h
        CudaMatrix.cu
        CudaMatrix.h
        utils.cu
        utils.h
        myThreadPool.h
        myThreadPool.cu
        main.cpp
        NRx/sendVideo.cpp
        NRx/SendVideo.h
        Config.h
        Config.cpp
)

add_executable(reader ${SOURCE_FILES})

# Find and link CUDA Toolkit libraries
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json 3.7.3 REQUIRED)
#find_package(OpenMP REQUIRED)

target_link_libraries(reader
        pthread
        rt
        CUDA::cublas
        CUDA::cufft
        nlohmann_json::nlohmann_json
#        OpenMP::OpenMP_CXX
)

# Include CUDA Toolkit headers
target_include_directories(reader PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ./NRx)

## Set CUDA separable compilation
#set_target_properties(reader PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON
#) 

