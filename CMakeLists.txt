cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME  "reader")

set(CMAKE_PREFIX_PATH "/usr/local/cuda")

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 80)

project(${PROJECT_NAME} CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS_DEBUG "-G -g")

# Enable CUDA language
enable_language(CUDA)
set(SOURCE_FILES
        utils.cu
        ThreadPool.cu
        main.cpp
        SendVideo.cpp
        NRx/plot.cpp
        Config.cpp
        DataSource.cpp
        WaveGroupProcessor.cu
        kelnels.cu
        SendVideo.h
        GpuQueueManager.cu
)

file(GLOB MYSRCS "${PROJECT_SOURCE_DIR}/NRx/include/*.cpp")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    file(GLOB DLLFILES "${PROJECT_SOURCE_DIR}/NRx/dlls/debug/*.dll")
else ()
    file(GLOB DLLFILES "${PROJECT_SOURCE_DIR}/NRx/dlls/release/*.dll")
endif ()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${MYSRCS})

find_package(CUDAToolkit REQUIRED)

 set(IPP_ROOT "/home/csic724/intel/oneapi/ipp/2021.12")
 include_directories("${IPP_ROOT}/include")

target_link_libraries(${PROJECT_NAME} PRIVATE
        pthread
        rt
        CUDA::cublas
        CUDA::cufft
        ${DLLFILES}
        ${IPP_ROOT}/lib/intel64/libippcore.so
        ${IPP_ROOT}/lib/intel64/libipps.so
        ${IPP_ROOT}/lib/intel64/libippi.so
)

# Include CUDA Toolkit headers
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ./NRx
        ./NRx/include
)

# 可选：打印IPP信息
message(STATUS "IPP include路径: ${IPP_ROOT}/include")
