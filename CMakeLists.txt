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

# 查找 matio 库
find_library(MATIO_LIBRARY NAMES matio)
find_path(MATIO_INCLUDE_DIR NAMES matio.h)

# 检查是否找到 matio
if (NOT MATIO_LIBRARY OR NOT MATIO_INCLUDE_DIR)
    message(FATAL_ERROR "matio library not found. Please install libmatio-dev.")
endif()

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
        ${MATIO_LIBRARY}
)

# Include CUDA Toolkit headers
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ./NRx
        ./NRx/include
        ${MATIO_INCLUDE_DIR}
)


# 可选：打印IPP信息
message(STATUS "IPP include路径: ${IPP_ROOT}/include")

find_library(HDF5_LIBRARY NAMES hdf5)
if (HDF5_LIBRARY)
    target_link_libraries(matio_example PRIVATE ${HDF5_LIBRARY})
endif()
