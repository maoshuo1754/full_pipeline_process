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
#        CudaMatrix.cu
        utils.cu
        ThreadPool.cu
        main.cpp
        NRx/sendVideo.cpp
        NRx/plot.cpp
        Config.cpp
        DataSource.cpp
        WaveGroupProcessor.cu
        kelnels.cu
)

file(GLOB MYSRCS "${PROJECT_SOURCE_DIR}/NRx/include/*.cpp")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    file(GLOB DLLFILES "${PROJECT_SOURCE_DIR}/NRx/dlls/debug/*.dll")
else ()
    file(GLOB DLLFILES "${PROJECT_SOURCE_DIR}/NRx/dlls/release/*.dll")
endif ()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${MYSRCS})

find_package(CUDAToolkit REQUIRED)

target_link_libraries(${PROJECT_NAME}
        pthread
        rt
        CUDA::cublas
        CUDA::cufft
        ${DLLFILES}
)

# Include CUDA Toolkit headers
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ./NRx
        ./NRx/include
)

