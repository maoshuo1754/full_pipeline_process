# 高速数据流GPU处理系统

## 项目简介
本项目是一个基于C++和CUDA的高性能数据读取和数据处理系统，专注于从PCIe设备高效读取和处理大规模二进制数据，支持多线程处理和GPU加速，
项目以来CUDA Toolkit和其他第三方库 (如nlohmann_json)。

## 环境要求
### 系统要求

- Ubuntu 20.04或更高版本
- 支持CUDA的GPU和正确安装的NVIDIA驱动程序

### 软件依赖

- CMake: version >= 3.27
- CUDA Toolkit: version >= 11.8
- GCC/G++: version >= 9.4
- nlohmann_json: version >= 3.7.3

### 文件目录
项目的文件目录结构如下：

```
.
├── queue.h             # 共享内存管理队列
├── CudaMatrix.h        # GPU矩阵处理
├── CudaMatrix.cu
├── utils.h             # 工具函数
├── utils.cu 
├── ThreadPool.h      # 线程池
├── ThreadPool.cu
├── main.h              # 主函数
├── main.cpp
├── NRx/
│  ├── sendVideo.h    # 检测视频发送
│  ├── SendVideo.cpp
│  ├── plot.h           # 点迹凝聚文件
│  ├── plot.cpp         
│  ├── include/         # 动态加载的其他源文件
│  ├── dlls/
│     ├── debug/        # Debug 模式下的 DLL
│     ├── release/      # Release 模式下的 DLL
├── config.json         # 配置文件
├── Config.h            # 配置管理
├── Config.cpp
├── CMakeLists.txt      # cmake构建配置
```

## 配置文件
项目使用JSON配置文件，程序会对配置文件进行监控，支持动态配置重载。


## 作者姓名
ms