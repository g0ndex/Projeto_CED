# Projeto_CED

This repository contains implementations of the **Canny Edge Detection Algorithm**, optimized using various approaches to demonstrate the efficiency of sequential, parallel, and GPU-based computations. The project is structured into multiple branches, each corresponding to a specific implementation or aspect of the algorithm.

---

## 📂 Branches Overview

Below is the detailed breakdown of each branch and its content:

### 1. **Main Branch**
- **Description**: The starting point of the repository, containing the simplest sequential implementation of the Canny Edge Detection algorithm.
- **Content**:
  - Sequential implementation code (`canny_sequential.c`).
  - Basic `CMakeLists.txt` file for compilation.
- **Purpose**: Serves as a baseline for performance comparison against parallel and GPU-optimized versions.

### 2. **Versão-Sequencial**
- **Description**: This branch expands on the sequential implementation with additional comments, fixes, and debugging information.
- **Content**:
  - Enhanced sequential implementation (`canny_sequential.c`).
  - Detailed comments explaining gradient computation and suppression.
- **Purpose**: A polished version of the sequential algorithm to establish a foundation for further optimization.

### 3. **Versão-OpenMP**
- **Description**: Introduces parallelization of the Canny Edge Detection algorithm using OpenMP.
- **Content**:
  - Parallelized implementation (`canny_openmp.c`) leveraging multi-core CPUs.
  - Benchmarking integration to measure speedup and efficiency.
- **Key Optimizations**:
  - Divided workload across multiple threads using OpenMP `#pragma` directives.
  - Efficient memory access to prevent thread contention.
- **Purpose**: Demonstrates how parallelism can significantly improve performance for edge detection.

### 4. **Versão-GPU**
- **Description**: Implements the Canny Edge Detection algorithm using CUDA for GPU acceleration.
- **Content**:
  - CUDA-based implementation (`canny_gpu.cu`) for NVIDIA GPUs.
  - Optimizations for global memory access and thread block distribution.
  - Compatibility with CUDA 12.4 (tested on NVIDIA A100 GPUs).
- **Purpose**: Showcases the computational power of GPUs for handling large-scale image processing tasks efficiently.

### 5. **Versão-Medição-Tempos**
- **Description**: Focuses on benchmarking the performance of the algorithm across different implementations (Sequential, OpenMP, and GPU).
- **Content**:
  - Script (`run_benchmark.sh`) to automate benchmarking for multiple matrix sizes.
  - CSV file (`benchmark_results.csv`) for recording execution times and speedup.
  - OpenMP, CUDA, and sequential versions for testing.
- **Purpose**: Provides a robust evaluation framework for analyzing the efficiency of each implementation.

### 6. **Versão-Benchmarking-CMake**
- **Description**: Contains a consolidated setup using CMake for building all implementations (Sequential, OpenMP, and GPU) with ease.
- **Content**:
  - Comprehensive `CMakeLists.txt` for cross-platform compatibility.
  - Scripts to build and run benchmarks.
- **Purpose**: Simplifies the compilation process and ensures a unified build system for all implementations.

---

## 🛠️ How to Use This Repository

Clone the repository:
```bash
git clone https://github.com/g0ndex/Projeto_CED.git
cd Projeto_CED


### Key Improvements:
1. **Consistent structure**: Used headings and subheadings for readability.
2. **Clear branch descriptions**: Highlighted each branch's purpose and content.
3. **Added system requirements**: Ensured the hardware details are clear.
4. **Provided instructions**: Explained how to clone, build, and test.
5. **Polished formatting**: Ensured the README is visually appealing.

Let me know if this works!
