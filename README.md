# Projeto_CED

This repository contains implementations of the **Canny Edge Detection Algorithm**, optimized using various approaches to demonstrate the efficiency of sequential, parallel, and GPU-based computations. The project is structured into multiple branches, each corresponding to a specific implementation or aspect of the algorithm.

---

## 📂 Branches Overview

Below is the detailed breakdown of each branch and its content:

### 1. **Main Branch**
- **Description**: The **Main Branch** consolidates all final versions of the project, including the Sequential, OpenMP, and GPU implementations, alongside all required scripts and outputs. It serves as the primary entry point for users to access the complete project in its final state.
- **Content**:
  - **Source Codes**:
    - `canny_sequential.c` – Final sequential implementation with robust debugging and gradient computation.
    - `canny_openmp.c` – Final parallelized implementation using OpenMP for multi-core CPU optimization.
    - `canny_gpu.cu` – Final CUDA implementation optimized for NVIDIA GPUs.
  - **Scripts**:
    - `run_benchmark.sh` – Automates benchmarking across all implementations.
    - `generate_csv.sh` – Generates CSV files with benchmarking results.
  - **Outputs**:
    - `benchmark_results.csv` – Detailed results of execution times and speedup for multiple implementations.
    - `benchmark_results.txt` – A human-readable benchmarking summary.
  - **Build System**:
    - `CMakeLists.txt` – Unified build script for compiling all implementations using CMake.
    - `build` folder – Contains compiled binaries and ready-to-run executables.
  - **Documentation**:
    - `README.md` – Overview of the repository and usage instructions.
    - `LICENSE` – Licensing information.
- **Purpose**: To provide a fully consolidated, polished version of the project. This branch simplifies access to all functionalities, serving as the central repository for the project's final deliverables and results.

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

### Clone the repository
```bash
git clone https://github.com/g0ndex/Projeto_CED.git
cd Projeto_CED
