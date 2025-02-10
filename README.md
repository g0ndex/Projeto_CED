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
- **Description**: The base implementation of the Canny Edge Detection algorithm using a sequential approach.
- **Content**:
  - `canny_sequential.c` – Sequential implementation with gradient computation and non-maximum suppression.
- **Purpose**: Serves as the baseline for performance comparisons with parallel and GPU versions.

### 3. **Versão-Supressão-Não-Máxima**
- **Description**: Focuses on implementing non-maximum suppression as a crucial part of the edge detection process.
- **Content**:
  - Updated `canny_sequential.c` with non-maximum suppression logic.
- **Purpose**: To refine the edge detection algorithm's accuracy and compare its impact on performance.

### 4. **Versão-Threshold**
- **Description**: Introduces thresholding for edge linking as part of the Canny algorithm.
- **Content**:
  - `canny_sequential.c` – Enhanced with thresholding to produce binary edge maps.
- **Purpose**: To study the effect of thresholding parameters on edge detection results.

### 5. **Versão-OpenMP**
- **Description**: Implements parallelization of the Canny Edge Detection algorithm using OpenMP.
- **Content**:
  - `canny_openmp.c` – Parallelized implementation leveraging multi-core CPU optimizations.
- **Key Optimizations**:
  - Workload divided across threads using OpenMP `#pragma` directives.
  - Efficient memory management to reduce contention.
- **Purpose**: To explore the speedup achieved through CPU parallelization.

### 6. **Versão-GPU**
- **Description**: Accelerates the Canny Edge Detection algorithm using CUDA for GPU processing.
- **Content**:
  - `canny_gpu.cu` – CUDA implementation optimized for NVIDIA GPUs.
  - Tested on NVIDIA A100-SXM4 GPUs with CUDA 12.4.
- **Purpose**: Highlights the computational advantages of GPUs for large-scale image processing.

### 7. **Versão-Medição-Tempos**
- **Description**: Focuses on benchmarking the performance of different implementations.
- **Content**:
  - `run_benchmark.sh` – Automates the benchmarking process across multiple implementations and matrix sizes.
  - `benchmark_results.csv` – Records execution times and speedups for all implementations.
- **Purpose**: Provides a comprehensive evaluation of the performance trade-offs across various implementations.

---

## 🛠️ How to Use This Repository

### Clone the repository
```bash
git clone https://github.com/g0ndex/Projeto_CED.git
cd Projeto_CED
