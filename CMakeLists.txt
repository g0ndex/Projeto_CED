cmake_minimum_required(VERSION 3.10)

# Project Name
project(CannyEdgeDetection_Goncalorente)

# Set C Standard and CXX Standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add Executables
add_executable(canny_sequential canny_sequential.c)
add_executable(canny_openmp canny_openmp.c)
add_executable(canny_gpu canny_gpu.cu)

# Enable CUDA for GPU version
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_50)

# Link Libraries
target_link_libraries(canny_sequential m)
target_link_libraries(canny_openmp m OpenMP::OpenMP_C)
target_link_libraries(canny_gpu m)

# Set Output Directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
