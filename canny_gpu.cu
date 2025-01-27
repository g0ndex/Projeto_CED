#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

// Kernel for gradient calculation
__global__ void calculate_gradients(double *input, double *gradient, double *direction, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        double gx = input[(row - 1) * width + (col + 1)] + 2 * input[row * width + (col + 1)] + input[(row + 1) * width + (col + 1)]
                  - input[(row - 1) * width + (col - 1)] - 2 * input[row * width + (col - 1)] - input[(row + 1) * width + (col - 1)];
        double gy = input[(row - 1) * width + (col - 1)] + 2 * input[(row - 1) * width + col] + input[(row - 1) * width + (col + 1)]
                  - input[(row + 1) * width + (col - 1)] - 2 * input[(row + 1) * width + col] - input[(row + 1) * width + (col + 1)];
        gradient[row * width + col] = sqrt(gx * gx + gy * gy);
        direction[row * width + col] = atan2(gy, gx) * 180 / PI;
    }
}

// CUDA error-checking function
void check_cuda_error(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int matrix_size = size * size * sizeof(double);

    // Allocate and initialize host memory
    double *h_input = (double *)malloc(matrix_size);
    double *h_gradient = (double *)malloc(matrix_size);
    double *h_direction = (double *)malloc(matrix_size);

    for (int i = 0; i < size * size; i++) {
        h_input[i] = i % 256;
    }

    // Allocate device memory
    double *d_input, *d_gradient, *d_direction;
    check_cuda_error(cudaMalloc(&d_input, matrix_size), "Failed to allocate device memory for input");
    check_cuda_error(cudaMalloc(&d_gradient, matrix_size), "Failed to allocate device memory for gradient");
    check_cuda_error(cudaMalloc(&d_direction, matrix_size), "Failed to allocate device memory for direction");

    // GPU timing with memory transfers
    cudaEvent_t start, stop;
    check_cuda_error(cudaEventCreate(&start), "Failed to create CUDA event for start");
    check_cuda_error(cudaEventCreate(&stop), "Failed to create CUDA event for stop");

    // Start timing
    check_cuda_error(cudaEventRecord(start), "Failed to record start event");

    // Copy input to device
    check_cuda_error(cudaMemcpy(d_input, h_input, matrix_size, cudaMemcpyHostToDevice), "Failed to copy input data to device");

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);
    calculate_gradients<<<gridDim, blockDim>>>(d_input, d_gradient, d_direction, size, size);
    check_cuda_error(cudaGetLastError(), "Kernel launch failed");

    // Copy results back to host
    check_cuda_error(cudaMemcpy(h_gradient, d_gradient, matrix_size, cudaMemcpyDeviceToHost), "Failed to copy gradient data back to host");
    check_cuda_error(cudaMemcpy(h_direction, d_direction, matrix_size, cudaMemcpyDeviceToHost), "Failed to copy direction data back to host");

    // Stop timing
    check_cuda_error(cudaEventRecord(stop), "Failed to record stop event");
    check_cuda_error(cudaEventSynchronize(stop), "Failed to synchronize stop event");

    // Calculate elapsed time
    float milliseconds = 0;
    check_cuda_error(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to compute elapsed time");

    printf("Execution completed on GPU for matrix size %dx%d\n", size, size);
    printf("Total Execution Time (GPU): %.6f seconds\n", milliseconds / 1000.0);

    // Cleanup
    free(h_input);
    free(h_gradient);
    free(h_direction);
    cudaFree(d_input);
    cudaFree(d_gradient);
    cudaFree(d_direction);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
