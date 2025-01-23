#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void kernelCanny(double *matriz, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < dim && idy < dim) {
        int index = idy * dim + idx;
        matriz[index] += 1.0;  // Simulação de cálculo
    }
}

void executarCannyGPU(int dim) {
    printf("Executando Canny GPU com matriz de dimensão %dx%d...\n", dim, dim);

    double *host_matriz = (double *)malloc(dim * dim * sizeof(double));
    for (int i = 0; i < dim * dim; i++) {
        host_matriz[i] = (double)i;
    }

    double *device_matriz;
    cudaMalloc(&device_matriz, dim * dim * sizeof(double));
    cudaMemcpy(device_matriz, host_matriz, dim * dim * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (dim + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernelCanny<<<blocksPerGrid, threadsPerBlock>>>(device_matriz, dim);
    cudaDeviceSynchronize();

    cudaMemcpy(host_matriz, device_matriz, dim * dim * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_matriz);
    free(host_matriz);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <dimensao_da_matriz>\n", argv[0]);
        return 1;
    }

    int dim = atoi(argv[1]);
    if (dim <= 0) {
        printf("Dimensão inválida. Deve ser um número inteiro positivo.\n");
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    executarCannyGPU(dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Tempo de execução (GPU): %.6f segundos\n", elapsed_time / 1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
