#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Dimensões da matriz
#define LARGURA 5
#define ALTURA 5

// Kernel para calcular os gradientes e direções
__global__ void calcular_gradientes(double *input, double *gradiente, double *direcao, int largura, int altura) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && j > 0 && i < altura - 1 && j < largura - 1) {
        double gx = input[(i - 1) * largura + (j + 1)] - input[(i - 1) * largura + (j - 1)] +
                    2 * input[i * largura + (j + 1)] - 2 * input[i * largura + (j - 1)] +
                    input[(i + 1) * largura + (j + 1)] - input[(i + 1) * largura + (j - 1)];
        double gy = input[(i - 1) * largura + (j - 1)] + 2 * input[(i - 1) * largura + j] + input[(i - 1) * largura + (j + 1)] -
                    input[(i + 1) * largura + (j - 1)] - 2 * input[(i + 1) * largura + j] - input[(i + 1) * largura + (j + 1)];
        
        gradiente[i * largura + j] = sqrt(gx * gx + gy * gy);
        direcao[i * largura + j] = atan2(gy, gx) * 180 / M_PI;
    }
}

// Kernel para aplicar thresholds (simplificado)
__global__ void aplicar_thresholds(double *gradiente, double *output, double low, double high, int largura, int altura) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < altura && j < largura) {
        double valor = gradiente[i * largura + j];
        if (valor >= high) {
            output[i * largura + j] = 255.0;
        } else if (valor >= low) {
            output[i * largura + j] = 128.0;
        } else {
            output[i * largura + j] = 0.0;
        }
    }
}

// Função principal
int main() {
    // Define a matriz de entrada
    double input[ALTURA][LARGURA] = {
        {0, 50, 100, 150, 200},
        {50, 100, 150, 200, 250},
        {100, 150, 200, 250, 300},
        {150, 200, 250, 300, 350},
        {200, 250, 300, 350, 400}
    };
    
    double *d_input, *d_gradiente, *d_direcao, *d_output;
    double gradiente[ALTURA][LARGURA] = {0};
    double direcao[ALTURA][LARGURA] = {0};
    double output[ALTURA][LARGURA] = {0};

    size_t size = ALTURA * LARGURA * sizeof(double);

    // Alocar memória na GPU
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_gradiente, size);
    cudaMalloc(&d_direcao, size);
    cudaMalloc(&d_output, size);

    // Copiar dados para a GPU
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Definir configuração de grid e blocos
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((LARGURA + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ALTURA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Executar kernel para gradientes e direções
    calcular_gradientes<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_gradiente, d_direcao, LARGURA, ALTURA);
    cudaMemcpy(gradiente, d_gradiente, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(direcao, d_direcao, size, cudaMemcpyDeviceToHost);

    // Executar kernel para aplicar thresholds
    aplicar_thresholds<<<blocksPerGrid, threadsPerBlock>>>(d_gradiente, d_output, 100.0, 200.0, LARGURA, ALTURA);
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Exibir resultados
    printf("Matriz de Entrada:\n");
    for (int i = 0; i < ALTURA; i++) {
        for (int j = 0; j < LARGURA; j++) {
            printf("%.2f ", input[i][j]);
        }
        printf("\n");
    }

    printf("\nGradientes:\n");
    for (int i = 0; i < ALTURA; i++) {
        for (int j = 0; j < LARGURA; j++) {
            printf("%.2f ", gradiente[i][j]);
        }
        printf("\n");
    }

    printf("\nMatriz Final com Thresholds Aplicados:\n");
    for (int i = 0; i < ALTURA; i++) {
        for (int j = 0; j < LARGURA; j++) {
            printf("%.2f ", output[i][j]);
        }
        printf("\n");
    }

    // Liberar memória
    cudaFree(d_input);
    cudaFree(d_gradiente);
    cudaFree(d_direcao);
    cudaFree(d_output);

    return 0;
}
