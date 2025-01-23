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
        double gy = input[(i - 1) * largura + (j - 1)] + 2 * input[(i - 1) * largura + j] +
                    input[(i - 1) * largura + (j + 1)] - input[(i + 1) * largura + (j - 1)] -
                    2 * input[(i + 1) * largura + j] - input[(i + 1) * largura + (j + 1)];

        gradiente[i * largura + j] = sqrt(gx * gx + gy * gy);
        direcao[i * largura + j] = atan2(gy, gx) * 180 / M_PI;
    }
}

// Kernel para aplicar supressão não máxima
__global__ void supressao_nao_maxima(double *gradiente, double *direcao, double *output, int largura, int altura) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && j > 0 && i < altura - 1 && j < largura - 1) {
        double angulo = direcao[i * largura + j];
        angulo = (angulo < 0) ? angulo + 180 : angulo;

        double q = 0, r = 0;
        if ((angulo >= 0 && angulo < 22.5) || (angulo >= 157.5 && angulo <= 180)) {
            q = gradiente[i * largura + (j + 1)];
            r = gradiente[i * largura + (j - 1)];
        } else if (angulo >= 22.5 && angulo < 67.5) {
            q = gradiente[(i + 1) * largura + (j - 1)];
            r = gradiente[(i - 1) * largura + (j + 1)];
        } else if (angulo >= 67.5 && angulo < 112.5) {
            q = gradiente[(i + 1) * largura + j];
            r = gradiente[(i - 1) * largura + j];
        } else if (angulo >= 112.5 && angulo < 157.5) {
            q = gradiente[(i - 1) * largura + (j - 1)];
            r = gradiente[(i + 1) * largura + (j + 1)];
        }

        output[i * largura + j] = (gradiente[i * largura + j] >= q && gradiente[i * largura + j] >= r) ? gradiente[i * largura + j] : 0;
    }
}

// Kernel para aplicar thresholds
__global__ void aplicar_thresholds(double *input, double *output, double low, double high, int largura, int altura) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && j > 0 && i < altura - 1 && j < largura - 1) {
        if (input[i * largura + j] >= high) {
            output[i * largura + j] = 255;
        } else if (input[i * largura + j] >= low) {
            output[i * largura + j] = 128;
        } else {
            output[i * largura + j] = 0;
        }
    }
}

// Função para imprimir a matriz
void imprimir_matriz(double *matriz, int largura, int altura) {
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < largura; j++) {
            printf("%.2f ", matriz[i * largura + j]);
        }
        printf("\n");
    }
}

int main() {
    // Inicialização de variáveis
    double h_input[LARGURA * ALTURA] = {
        0, 50, 100, 150, 200,
        50, 100, 150, 200, 250,
        100, 150, 200, 250, 300,
        150, 200, 250, 300, 350,
        200, 250, 300, 350, 400
    };
    double *d_input, *d_gradiente, *d_direcao, *d_output;
    double h_output[LARGURA * ALTURA];

    // Alocar memória no GPU
    cudaMalloc((void **)&d_input, LARGURA * ALTURA * sizeof(double));
    cudaMalloc((void **)&d_gradiente, LARGURA * ALTURA * sizeof(double));
    cudaMalloc((void **)&d_direcao, LARGURA * ALTURA * sizeof(double));
    cudaMalloc((void **)&d_output, LARGURA * ALTURA * sizeof(double));

    // Copiar dados para o GPU
    cudaMemcpy(d_input, h_input, LARGURA * ALTURA * sizeof(double), cudaMemcpyHostToDevice);

    // Configuração de threads e blocos
    dim3 threadsPorBloco(16, 16);
    dim3 blocosPorGrade((LARGURA + threadsPorBloco.x - 1) / threadsPorBloco.x, 
                        (ALTURA + threadsPorBloco.y - 1) / threadsPorBloco.y);

    // Executar os kernels
    calcular_gradientes<<<blocosPorGrade, threadsPorBloco>>>(d_input, d_gradiente, d_direcao, LARGURA, ALTURA);
    supressao_nao_maxima<<<blocosPorGrade, threadsPorBloco>>>(d_gradiente, d_direcao, d_output, LARGURA, ALTURA);
    aplicar_thresholds<<<blocosPorGrade, threadsPorBloco>>>(d_output, d_output, 100, 200, LARGURA, ALTURA);

    // Copiar resultado de volta para a CPU
    cudaMemcpy(h_output, d_output, LARGURA * ALTURA * sizeof(double), cudaMemcpyDeviceToHost);

    // Imprimir a matriz resultante
    printf("Matriz de Entrada:\n");
    imprimir_matriz(h_input, LARGURA, ALTURA);

    printf("\nMatriz Final com Thresholds Aplicados:\n");
    imprimir_matriz(h_output, LARGURA, ALTURA);

    // Liberar memória
    cudaFree(d_input);
    cudaFree(d_gradiente);
    cudaFree(d_direcao);
    cudaFree(d_output);

    return 0;
}
