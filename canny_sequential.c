#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Constantes para o filtro Gaussiano
#define KERNEL_SIZE 5
#define SIGMA 1.0

// Função para aplicar o filtro Gaussiano
void apply_gaussian_filter(double** image, int width, int height, double** output) {
    // Criar kernel Gaussiano
    double kernel[KERNEL_SIZE][KERNEL_SIZE];
    double sum = 0.0;
    int offset = KERNEL_SIZE / 2;

    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            double x = i - offset;
            double y = j - offset;
            kernel[i][j] = exp(-(x * x + y * y) / (2 * SIGMA * SIGMA)) / (2 * M_PI * SIGMA * SIGMA);
            sum += kernel[i][j];
        }
    }

    // Normalizar o kernel
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            kernel[i][j] /= sum;
        }
    }

    // Aplicar convolução
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double pixel = 0.0;
            for (int ki = -offset; ki <= offset; ki++) {
                for (int kj = -offset; kj <= offset; kj++) {
                    int x = i + ki;
                    int y = j + kj;
                    if (x >= 0 && x < height && y >= 0 && y < width) {
                        pixel += image[x][y] * kernel[ki + offset][kj + offset];
                    }
                }
            }
            output[i][j] = pixel;
        }
    }
}

// Função principal
int main() {
    // Configurar dimensões da imagem
    int width = 5, height = 5;

    // Alocar memória para a imagem e saída
    double** image = (double**)malloc(height * sizeof(double*));
    double** blurred = (double**)malloc(height * sizeof(double*));
    for (int i = 0; i < height; i++) {
        image[i] = (double*)malloc(width * sizeof(double));
        blurred[i] = (double*)malloc(width * sizeof(double));
    }

    // Inicializar a imagem com valores de exemplo
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i][j] = (i == j) ? 1.0 : 0.0; // Diagonal com valores 1.0
        }
    }

    // Aplicar o filtro Gaussiano
    apply_gaussian_filter(image, width, height, blurred);

    // Imprimir a saída
    printf("Imagem suavizada:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", blurred[i][j]);
        }
        printf("\n");
    }

    // Libertar memória
    for (int i = 0; i < height; i++) {
        free(image[i]);
        free(blurred[i]);
    }
    free(image);
    free(blurred);

    return 0;
}
