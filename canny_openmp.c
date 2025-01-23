#include <stdio.h>
#include <stdlib.h>
#include <math.h> // Necessário para sqrt e atan2

#ifndef M_PI
#define M_PI 3.14159265358979323846 // Definir M_PI se não estiver disponível
#endif

// Função para calcular os gradientes e suas direções
void calcular_gradientes(double **input, double **gradient, double **direction, int width, int height) {
    int i, j;
    double gx, gy;

    #pragma omp parallel for private(i, j, gx, gy) shared(input, gradient, direction)
    for (i = 1; i < height - 1; i++) {
        for (j = 1; j < width - 1; j++) {
            // Sobel - cálculo do gradiente nas direções X e Y
            gx = (input[i - 1][j + 1] + 2 * input[i][j + 1] + input[i + 1][j + 1]) -
                 (input[i - 1][j - 1] + 2 * input[i][j - 1] + input[i + 1][j - 1]);
            gy = (input[i + 1][j - 1] + 2 * input[i + 1][j] + input[i + 1][j + 1]) -
                 (input[i - 1][j - 1] + 2 * input[i - 1][j] + input[i - 1][j + 1]);

            // Cálculo da magnitude do gradiente
            gradient[i][j] = sqrt(gx * gx + gy * gy);

            // Cálculo da direção do gradiente
            direction[i][j] = atan2(gy, gx) * 180 / M_PI; // Converte para graus
        }
    }
}

// Função para imprimir matrizes para depuração
void imprimir_matriz(double **matriz, int height, int width, const char *titulo) {
    printf("%s:\n", titulo);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matriz[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Função para realizar supressão não máxima
void supressao_nao_maxima(double **gradient, double **direction, double **output, int width, int height) {
    int i, j;
    #pragma omp parallel for private(i, j) shared(gradient, direction, output)
    for (i = 1; i < height - 1; i++) {
        for (j = 1; j < width - 1; j++) {
            double angle = direction[i][j];
            double neighbor1 = 0, neighbor2 = 0;

            // Normalização do ângulo entre 0 e 180
            if (angle < 0) angle += 180;

            // Determinar vizinhos baseados na direção
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                neighbor1 = gradient[i][j - 1];
                neighbor2 = gradient[i][j + 1];
            } else if (angle >= 22.5 && angle < 67.5) {
                neighbor1 = gradient[i - 1][j + 1];
                neighbor2 = gradient[i + 1][j - 1];
            } else if (angle >= 67.5 && angle < 112.5) {
                neighbor1 = gradient[i - 1][j];
                neighbor2 = gradient[i + 1][j];
            } else if (angle >= 112.5 && angle < 157.5) {
                neighbor1 = gradient[i - 1][j - 1];
                neighbor2 = gradient[i + 1][j + 1];
            }

            // Verificar se é máximo local
            if (gradient[i][j] >= neighbor1 && gradient[i][j] >= neighbor2) {
                output[i][j] = gradient[i][j];
            } else {
                output[i][j] = 0;
            }
        }
    }
}

// Função para aplicar thresholds
void aplicar_thresholds(double **input, double **output, int width, int height, double low, double high) {
    int i, j;
    #pragma omp parallel for private(i, j) shared(input, output)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (input[i][j] >= high) {
                output[i][j] = 255; // Fortes
            } else if (input[i][j] >= low) {
                output[i][j] = 128; // Fracos
            } else {
                output[i][j] = 0; // Eliminados
            }
        }
    }
}

// Função principal
int main() {
    int width = 5, height = 5;
    double **input, **gradient, **direction, **non_max_supp, **thresholded;

    // Alocação de matrizes
    input = (double **)malloc(height * sizeof(double *));
    gradient = (double **)malloc(height * sizeof(double *));
    direction = (double **)malloc(height * sizeof(double *));
    non_max_supp = (double **)malloc(height * sizeof(double *));
    thresholded = (double **)malloc(height * sizeof(double *));
    for (int i = 0; i < height; i++) {
        input[i] = (double *)malloc(width * sizeof(double));
        gradient[i] = (double *)malloc(width * sizeof(double));
        direction[i] = (double *)malloc(width * sizeof(double));
        non_max_supp[i] = (double *)malloc(width * sizeof(double));
        thresholded[i] = (double *)malloc(width * sizeof(double));
    }

    // Preencher a matriz de entrada com valores de exemplo
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input[i][j] = i * 50 + j * 50;
        }
    }

    // Processamento
    imprimir_matriz(input, height, width, "Matriz de Entrada");
    calcular_gradientes(input, gradient, direction, width, height);
    imprimir_matriz(gradient, height, width, "Gradientes");
    imprimir_matriz(direction, height, width, "Direções");
    supressao_nao_maxima(gradient, direction, non_max_supp, width, height);
    imprimir_matriz(non_max_supp, height, width, "Supressão Não Máxima");
    aplicar_thresholds(non_max_supp, thresholded, width, height, 100, 200);
    imprimir_matriz(thresholded, height, width, "Thresholds Aplicados");

    // Liberar memória
    for (int i = 0; i < height; i++) {
        free(input[i]);
        free(gradient[i]);
        free(direction[i]);
        free(non_max_supp[i]);
        free(thresholded[i]);
    }
    free(input);
    free(gradient);
    free(direction);
    free(non_max_supp);
    free(thresholded);

    return 0;
}
