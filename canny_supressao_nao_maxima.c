#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Função para calcular gradientes e direções
void calcular_gradientes(double **input, double **gradient, double **direction, int width, int height) {
    int i, j;
    double GX, GY;
    for (i = 1; i < height - 1; i++) {
        for (j = 1; j < width - 1; j++) {
            GX = (input[i - 1][j + 1] + 2 * input[i][j + 1] + input[i + 1][j + 1]) -
                 (input[i - 1][j - 1] + 2 * input[i][j - 1] + input[i + 1][j - 1]);
            GY = (input[i + 1][j - 1] + 2 * input[i + 1][j] + input[i + 1][j + 1]) -
                 (input[i - 1][j - 1] + 2 * input[i - 1][j] + input[i - 1][j + 1]);
            gradient[i][j] = sqrt(GX * GX + GY * GY);
            direction[i][j] = atan2(GY, GX) * 180 / M_PI; // Converter para graus
        }
    }
}

// Função para supressão não máxima
void supressao_nao_maxima(double **gradient, double **direction, double **output, int width, int height) {
    int i, j;
    for (i = 1; i < height - 1; i++) {
        for (j = 1; j < width - 1; j++) {
            double angle = direction[i][j];
            double magnitude = gradient[i][j];
            double neighbor1 = 0.0, neighbor2 = 0.0;

            // Normalizar o ângulo entre 0 e 180
            if (angle < 0) angle += 180;

            // Determinar os vizinhos com base no ângulo
            if ((angle >= 0 && angle <= 22.5) || (angle > 157.5 && angle <= 180)) {
                neighbor1 = gradient[i][j - 1];
                neighbor2 = gradient[i][j + 1];
            } else if (angle > 22.5 && angle <= 67.5) {
                neighbor1 = gradient[i - 1][j + 1];
                neighbor2 = gradient[i + 1][j - 1];
            } else if (angle > 67.5 && angle <= 112.5) {
                neighbor1 = gradient[i - 1][j];
                neighbor2 = gradient[i + 1][j];
            } else if (angle > 112.5 && angle <= 157.5) {
                neighbor1 = gradient[i - 1][j - 1];
                neighbor2 = gradient[i + 1][j + 1];
            }

            // Verificar se o pixel é o maior na direção
            if (magnitude > neighbor1 && magnitude > neighbor2) {
                output[i][j] = magnitude; // Máximo local
            } else {
                output[i][j] = 0.0; // Não é máximo local
            }
        }
    }
}

// Função para imprimir matrizes para depuração
void imprimir_matriz(double **matriz, int width, int height, const char *titulo) {
    printf("%s:\n", titulo);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matriz[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Função principal
int main() {
    int width = 5, height = 5;

    // Matriz de entrada para teste
    double **input = (double **)malloc(height * sizeof(double *));
    double **gradient = (double **)malloc(height * sizeof(double *));
    double **direction = (double **)malloc(height * sizeof(double *));
    double **output = (double **)malloc(height * sizeof(double *));
    for (int i = 0; i < height; i++) {
        input[i] = (double *)calloc(width, sizeof(double));
        gradient[i] = (double *)calloc(width, sizeof(double));
        direction[i] = (double *)calloc(width, sizeof(double));
        output[i] = (double *)calloc(width, sizeof(double));
    }

    // Preencher a matriz de entrada
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            input[i][j] = 255.0;
        }
    }

    // Calcular gradientes e direções
    calcular_gradientes(input, gradient, direction, width, height);

    // Aplicar supressão não máxima
    supressao_nao_maxima(gradient, direction, output, width, height);

    // Imprimir resultados
    imprimir_matriz(input, width, height, "Matriz de Entrada");
    imprimir_matriz(gradient, width, height, "Gradientes");
    imprimir_matriz(direction, width, height, "Direções");
    imprimir_matriz(output, width, height, "Resultado Supressão Não Máxima");

    // Liberar memória
    for (int i = 0; i < height; i++) {
        free(input[i]);
        free(gradient[i]);
        free(direction[i]);
        free(output[i]);
    }
    free(input);
    free(gradient);
    free(direction);
    free(output);

    return 0;
}
