#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Função para calcular os gradientes e suas direções
void calcular_gradientes(double **input, double **gradient, double **direction, int width, int height) {
    int i, j;
    double gx, gy;
    for (i = 1; i < height - 1; i++) {
        for (j = 1; j < width - 1; j++) {
            // Sobel - cálculo do gradiente nas direções X e Y
            gx = (input[i - 1][j + 1] + 2 * input[i][j + 1] + input[i + 1][j + 1]) -
                 (input[i - 1][j - 1] + 2 * input[i][j - 1] + input[i + 1][j - 1]);
            gy = (input[i - 1][j - 1] + 2 * input[i - 1][j] + input[i - 1][j + 1]) -
                 (input[i + 1][j - 1] + 2 * input[i + 1][j] + input[i + 1][j + 1]);
            
            // Cálculo da magnitude do gradiente
            gradient[i][j] = sqrt(gx * gx + gy * gy);
            
            // Cálculo da direção do gradiente
            direction[i][j] = atan2(gy, gx) * 180 / M_PI; // Converte para graus
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
    int width = 5, height = 5; // Tamanho da matriz de exemplo
    double **input, **gradient, **direction;

    // Alocação dinâmica de memória
    input = (double **)malloc(height * sizeof(double *));
    gradient = (double **)malloc(height * sizeof(double *));
    direction = (double **)malloc(height * sizeof(double *));
    for (int i = 0; i < height; i++) {
        input[i] = (double *)malloc(width * sizeof(double));
        gradient[i] = (double *)malloc(width * sizeof(double));
        direction[i] = (double *)malloc(width * sizeof(double));
    }

    // Dados de entrada (exemplo com borda)
    double input_example[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 255, 255, 255, 0},
        {0, 255, 255, 255, 0},
        {0, 255, 255, 255, 0},
        {0, 0, 0, 0, 0}
    };

    // Copia os dados de entrada para a matriz alocada
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input[i][j] = input_example[i][j];
            gradient[i][j] = 0;   // Inicializa gradientes com 0
            direction[i][j] = 0;  // Inicializa direções com 0
        }
    }

    // Calcula gradientes e direções
    calcular_gradientes(input, gradient, direction, width, height);

    // Imprime as matrizes para depuração
    imprimir_matriz(input, width, height, "Matriz de Entrada");
    imprimir_matriz(gradient, width, height, "Gradientes");
    imprimir_matriz(direction, width, height, "Direções");

    // Liberação de memória
    for (int i = 0; i < height; i++) {
        free(input[i]);
        free(gradient[i]);
        free(direction[i]);
    }
    free(input);
    free(gradient);
    free(direction);

    return 0;
}
