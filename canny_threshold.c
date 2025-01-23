#include <stdio.h>
#include <stdlib.h>

#define WIDTH 5   // Largura da matriz
#define HEIGHT 5  // Altura da matriz

// Função para aplicar thresholds a uma matriz
void aplicar_threshold(double **input, double **output, int width, int height, double low_threshold, double high_threshold) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Aplica thresholds:
            // - Valores abaixo do low_threshold são 0 (fracos).
            // - Valores entre low_threshold e high_threshold são 128 (médios).
            // - Valores acima ou iguais ao high_threshold são 255 (fortes).
            if (input[i][j] >= high_threshold) {
                output[i][j] = 255;  // Forte
            } else if (input[i][j] >= low_threshold) {
                output[i][j] = 128;  // Médio
            } else {
                output[i][j] = 0;    // Fraco
            }
        }
    }
}

// Função para imprimir matrizes no terminal
void imprimir_matriz(double **matriz, int width, int height, const char *titulo) {
    printf("%s:\n", titulo);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matriz[i][j]);  // Imprime o valor com 2 casas decimais
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Inicialização de thresholds
    double low_threshold = 100;  // Limite inferior para valores fracos
    double high_threshold = 200; // Limite superior para valores fortes

    // Alocação dinâmica da matriz de entrada e saída
    double **input = (double **)malloc(HEIGHT * sizeof(double *));
    double **output = (double **)malloc(HEIGHT * sizeof(double *));
    for (int i = 0; i < HEIGHT; i++) {
        input[i] = (double *)malloc(WIDTH * sizeof(double));
        output[i] = (double *)malloc(WIDTH * sizeof(double));
    }

    // Exemplo de inicialização da matriz de entrada
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (i > 0 && i < HEIGHT - 1 && j > 0 && j < WIDTH - 1) {
                input[i][j] = 50 * (i + j);  // Valores de teste
            } else {
                input[i][j] = 0;  // Zonas de borda
            }
        }
    }

    // Imprime a matriz de entrada
    imprimir_matriz(input, WIDTH, HEIGHT, "Matriz de Entrada");

    // Aplica o threshold à matriz de entrada e salva o resultado na matriz de saída
    aplicar_threshold(input, output, WIDTH, HEIGHT, low_threshold, high_threshold);

    // Imprime a matriz após aplicar o threshold
    imprimir_matriz(output, WIDTH, HEIGHT, "Matriz com Threshold");

    // Libera memória alocada dinamicamente
    for (int i = 0; i < HEIGHT; i++) {
        free(input[i]);
        free(output[i]);
    }
    free(input);
    free(output);

    return 0;
}
