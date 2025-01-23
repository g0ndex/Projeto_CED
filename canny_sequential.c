#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void executarCanny(int dim) {
    double **matriz = (double **)malloc(dim * sizeof(double *));
    for (int i = 0; i < dim; i++) {
        matriz[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            matriz[i][j] = (double)(i + j);
        }
    }

    printf("Executando Canny Sequencial com matriz de dimensão %dx%d...\n", dim, dim);

    // Simulação de cálculo (alteração dos valores da matriz)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            matriz[i][j] += 1.0;
        }
    }

    for (int i = 0; i < dim; i++) {
        free(matriz[i]);
    }
    free(matriz);
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

    clock_t start = clock();
    executarCanny(dim);
    clock_t end = clock();

    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tempo de execução (Sequencial): %.6f segundos\n", elapsed_time);

    return 0;
}
