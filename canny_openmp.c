#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void executarCannyOpenMP(int dim) {
    double **matriz = (double **)malloc(dim * sizeof(double *));
    for (int i = 0; i < dim; i++) {
        matriz[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            matriz[i][j] = (double)(i + j);
        }
    }

    printf("Executando Canny OpenMP com matriz de dimensão %dx%d...\n", dim, dim);

    #pragma omp parallel for
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

    double start = omp_get_wtime();
    executarCannyOpenMP(dim);
    double end = omp_get_wtime();

    double elapsed_time = end - start;
    printf("Tempo de execução (OpenMP): %.6f segundos\n", elapsed_time);

    return 0;
}
