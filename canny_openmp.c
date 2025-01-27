#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

double **allocate_matrix(int width, int height) {
    double **matrix = (double **)malloc(height * sizeof(double *));
    for (int i = 0; i < height; i++) {
        matrix[i] = (double *)malloc(width * sizeof(double));
    }
    return matrix;
}

void free_matrix(double **matrix, int height) {
    for (int i = 0; i < height; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void calculate_gradients(double **input, double **gradient, double **direction, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            double gx = input[i - 1][j + 1] + 2 * input[i][j + 1] + input[i + 1][j + 1]
                      - input[i - 1][j - 1] - 2 * input[i][j - 1] - input[i + 1][j - 1];
            double gy = input[i - 1][j - 1] + 2 * input[i - 1][j] + input[i - 1][j + 1]
                      - input[i + 1][j - 1] - 2 * input[i + 1][j] - input[i + 1][j + 1];
            gradient[i][j] = sqrt(gx * gx + gy * gy);
            direction[i][j] = atan2(gy, gx) * 180 / PI;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    double **input = allocate_matrix(size, size);
    double **gradient = allocate_matrix(size, size);
    double **direction = allocate_matrix(size, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            input[i][j] = (i + j) % 256;
        }
    }

    printf("Executing Canny OpenMP with matrix size %dx%d...\n", size, size);

    double start = omp_get_wtime();
    calculate_gradients(input, gradient, direction, size, size);
    double end = omp_get_wtime();

    printf("Execution Time (OpenMP): %.6f seconds\n", end - start);

    free_matrix(input, size);
    free_matrix(gradient, size);
    free_matrix(direction, size);

    return 0;
}
