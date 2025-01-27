#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define PI 3.14159265358979323846

// Function prototypes
void calculate_gradients(double **input, double **gradient, double **direction, int width, int height);
bool compare_matrices(double **matrix1, double **matrix2, int width, int height);
double **allocate_matrix(int width, int height);
void free_matrix(double **matrix, int height);
void print_matrix(double **matrix, int width, int height);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    double **input = allocate_matrix(size, size);
    double **gradient = allocate_matrix(size, size);
    double **direction = allocate_matrix(size, size);

    // Initialize input matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            input[i][j] = (i + j) % 256;
        }
    }

    printf("Executing Canny Sequential with matrix size %dx%d...\n", size, size);

    clock_t start = clock();
    calculate_gradients(input, gradient, direction, size, size);
    clock_t end = clock();

    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution Time (Sequential): %.6f seconds\n", elapsed_time);

    free_matrix(input, size);
    free_matrix(gradient, size);
    free_matrix(direction, size);

    return 0;
}

void calculate_gradients(double **input, double **gradient, double **direction, int width, int height) {
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

bool compare_matrices(double **matrix1, double **matrix2, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (fabs(matrix1[i][j] - matrix2[i][j]) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}

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

void print_matrix(double **matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}
