/*
 * Programa: Implementação do algoritmo de detecção de bordas de Canny.
 * Foco: Supressão Não Máxima (Non-Maximum Suppression).
 * Entrada: Matrizes de gradiente e direção (previamente calculadas).
 * Saída: Matriz com supressão não máxima aplicada.
 * 
 * Funções principais:
 * - Supressão Não Máxima: Identifica e mantém apenas máximos locais nas direções especificadas.
 * - Depuração: Imprime dados de pixel e resultados intermediários.
 */


#include <stdio.h>
#include <math.h>

// Função de supressão não máxima
void non_max_suppression(double** gradient, double** direction, int width, int height, double** output) {
    // Imprimir gradientes para depuração
    printf("Gradientes:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", gradient[i][j]);
        }
        printf("\n");
    }

    // Imprimir direções para depuração
    printf("Direções:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", direction[i][j]);
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            double angle = direction[i][j];
            double magnitude = gradient[i][j];
            double neighbor1 = 0.0, neighbor2 = 0.0;

            // Normalizar ângulo para estar entre 0° e 180°
            if (angle < 0) angle += 180;

            // Determinar os vizinhos com base na direção
            if ((angle >= 0 && angle <= 22.5) || (angle > 157.5 && angle <= 180)) {
                neighbor1 = gradient[i][j - 1]; // Horizontal esquerda
                neighbor2 = gradient[i][j + 1]; // Horizontal direita
            } else if (angle > 22.5 && angle <= 67.5) {
                neighbor1 = gradient[i - 1][j + 1]; // Diagonal superior direita
                neighbor2 = gradient[i + 1][j - 1]; // Diagonal inferior esquerda
            } else if (angle > 67.5 && angle <= 112.5) {
                neighbor1 = gradient[i - 1][j]; // Vertical superior
                neighbor2 = gradient[i + 1][j]; // Vertical inferior
            } else if (angle > 112.5 && angle <= 157.5) {
                neighbor1 = gradient[i - 1][j - 1]; // Diagonal superior esquerda
                neighbor2 = gradient[i + 1][j + 1]; // Diagonal inferior direita
            }

            // Mensagem de depuração detalhada
            printf("DEBUG: Pixel [%d][%d] - Mag = %.2f, Neighbor1 = %.2f, Neighbor2 = %.2f, Angle = %.2f\n",
                   i, j, magnitude, neighbor1, neighbor2, angle);

            // Verificar se o pixel atual é o maior na sua direção
            if (magnitude > neighbor1 && magnitude > neighbor2) {
                output[i][j] = magnitude; // É máximo local
                printf("DEBUG: Pixel [%d][%d] é um máximo local.\n", i, j);
            } else {
                output[i][j] = 0.0; // Não é máximo local
                printf("DEBUG: Pixel [%d][%d] NÃO é um máximo local.\n", i, j);
            }
        }
    }

    // Imprimir resultado final
    printf("\nResultado da supressão não máxima:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", output[i][j]);
        }
        printf("\n");
    }
}


int main() {
    // Dimensões da matriz
    int width = 5, height = 5;

    // Inicializar matrizes de gradiente e direção (exemplo)
    double gradient[5][5] = {
        {1.00, 1.03, 1.00, 0.61, 0.22},
        {1.03, 0.13, 0.79, 0.96, 0.61},
        {1.00, 0.79, 0.00, 0.79, 1.00},
        {0.61, 0.96, 0.79, 0.13, 1.03},
        {0.22, 0.61, 1.00, 1.03, 1.00}
    };

    double direction[5][5] = {
        {45.00, 102.88, 123.27, 131.04, 135.00},
        {-12.88, 45.00, 133.34, 135.00, 138.96},
        {-33.27, -43.34, 0.00, 136.66, 146.73},
        {-41.04, -45.00, -46.66, -135.00, 167.12},
        {-45.00, -48.96, -56.73, -77.12, -135.00}
    };

    // Matriz de saída
    double output[5][5] = {0};

    // Criar ponteiros para as matrizes
    double* gradient_ptr[5], *direction_ptr[5], *output_ptr[5];
    for (int i = 0; i < height; i++) {
        gradient_ptr[i] = gradient[i];
        direction_ptr[i] = direction[i];
        output_ptr[i] = output[i];
    }

    // Chamar a função de supressão não máxima
    non_max_suppression(gradient_ptr, direction_ptr, width, height, output_ptr);

    return 0;
}
