#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Função de exemplo que simula uma operação pesada
void exemploOperacaoPesada() {
    for (int i = 0; i < 100000000; i++) {
        // Simulação de operação
    }
}

int main() {
    struct timespec inicio, fim;
    double tempoGasto;

    printf("Iniciando medição de tempo...\n");

    // Início da medição
    clock_gettime(CLOCK_MONOTONIC, &inicio);

    // Operação cuja duração será medida
    exemploOperacaoPesada();

    // Fim da medição
    clock_gettime(CLOCK_MONOTONIC, &fim);

    // Calcular tempo gasto em segundos
    tempoGasto = (fim.tv_sec - inicio.tv_sec) + (fim.tv_nsec - inicio.tv_nsec) / 1e9;

    printf("Tempo gasto para a execução: %f segundos\n", tempoGasto);

    return 0;
}
