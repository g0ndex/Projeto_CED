#!/bin/bash

# Ficheiro de saída para os resultados
output_file="benchmark_results.txt"
echo "Iniciando benchmark com medições de tempo..." > $output_file
echo "Resultados de execução:" >> $output_file

# Dimensões das matrizes para teste
dimensoes=(128 256 512 1024)

# Loop para cada dimensão de matriz
for dim in "${dimensoes[@]}"; do
    echo "Executando para matriz de dimensão ${dim}x${dim}..." >> $output_file

    # Variável para armazenar tempo da versão sequencial
    sequencial_time=0

    # Compilar e executar a versão Sequencial
    echo "Compilando e executando a versão Sequencial para ${dim}x${dim}..."
    gcc canny_sequential.c -o canny_sequential -lm
    start=$(date +%s.%N)
    ./canny_sequential $dim >> $output_file
    end=$(date +%s.%N)
    sequencial_time=$(echo "$end - $start" | bc)
    echo "Tempo de execução (Sequencial): $sequencial_time segundos" >> $output_file

    # Compilar e executar a versão OpenMP
    echo "Compilando e executando a versão OpenMP para ${dim}x${dim}..."
    gcc canny_openmp.c -o canny_openmp -fopenmp -lm
    start=$(date +%s.%N)
    ./canny_openmp $dim >> $output_file
    end=$(date +%s.%N)
    openmp_time=$(echo "$end - $start" | bc)
    openmp_speedup=$(awk "BEGIN {printf \"%.6f\", $sequencial_time / $openmp_time}")
    echo "Tempo de execução (OpenMP): $openmp_time segundos" >> $output_file
    echo "Speedup (OpenMP): $openmp_speedup" >> $output_file

    # Compilar e executar a versão GPU
    echo "Compilando e executando a versão GPU para ${dim}x${dim}..."
    nvcc canny_gpu.cu -o canny_gpu -lm
    start=$(date +%s.%N)
    ./canny_gpu $dim >> $output_file
    end=$(date +%s.%N)
    gpu_time=$(echo "$end - $start" | bc)
    gpu_speedup=$(awk "BEGIN {printf \"%.6f\", $sequencial_time / $gpu_time}")
    echo "Tempo de execução (GPU): $gpu_time segundos" >> $output_file
    echo "Speedup (GPU): $gpu_speedup" >> $output_file

    echo "-----------------------------" >> $output_file
done

# Exibir resultados
echo "Benchmark concluído. Resultados salvos em $output_file."
cat $output_file
