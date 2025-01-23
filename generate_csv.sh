#!/bin/bash

output_file="benchmark_results.csv"

# Cabeçalho
echo "Dimensão,Sequencial (s),OpenMP (s),GPU (s),Speedup OpenMP,Speedup GPU" > $output_file

# Adiciona resultados
seq_times=(0.008279 0.002008)
omp_times=(0.020485 0.021755)
gpu_times=(0.014818 0.001484)
dims=("1024x1024" "512x512")

for i in "${!dims[@]}"; do
    seq_time=${seq_times[i]}
    omp_time=${omp_times[i]}
    gpu_time=${gpu_times[i]}
    speedup_omp=$(awk "BEGIN {printf \"%.4f\", $seq_time / $omp_time}")
    speedup_gpu=$(awk "BEGIN {printf \"%.4f\", $seq_time / $gpu_time}")
    echo "${dims[i]},$seq_time,$omp_time,$gpu_time,$speedup_omp,$speedup_gpu" >> $output_file
done

echo "Resultados exportados para $output_file"
