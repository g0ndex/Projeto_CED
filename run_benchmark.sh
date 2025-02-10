#!/bin/bash

# Output file for benchmark results
output_file="benchmark_results.txt"
csv_file="benchmark_results.csv"
echo "Starting benchmark with time measurements..." > $output_file
echo "Matrix Size,Version,Execution Time (s),Speedup" > $csv_file

# Function to calculate speedup
calculate_speedup() {
    seq_time=$1
    parallel_time=$2
    if (( $(echo "$parallel_time > 0" | bc -l) )); then
        echo "scale=4; $seq_time / $parallel_time" | bc
    else
        echo "Infinity"
    fi
}

# Matrix sizes to benchmark
matrix_sizes=(128 256 512 1024 2048 4096)

# Compile programs
gcc canny_sequential.c -o canny_sequential -lm
gcc canny_openmp.c -o canny_openmp -fopenmp -lm
nvcc canny_gpu.cu -o canny_gpu -lm

# Benchmark loop
for size in "${matrix_sizes[@]}"; do
    echo "Running for matrix size ${size}x${size}..."

    # Sequential Version
    echo "Compiling and running Sequential version for ${size}x${size}..."
    start=$(date +%s.%N)
    ./canny_sequential $size >> $output_file
    end=$(date +%s.%N)
    seq_time=$(echo "$end - $start" | bc)
    echo "Execution Time (Sequential): $seq_time seconds" >> $output_file

    # OpenMP Version
    echo "Compiling and running OpenMP version for ${size}x${size}..."
    start=$(date +%s.%N)
    ./canny_openmp $size >> $output_file
    end=$(date +%s.%N)
    openmp_time=$(echo "$end - $start" | bc)
    openmp_speedup=$(calculate_speedup $seq_time $openmp_time)
    echo "Execution Time (OpenMP): $openmp_time seconds" >> $output_file
    echo "Speedup (OpenMP): $openmp_speedup" >> $output_file

    # GPU Version
    echo "Compiling and running GPU version for ${size}x${size}..."
    start=$(date +%s.%N)
    ./canny_gpu $size >> $output_file
    end=$(date +%s.%N)
    gpu_time=$(echo "$end - $start" | bc)
    gpu_speedup=$(calculate_speedup $seq_time $gpu_time)
    echo "Execution Time (GPU): $gpu_time seconds" >> $output_file
    echo "Speedup (GPU): $gpu_speedup" >> $output_file

    # Save results to CSV
    echo "$size,Sequential,$seq_time," >> $csv_file
    echo "$size,OpenMP,$openmp_time,$openmp_speedup" >> $csv_file
    echo "$size,GPU,$gpu_time,$gpu_speedup" >> $csv_file

    echo "---------------------------------------------"
done

echo "Benchmark completed. Results saved to $output_file and $csv_file."
