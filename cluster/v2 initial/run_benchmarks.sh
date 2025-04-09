#!/bin/bash

# Compile the codes
module load gcc
module load openmpi

g++ -O3 -std=c++17 nbody.cc -o nbody
g++ -fopenmp -O3 -std=c++17 nbody_omp.cc -o nbody_omp
mpicxx -O3 -std=c++17 nbody_mpi.cc -o nbody_mpi
mpicxx -fopenmp -O3 -std=c++17 nbody_mpi_omp.cc -o nbody_mpi_omp

# Serial runs
echo "Running serial version..."
for N in 128 256 512 1024 2048; do
    echo "N = $N"
    ./nbody $N > serial_N${N}.out
done

# OpenMP runs
echo "Running OpenMP version..."
for N in 1024 2048 4096 8192; do
    for threads in 1 2 4 8 16; do
        echo "N = $N, threads = $threads"
        export OMP_NUM_THREADS=$threads
        ./nbody_omp $N > omp_N${N}_threads${threads}.out
    done
done

# MPI runs
echo "Running MPI version..."
for N in 1024 2048 4096 8192; do
    for ranks in 1 2 4 8 16; do
        echo "N = $N, ranks = $ranks"
        mpiexec -n $ranks ./nbody_mpi $N > mpi_N${N}_ranks${ranks}.out
    done
done

# Hybrid MPI+OpenMP runs
echo "Running Hybrid MPI+OpenMP version..."
for N in 8192 16384 32768; do
    for ranks in 2 4 8; do
        for threads in 2 4 8; do
            echo "N = $N, ranks = $ranks, threads = $threads"
            export OMP_NUM_THREADS=$threads
            mpiexec -n $ranks ./nbody_mpi_omp $N > hybrid_N${N}_ranks${ranks}_threads${threads}.out
        done
    done
done

echo "All runs completed."