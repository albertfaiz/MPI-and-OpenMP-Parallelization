#!/bin/bash
# run_experiments.sh
# This script runs the serial, OpenMP, MPI, and Hybrid MPI+OpenMP versions
# over a range of problem sizes.
# Make sure to set OMP_NUM_THREADS for OpenMP/hybrid runs as needed.

# List of N values to test (number of masses)
N_values=(128 256 512 1024 2048 4096 8192) 

echo "=== Running Serial Version (nbody) ==="
for N in "${N_values[@]}"; do
    echo "Running nbody with N=$N"
    ./nbody $N
done

echo "=== Running OpenMP Version (nbody_omp) ==="
for N in "${N_values[@]}"; do
    echo "Running nbody_omp with N=$N"
    ./nbody_omp $N
done

echo "=== Running MPI Version (nbody_mpi) with 1 MPI process ==="
for N in "${N_values[@]}"; do
    echo "Running nbody_mpi with N=$N"
    mpiexec -n 1 ./nbody_mpi $N
done

echo "=== Running Hybrid MPI+OpenMP Version (nbody_mpi_omp) with 2 MPI processes, 4 threads each ==="
export OMP_NUM_THREADS=4
for N in "${N_values[@]}"; do
    echo "Running nbody_mpi_omp with N=$N"
    mpiexec -n 2 ./nbody_mpi_omp $N
done
