#!/bin/bash
#SBATCH --job-name=nbody_mpi_omp
#SBATCH --output=nbody_mpi_omp_%j.out
#SBATCH --error=nbody_mpi_omp_%j.err
#SBATCH --time=00:30:00       # Adjust based on expected runtime
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=4GB

# Load MPI module (if necessary)
module load mpi

# Set the number of OpenMP threads
export OMP_NUM_THREADS=4

# Compile if needed (only if binary not already built)
# mpicxx -fopenmp -O3 -std=c++17 nbody_mpi_omp.cc -o nbody_mpi_omp

# Run the hybrid MPI+OpenMP version with 1024 masses
mpiexec -n 2 ./nbody_mpi_omp 1024
