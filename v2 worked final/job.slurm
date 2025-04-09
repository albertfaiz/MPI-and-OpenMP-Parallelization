#!/bin/bash
#SBATCH --job-name=nbody_benchmark_heavy
#SBATCH --output=nbody_%j.out
#SBATCH --error=nbody_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=04:00:00
#SBATCH --partition=normal

# Load necessary modules
module load intel/18.0.2.199
module load mvapich2/2.3.1

# Compile all versions with optimization
icc -O3 -std=c++17 nbody.cc -o nbody || exit 1
icc -O3 -std=c++17 -qopenmp nbody_omp.cc -o nbody_omp || exit 1
mpiicc -O3 -std=c++17 nbody_mpi.cc -o nbody_mpi || exit 1
mpiicc -O3 -std=c++17 -qopenmp nbody_mpi_omp.cc -o nbody_mpi_omp || exit 1
mpiicc -O3 -std=c++17 nbody_mpi_shared.cc -o nbody_mpi_shared || exit 1

# Benchmark serial version
echo "Running Serial Version"
for N in 512 1024 2048 4096; do
    echo "N=$N" >> serial_times.txt
    ./nbody $N >> serial_times.txt 2>&1
done

# Benchmark OpenMP version
echo "Running OpenMP Version"
for threads in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$threads
    for N in 1024 2048 4096; do
        echo "Threads=$threads N=$N" >> omp_times_${threads}.txt
        ./nbody_omp $N >> omp_times_${threads}.txt 2>&1
    done
done

# Benchmark MPI version
echo "Running MPI Version"
for ranks in 1 2 4 8 16; do
    for N in 1024 2048 4096; do
        echo "Ranks=$ranks N=$N" >> mpi_times_${ranks}.txt
        mpiexec -n $ranks ./nbody_mpi $N >> mpi_times_${ranks}.txt 2>&1
    done
done

# Benchmark Hybrid MPI+OpenMP version
echo "Running Hybrid MPI+OpenMP Version"
export MV2_ENABLE_AFFINITY=0  # Disable affinity to avoid oversubscription
for ranks in 2 4 8; do
    for threads in 2 4; do
        export OMP_NUM_THREADS=$threads
        for N in 1024 2048 4096; do
            echo "Ranks=$ranks Threads=$threads N=$N" >> hybrid_times_${ranks}_${threads}.txt
            mpiexec -n $ranks ./nbody_mpi_omp $N >> hybrid_times_${ranks}_${threads}.txt 2>&1
        done
    done
done

# Benchmark Shared-Memory MPI version
echo "Running Shared-Memory MPI Version"
for ranks in 1 2 4 8 16; do
    for N in 1024 2048 4096; do
        echo "Ranks=$ranks N=$N" >> shared_times_${ranks}.txt
        mpiexec -n $ranks ./nbody_mpi_shared $N >> shared_times_${ranks}.txt 2>&1
    done
done

echo "Benchmarking complete. Check output files for results."
