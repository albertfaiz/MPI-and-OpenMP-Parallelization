#!/bin/bash
# run_all.sh (Force g++-14 and Clear Env)

# Clear CXX and CC (Important!)
unset CXX
unset CC

# Force g++-14
COMPILER="/opt/homebrew/bin/g++-14"

# Force mpicxx
MPI_COMPILER="/opt/homebrew/bin/mpicxx"

# Check if g++-14 and mpicxx exist
if [[ ! -f "$COMPILER" ]]; then
    echo "Error: g++-14 not found at $COMPILER. Please install GCC 14 via Homebrew."
    exit 1
fi

if [[ ! -f "$MPI_COMPILER" ]]; then
    echo "Error: mpicxx not found at $MPI_COMPILER. Please install MPI via Homebrew."
    exit 1
fi

# Set common compiler flags
FLAGS="-O3 -std=c++17"
OPENMP_FLAGS="-fopenmp $FLAGS"

# Define the set of N values to test
N_values=(128 256 512 1024)

echo "Compiling serial version (nbody.cc)..."
"$COMPILER" "$FLAGS" nbody.cc -o nbody_serial || { echo "Serial compilation failed"; exit 1; }

echo "Compiling OpenMP version (nbody_omp.cc)..."
"$COMPILER" "$OPENMP_FLAGS" nbody_omp.cc -o nbody_omp || { echo "OpenMP compilation failed"; exit 1; }

echo "Compiling MPI-only version (nbody_mpi.cc)..."
"$MPI_COMPILER" "$FLAGS" nbody_mpi.cc -o nbody_mpi || { echo "MPI compilation failed"; exit 1; }

echo "Compiling Hybrid MPI+OpenMP version (nbody_mpi_omp.cc)..."
"$MPI_COMPILER" "$OPENMP_FLAGS" nbody_mpi_omp.cc -o nbody_mpi_omp || { echo "Hybrid MPI+OpenMP compilation failed"; exit 1; }

echo "Compiling Shared-Memory MPI version (nbody_mpi_shared.cc)..."
"$MPI_COMPILER" "$FLAGS" nbody_mpi_shared.cc -o nbody_mpi_shared || { echo "Shared-Memory MPI compilation failed"; exit 1; }

# Loop over each N value and run all executables.
for N in "${N_values[@]}"; do
    echo "-----------------------------"
    echo "Running Serial version for N = $N"
    ./nbody_serial $N > serial_${N}.txt

    echo "Running OpenMP version for N = $N"
    ./nbody_omp $N > openmp_${N}.txt

    echo "Running MPI-only version for N = $N (using 4 MPI processes)"
    mpiexec -n 4 ./nbody_mpi $N > mpi_${N}.txt

    echo "Running Hybrid MPI+OpenMP version for N = $N (using 2 MPI processes)"
    mpiexec -n 2 ./nbody_mpi_omp $N > hybrid_${N}.txt

    echo "Running Shared-Memory MPI version for N = $N (using 4 MPI processes)"
    mpiexec -n 4 ./nbody_mpi_shared $N > shared_${N}.txt
done

echo "All runs completed. Generated output files:"
ls -1 *_*.txt