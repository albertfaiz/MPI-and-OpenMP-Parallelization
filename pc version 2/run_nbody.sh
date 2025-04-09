#!/bin/bash

# Function to check if compilation was successful
check_compile() {
    if [ $? -ne 0 ]; then
        echo "Error: Compilation of $1 failed"
        exit 1
    fi
}

# Function to check if executable exists
check_executable() {
    if [ ! -f "$1" ]; then
        echo "Error: Executable $1 not found"
        exit 1
    fi
}

echo "Compiling serial code..."
g++ -std=c++17 -o nbody nbody.cc
check_compile "nbody.cc"

echo "Compiling Threaded code..."
g++ -std=c++17 -o nbody_threaded nbody_threaded.cc
check_compile "nbody_threaded.cc"

echo "Compiling MPI code..."
mpic++ -std=c++17 -o nbody_mpi nbody_mpi.cc
check_compile "nbody_mpi.cc"

echo "Compiling Hybrid MPI+Threads code..."
mpic++ -std=c++17 -o nbody_mpi_threaded nbody_mpi_threaded.cc
check_compile "nbody_mpi_threaded.cc"

echo "Compiling Shared-Memory MPI code..."
mpic++ -std=c++17 -o nbody_mpi_shared nbody_mpi_shared.cc
check_compile "nbody_mpi_shared.cc"

# Loop over different problem sizes
for N in 128 256 512 1024
do
    echo "Running for N=$N..."

    echo "Running serial..."
    check_executable "./nbody"
    ./nbody $N > serial_$N.txt 2>&1

    echo "Running Threaded (4 threads)..."
    check_executable "./nbody_threaded"
    ./nbody_threaded $N > threaded_$N.txt 2>&1

    echo "Running MPI (4 ranks)..."
    check_executable "./nbody_mpi"
    mpirun -np 4 ./nbody_mpi $N > mpi_$N.txt 2>&1

    echo "Running Hybrid (2 ranks, 4 threads each)..."
    check_executable "./nbody_mpi_threaded"
    mpirun -np 2 ./nbody_mpi_threaded $N > hybrid_$N.txt 2>&1

    echo "Running Shared-Memory MPI (4 ranks)..."
    check_executable "./nbody_mpi_shared"
    mpirun -np 4 ./nbody_mpi_shared $N > mpi_shared_$N.txt 2>&1
done

echo "Done! Outputs are in serial_*.txt, threaded_*.txt, mpi_*.txt, hybrid_*.txt, mpi_shared_*.txt"
echo "Kinetic energy data is in KE_*.txt files, and time data is in time_*.txt files."