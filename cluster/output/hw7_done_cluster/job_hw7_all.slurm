#!/bin/bash
#SBATCH --job-name=hw7_all
#SBATCH --output=hw7_all_%j.out
#SBATCH --error=hw7_all_%j.err
#SBATCH --time=01:00:00       # Adjust total runtime as needed
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8GB

# -------------------------
# Module Setup
# -------------------------
# Clean the environment first.
module purge

# Load the GNU compiler module that supports C++17.
module load gnu12/12.4.0 || { echo "Failed to load gnu12/12.4.0"; exit 1; }

# Load the desired MPI module.
# Choose ONE MPI module here. In your error message you were trying to load mpich.
module load mpich/3.4.3-ucx || { echo "Failed to load mpich/3.4.3-ucx"; exit 1; }

# Print loaded modules for debugging.
module list
echo "Using GCC: $(which gcc)"
gcc --version
echo "Using MPICXX: $(which mpicxx)"
mpicxx --version

# Set OpenMP environment variable.
export OMP_NUM_THREADS=4

# -------------------------
# Compilation
# -------------------------
echo "Compiling serial version (nbody.cc)..."
g++ -O3 -std=c++17 nbody.cc -o nbody_serial || { echo "Serial compilation failed"; exit 1; }

echo "Compiling OpenMP version (nbody_omp.cc)..."
g++ -fopenmp -O3 -std=c++17 nbody_omp.cc -o nbody_omp || { echo "OpenMP compilation failed"; exit 1; }

echo "Compiling MPI version (nbody_mpi.cc)..."
mpicxx -O3 -std=c++17 nbody_mpi.cc -o nbody_mpi || { echo "MPI compilation failed"; exit 1; }

echo "Compiling Hybrid MPI+OpenMP version (nbody_mpi_omp.cc)..."
mpicxx -fopenmp -O3 -std=c++17 nbody_mpi_omp.cc -o nbody_mpi_omp || { echo "Hybrid MPI+OpenMP compilation failed"; exit 1; }

if [ -f nbody_mpi_shared.cc ]; then
  echo "Compiling Shared-Memory MPI version (nbody_mpi_shared.cc)..."
  mpicxx -O3 -std=c++17 nbody_mpi_shared.cc -o nbody_mpi_shared || echo "Shared-Memory MPI compilation failed."
fi

# -------------------------
# Execution Loop
# -------------------------
# Define an array of N values to test.
N_values=(128 256 512 1024)

for N in "${N_values[@]}"; do
  echo "-----------------------------------------------"
  echo "Running Serial version with N = $N"
  ./nbody_serial $N > serial_${N}.txt
  
  echo "Running OpenMP version with N = $N (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
  ./nbody_omp $N > openmp_${N}.txt
  
  echo "Running MPI version with N = $N using 4 MPI ranks"
  mpiexec -n 4 ./nbody_mpi $N > mpi_${N}.txt
  
  echo "Running Hybrid MPI+OpenMP version with N = $N using 2 MPI ranks"
  mpiexec -n 2 ./nbody_mpi_omp $N > hybrid_${N}.txt
  
  if [ -f nbody_mpi_shared ]; then
    echo "Running Shared-Memory MPI version with N = $N using 4 MPI ranks"
    mpiexec -n 4 ./nbody_mpi_shared $N > shared_${N}.txt
  fi
done

echo "Job completed. Output files: serial_*.txt, openmp_*.txt, mpi_*.txt, hybrid_*.txt, shared_*.txt (if compiled)."
