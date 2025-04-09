#!/bin/bash
#SBATCH --job-name=nbody_simulation
#SBATCH --output=nbody_%j.out
#SBATCH --error=nbody_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32GB

# Load modules
module purge
module load gcc
module load openmpi

# Run the benchmark script
chmod +x run_benchmarks.sh
./run_benchmarks.sh

# Generate a simple summary of runtimes
echo "Summary of Runtimes:" > summary.txt
grep "Runtime" *.out >> summary.txt