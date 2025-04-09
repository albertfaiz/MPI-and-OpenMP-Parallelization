# MPI-and-OpenMP-Parallelization


This project investigates the parallelization of an N-body simulation across three
distinct platforms: a small-scale HPC cluster, a large-scale HPC cluster, and a local
Mac M2 processor. Using serial, OpenMP, MPI, hybrid MPI+OpenMP, and shared-
memory MPI implementations, we simulate gravitational interactions among N
particles, benchmarking performance for N ranging from 128 to 8192 over 300 to
5000 time steps. The hybrid MPI+OpenMP approach consistently delivers the best
speedup, achieving up to 3.6x on the small-scale HPC and 10x on the large-scale
HPC for N = 1024 and N = 4096, respectively. Kinetic energy evolution is ana-
lyzed to validate correctness, with detailed visualizations including full data plots,
rolling averages, and segment-specific analyses. Runtime comparisons and scaling
plots highlight the trade-offs of each platform and implementation. This study un-
derscores the power of HPC techniques in computational physics and provides a
foundation for optimizing N-body simulations across diverse systems.
