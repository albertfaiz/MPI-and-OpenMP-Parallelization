// nbody_mpi_shared.cc
// MPI Shared-Memory version of the N-body simulation.
// This version allocates shared windows for positions and velocities.
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cassert>

static int N = 128;
static const int D = 3;
static int ND = N * D;
static const double G = 0.5;
static const double dt = 1e-3;
static const int T = 300;
static const double t_max = T * dt;
static const double x_min = 0.;
static const double x_max = 1.;
static const double v_min = 0.;
static const double v_max = 0.;
static const double m_0 = 1.;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;

using Vec = std::vector<double>;

// Shared arrays pointers and MPI windows.
double *x_shared = nullptr, *v_shared = nullptr;
MPI_Win win_x, win_v;

// Only rank 0 allocates shared memory; then all processes attach.
void allocate_shared_memory(int total_size) {
    MPI_Win_allocate_shared(total_size * sizeof(double), sizeof(double),
                              MPI_INFO_NULL, MPI_COMM_WORLD, &x_shared, &win_x);
    MPI_Win_allocate_shared(total_size * sizeof(double), sizeof(double),
                              MPI_INFO_NULL, MPI_COMM_WORLD, &v_shared, &win_v);
}

// Initialize shared memory (only rank 0 writes, then barrier ensures others see it).
void init_shared_memory() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> ran(0., 1.);
        double dx = x_max - x_min;
        double dv = v_max - v_min;
        for (int i = 0; i < ND; ++i) {
            x_shared[i] = ran(gen) * dx + x_min;
            v_shared[i] = ran(gen) * dv + v_min;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// Compute acceleration using the shared x_shared array.
void compute_acceleration(const double *x, double *a) {
    // Set a[] = 0 first.
    for (int i = 0; i < ND; ++i)
        a[i] = 0.0;
    for (int i = 0; i < N; ++i) {
        int iD = i * D;
        double dx[D];
        for (int j = 0; j < N; ++j) {
            int jD = j * D;
            double dx2 = epsilon2;
            for (int k = 0; k < D; ++k) {
                dx[k] = x[jD + k] - x[iD + k];
                dx2 += dx[k] * dx[k];
            }
            double denom = dx2 * std::sqrt(dx2);
            double factor = G * m_0 / denom; // masses are all m_0
            for (int k = 0; k < D; ++k)
                a[iD + k] += factor * dx[k];
        }
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }
    // Allocate shared memory windows (all processes share the same arrays)
    allocate_shared_memory(ND);
    init_shared_memory();
    // For simplicity, we simulate on the shared arrays.
    // We allocate a temporary acceleration array.
    std::vector<double> a(ND, 0.0);
    // Time loop: do T time steps.
    for (int t_step = 0; t_step < T; ++t_step) {
        // Compute acceleration into a[].
        compute_acceleration(x_shared, a.data());
        // Update velocities and positions.
        for (int i = 0; i < ND; ++i) {
            v_shared[i] += a[i] * dt;
            x_shared[i] += v_shared[i] * dt;
        }
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize after each time step.
    }
    // Compute kinetic energy (sum over all masses).
    double KE_local = 0.0;
    for (int i = 0; i < N; ++i) {
        double v2 = 0.0;
        int iD = i * D;
        for (int k = 0; k < D; ++k)
            v2 += v_shared[iD + k] * v_shared[iD + k];
        KE_local += 0.5 * m_0 * v2;
    }
    double KE_total = 0.0;
    MPI_Reduce(&KE_local, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << "Shared-memory update complete.\nTotal KE (final): " << KE_total << std::endl;
    // Free windows
    MPI_Win_free(&win_v);
    MPI_Win_free(&win_x);
    MPI_Finalize();
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000.0;
        std::cout << "Runtime = " << elapsed << " s for N = " << N << std::endl;
    }
    return 0;
}
