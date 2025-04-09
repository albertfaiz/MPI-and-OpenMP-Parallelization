// nbody_mpi_omp.cc
// Hybrid MPI+OpenMP version: MPI distributes the data among processes,
// and within each process, OpenMP parallelizes the loops.
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>
#include <cstdlib>

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
using Vecs = std::vector<Vec>;

// Global MPI variables for domain decomposition.
int rank, n_ranks;
std::vector<int> counts, displs;
std::vector<int> countsD, displsD;
int N_beg, N_end, N_local;
int ND_beg, ND_end, ND_local;

void setup_parallelism() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    counts.resize(n_ranks);
    displs.resize(n_ranks);
    countsD.resize(n_ranks);
    displsD.resize(n_ranks);
    int remainder = N % n_ranks;
    for (int i = 0; i < n_ranks; ++i) {
        counts[i] = N / n_ranks;
        displs[i] = i * counts[i];
        if (i < remainder) {
            counts[i] += 1;
            displs[i] += i;
        } else {
            displs[i] += remainder;
        }
        countsD[i] = counts[i] * D;
        displsD[i] = displs[i] * D;
    }
    N_beg = displs[rank];
    N_end = N_beg + counts[rank];
    ND_beg = N_beg * D;
    ND_end = N_end * D;
    N_local = counts[rank];
    ND_local = countsD[rank];
}

// Hybrid random generator: Each thread creates its own local generator.
double rand_val(int global_index, int tid) {
    std::mt19937 local_gen((unsigned) (std::chrono::steady_clock::now().time_since_epoch().count() ^ (tid * 1000 + global_index)));
    std::uniform_real_distribution<> dist(0.0, 1.0);
    return dist(local_gen);
}

std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    double dx = x_max - x_min;
    double dv = v_max - v_min;
#pragma omp parallel for
    for (int i = 0; i < ND; ++i) {
        int tid = omp_get_thread_num();
        x[i] = rand_val(i, tid) * dx + x_min;
        v[i] = rand_val(i, tid) * dv + v_min;
    }
    // Broadcast full state to all ranks.
    MPI_Bcast(x.data(), ND, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v.data(), ND, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return {x, v};
}

Vec acceleration(const Vec &x, const Vec &m) {
    Vec a(ND, 0.0);
#pragma omp parallel for
    for (int i = N_beg; i < N_end; ++i) {
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
            double factor = G * m[j] / denom;
            for (int k = 0; k < D; ++k) {
#pragma omp atomic
                a[iD + k] += factor * dx[k];
            }
        }
    }
    return a;
}

std::tuple<Vec, Vec> timestep(const Vec &x0, const Vec &v0, const Vec &m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
#pragma omp parallel for
    for (int i = N_beg; i < N_end; ++i) {
        int iD = i * D;
        for (int k = 0; k < D; ++k) {
            v1[iD + k] = v0[iD + k] + a0[iD + k] * dt;
            x1[iD + k] = x0[iD + k] + v1[iD + k] * dt;
        }
    }
    return {x1, v1};
}

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }
    setup_parallelism();
    Vec t(T+1);
    for (int i = 0; i <= T; ++i)
        t[i] = i * dt;
    Vec m(N, m_0);
    Vec x, v;
    std::tie(x, v) = initial_conditions();
    Vec KE(T+1, 0.0);
    for (int n = 0; n < T; ++n) {
        Vec x_new, v_new;
        std::tie(x_new, v_new) = timestep(x, v, m);
        // Update only our local portion
        for (int i = N_beg; i < N_end; ++i) {
            int iD = i * D;
            for (int k = 0; k < D; ++k) {
                x[iD + k] = x_new[i * D + k];
                v[iD + k] = v_new[i * D + k];
            }
        }
        double KE_local = 0.0;
        for (int i = N_beg; i < N_end; ++i) {
            double v2 = 0.0;
            int iD = i * D;
            for (int k = 0; k < D; ++k)
                v2 += v[iD + k] * v[iD + k];
            KE_local += 0.5 * m[i] * v2;
        }
        double KE_total = 0.0;
        MPI_Reduce(&KE_local, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            KE[n+1] = KE_total;
    }
    if (rank == 0) {
        save(KE, "KE_MPI_OMP_" + std::to_string(N) + ".txt", "Kinetic Energy (MPI+OpenMP)");
        save(t, "time_MPI_OMP_" + std::to_string(N) + ".txt", "Time");
        std::cout << "Total KE (initial): " << KE[0] << std::endl;
    }
    MPI_Finalize();
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000.0;
        std::cout << "Runtime = " << elapsed << " s for N = " << N << std::endl;
    }
    return 0;
}
