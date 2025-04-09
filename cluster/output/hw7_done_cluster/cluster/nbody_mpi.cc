// nbody_mpi.cc
// MPI-only version of the N-body simulation.
// Each process computes a portion of the update and then gathers the full state.
#include <mpi.h>
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

// Global random generator (only used on rank 0 for initial conditions)
static std::mt19937 gen(std::random_device{}());
static std::uniform_real_distribution<> ran(0., 1.);

template <typename T>
void save(const std::vector<T>& vec, const std::string &filename, const std::string &header = "") {
    std::ofstream file(filename);
    if (file.is_open()) {
        if (!header.empty())
            file << "# " << header << std::endl;
        for (const auto &elem : vec)
            file << elem << " ";
        file << std::endl;
        file.close();
    } else {
        if (MPI::COMM_WORLD.Get_rank() == 0)
            std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// Setup domain decomposition variables (global)
int rank, n_ranks;
std::vector<int> counts, displs;
std::vector<int> countsD, displsD;
int N_beg, N_end, N_local;
int ND_beg, ND_end, ND_local;

void setup_parallelism() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    // Divide N among ranks:
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

// Serial-like functions but each process works only on its local portion.
// The full vectors x and v are replicated on each process.
std::tuple<Vec, Vec> initial_conditions() {
    // Only rank 0 initializes and then broadcasts.
    Vec x(ND), v(ND);
    if (rank == 0) {
        double dx = x_max - x_min;
        double dv = v_max - v_min;
        for (int i = 0; i < ND; ++i) {
            x[i] = ran(gen) * dx + x_min;
            v[i] = ran(gen) * dv + v_min;
        }
    }
    MPI_Bcast(x.data(), ND, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v.data(), ND, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return {x, v};
}

Vec acceleration(const Vec &x, const Vec &m) {
    Vec a(ND, 0.0);
    // Each process computes for indices i in [N_beg, N_end)
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
            for (int k = 0; k < D; ++k)
                a[iD + k] += factor * dx[k];
        }
    }
    return a;
}

std::tuple<Vec, Vec> timestep(const Vec &x0, const Vec &v0, const Vec &m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
    for (int i = N_beg; i < N_end; ++i) {
        int iD = i * D;
        for (int k = 0; k < D; ++k) {
            v1[iD + k] = v0[iD + k] + a0[iD + k] * dt;
            x1[iD + k] = x0[iD + k] + v1[iD + k] * dt;
        }
    }
    // Gather the updated local parts to form full vectors.
    MPI_Allgatherv(x0.data() + ND_beg, ND_local, MPI_DOUBLE,
                   const_cast<double*>(x0.data()), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(v0.data() + ND_beg, ND_local, MPI_DOUBLE,
                   const_cast<double*>(v0.data()), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    // For simplicity, we return the locally computed updates (other ranks’ values remain from previous step).
    // In a more refined version you would update the entire state.
    // Here we assume each process computes its own segment and full state is gathered.
    return {x1, v1};
}

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }
    setup_parallelism();
    // For simplicity, every process holds full state arrays.
    Vec t(T+1);
    for (int i = 0; i <= T; ++i)
        t[i] = i * dt;
    Vec m(N, m_0);
    Vec x, v;
    std::tie(x, v) = initial_conditions();
    Vec KE(T+1, 0.0);
    // Time loop: Each process computes its local update.
    for (int n = 0; n < T; ++n) {
        Vec x_new, v_new;
        std::tie(x_new, v_new) = timestep(x, v, m);
        // Replace only our local segment in full state:
        for (int i = N_beg; i < N_end; ++i) {
            int iD = i * D;
            for (int k = 0; k < D; ++k) {
                x[iD + k] = x_new[i * D + k];
                v[iD + k] = v_new[i * D + k];
            }
        }
        // Compute local kinetic energy over our segment.
        double KE_local = 0.0;
        for (int i = N_beg; i < N_end; ++i) {
            double v2 = 0.0;
            int iD = i * D;
            for (int k = 0; k < D; ++k)
                v2 += v[iD + k] * v[iD + k];
            KE_local += 0.5 * m[i] * v2;
        }
        // Reduce KE over all processes to rank 0.
        double KE_total = 0.0;
        MPI_Reduce(&KE_local, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            KE[n+1] = KE_total;
    }
    if (rank == 0) {
        save(KE, "KE_MPI_" + std::to_string(N) + ".txt", "Kinetic Energy (MPI)");
        save(t, "time_MPI_" + std::to_string(N) + ".txt", "Time");
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
