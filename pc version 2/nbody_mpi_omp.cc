// nbody_mpi_omp.cc
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>

static int N = 128;
static const int D = 3;
static int ND = N * D;
static const double G = 0.5;
static const double dt = 1e-3;
static const int T = 300;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;

static int rank, n_ranks;
static int N_beg, N_end, ND_beg, ND_end, N_local, ND_local;

using Vec = std::vector<double>;
using Vecs = std::vector<Vec>;

std::mt19937 gen;
std::uniform_real_distribution<> ran(0.0, 1.0);

template <typename T>
void save(const std::vector<T>& vec, const std::string& filename, const std::string& header = "") {
    std::ofstream file(filename);
    if (file.is_open()) {
        if (!header.empty())
            file << "# " << header << "\n";
        for (const auto& elem : vec)
            file << elem << " ";
        file << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << "\n";
    }
}

void setup_parallelism() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    N_local = N / n_ranks + (rank < N % n_ranks ? 1 : 0);
    N_beg = rank * (N / n_ranks) + (rank < N % n_ranks ? rank : N % n_ranks);
    N_end = N_beg + N_local;
    ND_beg = N_beg * D;
    ND_end = N_end * D;
    ND_local = ND_end - ND_beg;
}

std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    double dx = 1.0, dv = 0.0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937 local_gen;
        local_gen.seed(gen() ^ (tid * n_ranks + rank));
        std::uniform_real_distribution<> local_ran(0.0, 1.0);
        #pragma omp for schedule(dynamic)
        for (int i = ND_beg; i < ND_end; ++i) {
            x[i] = local_ran(local_gen) * dx;
            v[i] = local_ran(local_gen) * dv;
        }
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    return {x, v};
}

Vec acceleration(const Vec& x, const Vec& m) {
    Vec a(ND, 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (int i = N_beg; i < N_end; ++i) {
        int iD = i * D;
        for (int j = 0; j < N; ++j) {
            int jD = j * D;
            double dx[D];
            double dx2 = epsilon2;
            for (int k = 0; k < D; ++k) {
                dx[k] = x[jD + k] - x[iD + k];
                dx2 += dx[k] * dx[k];
            }
            double factor = G / (dx2 * std::sqrt(dx2));
            for (int k = 0; k < D; ++k) {
                #pragma omp atomic
                a[iD + k] += factor * dx[k];
            }
        }
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, a.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    return a;
}

std::tuple<Vec, Vec> timestep(const Vec& x0, const Vec& v0, const Vec& m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
    #pragma omp parallel for
    for (int i = ND_beg; i < ND_end; ++i) {
        v1[i] = v0[i] + a0[i] * dt;
        x1[i] = x0[i] + v1[i] * dt;
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x1.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v1.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    return {x1, v1};
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    setup_parallelism();
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    gen.seed(seed ^ rank);
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
        setup_parallelism();
    }
    Vec t(T+1);
    for (int i = 0; i <= T; ++i)
        t[i] = i * dt;
    if (rank == 0)
        save(t, "time_MPI_OMP_" + std::to_string(N) + ".txt", "Time (MPI+OpenMP)");
    Vec m(N, 1.0);
    Vecs x(T+1), v(T+1);
    std::tie(x[0], v[0]) = initial_conditions();
    for (int n = 0; n < T; ++n)
        std::tie(x[n+1], v[n+1]) = timestep(x[n], v[n], m);
    Vec local_KE(T+1, 0.0);
    for (int n = 0; n <= T; ++n) {
        double ke = 0.0;
        #pragma omp parallel for reduction(+:ke)
        for (int i = N_beg; i < N_end; ++i) {
            double v2 = 0.0;
            for (int k = 0; k < D; ++k)
                v2 += v[n][i * D + k] * v[n][i * D + k];
            ke += 0.5 * m[i] * v2;
        }
        local_KE[n] = ke;
    }
    Vec global_KE(T+1, 0.0);
    MPI_Reduce(local_KE.data(), global_KE.data(), T+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        save(global_KE, "KE_MPI_OMP_" + std::to_string(N) + ".txt", "Kinetic Energy (MPI+OpenMP)");
        std::cout << "Total KE (first timestep) = " << global_KE[0] << "\n";
    }
    MPI_Finalize();
    return 0;
}