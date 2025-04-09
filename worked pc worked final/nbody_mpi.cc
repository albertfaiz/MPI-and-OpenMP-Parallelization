// nbody_mpi.cc
// MPI-only parallel n-body simulation
// Compile with: mpicxx -O3 -std=c++17 nbody_mpi.cc -o nbody_mpi
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdlib>
#include <mpi.h>

static int N = 128;
static const int D = 3;
static int ND = N * D;
static const double G = 0.5;
static const double dt = 1e-3;
static const int T = 300;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;
static const double m0 = 1.0;

using Vec = std::vector<double>;

// MPI domain decomposition variables
int rank, n_ranks;
int local_N, local_start;
std::vector<int> counts, displs;
std::vector<int> countsD, displsD;

void setup_domain_decomposition(int N_total) {
    counts.resize(n_ranks);
    displs.resize(n_ranks);
    countsD.resize(n_ranks);
    displsD.resize(n_ranks);
    int base = N_total / n_ranks;
    int remainder = N_total % n_ranks;
    for (int i = 0; i < n_ranks; i++) {
         counts[i] = base + (i < remainder ? 1 : 0);
         displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
         countsD[i] = counts[i] * D;
         displsD[i] = displs[i] * D;
    }
    local_N = counts[rank];
    local_start = displs[rank];
}

template <typename T>
void save(const std::vector<T>& vec, const std::string & filename, const std::string & header = "") {
    std::ofstream file(filename);
    if (file.is_open()) {
        if (!header.empty())
            file << "# " << header << "\n";
        for (const auto &elem : vec)
            file << elem << " ";
        file << "\n";
        file.close();
    } else {
        std::cerr << "Error opening file " << filename << "\n";
    }
}

void initialize_conditions(Vec &x, Vec &v) {
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    seed ^= rank;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = local_start * D; i < (local_start + local_N) * D; i++) {
         x[i] = dist(gen);
         v[i] = 0.0;
    }
}

Vec acceleration(const Vec &x, const Vec &m) {
    Vec a(ND, 0.0);
    for (int i = local_start; i < local_start + local_N; i++) {
         int iD = i * D;
         for (int j = 0; j < N; j++) {
              int jD = j * D;
              double dx[D];
              double dist2 = epsilon2;
              for (int k = 0; k < D; k++) {
                   dx[k] = x[jD + k] - x[iD + k];
                   dist2 += dx[k] * dx[k];
              }
              double inv_dist3 = 1.0 / (dist2 * sqrt(dist2));
              double factor = G * m[j] * inv_dist3;
              for (int k = 0; k < D; k++) {
                   a[iD + k] += factor * dx[k];
              }
         }
    }
    return a;
}

std::tuple<Vec, Vec> timestep(const Vec &x0, const Vec &v0, const Vec &m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
    for (int i = local_start * D; i < (local_start + local_N) * D; i++) {
         v1[i] = v0[i] + a0[i] * dt;
         x1[i] = x0[i] + v1[i] * dt;
    }
    return {x1, v1};
}

double compute_kinetic_energy(const Vec &v, const Vec &m) {
    double KE_local = 0.0;
    for (int i = local_start; i < local_start + local_N; i++) {
         double v2 = 0.0;
         int base = i * D;
         for (int k = 0; k < D; k++)
              v2 += v[base + k] * v[base + k];
         KE_local += 0.5 * m[i] * v2;
    }
    return KE_local;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (argc > 1) {
         N = std::atoi(argv[1]);
         ND = N * D;
    }
    setup_domain_decomposition(N);

    Vec t(T + 1);
    for (int i = 0; i <= T; i++)
         t[i] = i * dt;
    Vec m(N, m0);
    Vec x(ND, 0.0), v(ND, 0.0);
    // Each process initializes its own portion.
    initialize_conditions(x, v);
    // Gather the full vectors so all processes have complete state.
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    Vec KE(T + 1, 0.0);
    double KE_local = compute_kinetic_energy(v, m);
    double KE_total = 0.0;
    MPI_Reduce(&KE_local, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) KE[0] = KE_total;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int t_step = 0; t_step < T; t_step++) {
         Vec new_x, new_v;
         std::tie(new_x, new_v) = timestep(x, v, m);
         for (int i = local_start * D; i < (local_start + local_N) * D; i++) {
              x[i] = new_x[i];
              v[i] = new_v[i];
         }
         MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
         MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
         KE_local = compute_kinetic_energy(v, m);
         MPI_Reduce(&KE_local, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
         if (rank == 0) KE[t_step + 1] = KE_total;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    if (rank == 0) {
         std::cout << "MPI simulation complete. Time: " << elapsed << " s for N = " << N << "\n";
         save(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
         save(t, "time_" + std::to_string(N) + ".txt", "Time");
    }
    MPI_Finalize();
    return 0;
}
