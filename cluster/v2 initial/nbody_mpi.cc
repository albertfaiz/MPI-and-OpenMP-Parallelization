#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <mpi.h>

// Global constants
static int N = 128;
static const int D = 3;
static int ND = N * D;
static const double G = 0.5;
static const double dt = 1e-3;
static const int T = 300;
static const double t_max = static_cast<double>(T) * dt;
static const double x_min = 0.;
static const double x_max = 1.;
static const double v_min = 0.;
static const double v_max = 0.;
static const double m_0 = 1.;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;

using Vec = std::vector<double>;
using Vecs = std::vector<Vec>;

// MPI variables
static int rank, n_ranks;
static std::vector<int> counts, displs;
static std::vector<int> countsD, displsD;
static int N_beg, N_end, N_local;
static int ND_beg, ND_end, ND_local;

// Random number generator
static std::mt19937 gen;
static std::uniform_real_distribution<> ran(0., 1.);

// Print a vector to a file
template <typename T>
void save(const std::vector<T>& vec, const std::string& filename, const std::string& header = " ") {
    std::ofstream file(filename);
    if (file.is_open()) {
        if (!header.empty())
            file << "# " << header << std::endl;
        for (const auto& elem : vec)
            file << elem << " ";
        file << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// Set up parallelism
void setup_parallelism() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    auto now = std::chrono::high_resolution_clock::now();
    auto now_cast = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto now_int = now_cast.time_since_epoch().count();
    gen.seed(now_int ^ rank);

    counts.resize(n_ranks);
    displs.resize(n_ranks);
    countsD.resize(n_ranks);
    displsD.resize(n_ranks);
    const int remainder = N % n_ranks;
    for (int i = 0; i < n_ranks; ++i) {
        counts[i] = N / n_ranks;
        if (i < remainder) counts[i] += 1;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        countsD[i] = counts[i] * D;
        displsD[i] = displs[i] * D;
    }
    N_beg = displs[rank];
    N_end = N_beg + counts[rank];
    ND_beg = N_beg * D;
    ND_end = N_end * D;
    N_local = N_end - N_beg;
    ND_local = ND_end - ND_beg;
}

// Generate random initial conditions for N masses
std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    const double dx = x_max - x_min;
    const double dv = v_max - v_min;
    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = ran(gen) * dx + x_min;
        v[i] = ran(gen) * dv + v_min;
    }
    if (n_ranks > 1) {
        MPI_Allgatherv(x.data() + ND_beg, ND_local, MPI_DOUBLE, x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(v.data() + ND_beg, ND_local, MPI_DOUBLE, v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    return {x, v};
}

// Compute the acceleration of all masses
Vec acceleration(const Vec& x, const Vec& m) {
    Vec a(ND, 0.0);
    for (int i = N_beg; i < N_end; ++i) {
        const int iD = i * D;
        double dx[D];
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            const int jD = j * D;
            double dx2 = epsilon2;
            for (int k = 0; k < D; ++k) {
                dx[k] = x[jD + k] - x[iD + k];
                dx2 += dx[k] * dx[k];
            }
            const double G_m_dx3 = G * m[j] / (dx2 * sqrt(dx2));
            for (int k = 0; k < D; ++k) {
                const int iDk = iD + k;
                a[iDk] += G_m_dx3 * dx[k];
            }
        }
    }
    if (n_ranks > 1) {
        MPI_Allgatherv(a.data() + ND_beg, ND_local, MPI_DOUBLE, a.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    return a;
}

// Compute the next position and velocity for all masses
std::tuple<Vec, Vec> timestep(const Vec& x0, const Vec& v0, const Vec& m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
    for (int i = ND_beg; i < ND_end; ++i) {
        v1[i] = a0[i] * dt + v0[i];
        x1[i] = v1[i] * dt + x0[i];
    }
    if (n_ranks > 1) {
        MPI_Allgatherv(x1.data() + ND_beg, ND_local, MPI_DOUBLE, x1.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(v1.data() + ND_beg, ND_local, MPI_DOUBLE, v1.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    return {x1, v1};
}

// Main function
int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }

    setup_parallelism();

    // Prepare vectors
    Vec t(T + 1);
    for (int i = 0; i <= T; ++i)
        t[i] = static_cast<double>(i) * dt;

    Vec m(N, m_0);
    Vecs x(T + 1), v(T + 1);
    std::tie(x[0], v[0]) = initial_conditions();

    // Simulate
    for (int n = 0; n < T; ++n) {
        std::tie(x[n + 1], v[n + 1]) = timestep(x[n], v[n], m);
    }

    // Calculate kinetic energy
    Vec KE(T + 1);
    for (int n = 0; n <= T; ++n) {
        double KE_n = 0.0;
        auto& v_n = v[n];
        for (int i = N_beg; i < N_end; ++i) {
            double v2 = 0.0;
            for (int j = 0; j < D; ++j) {
                const int k = i * D + j;
                v2 += v_n[k] * v_n[k];
            }
            KE_n += 0.5 * m[i] * v2;
        }
        KE[n] = KE_n;
    }

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, KE.data(), T + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        save(KE, "KE_MPI_" + std::to_string(N) + ".txt", "Kinetic Energy");
        save(t, "time_MPI_" + std::to_string(N) + ".txt", "Time");

        std::cout << "Total Kinetic Energy = [" << KE[0];
        const int Tskip = T / 50;
        for (int n = 1; n <= T; n += Tskip)
            std::cout << " " << KE[n];
        std::cout << "]" << std::endl;
    } else {
        MPI_Reduce(KE.data(), NULL, T + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    if (rank == 0) {
        std::cout << "Runtime = " << elapsed << " s for N = " << N << " with " << n_ranks << " ranks" << std::endl;
    }

    MPI_Finalize();
    return 0;
}