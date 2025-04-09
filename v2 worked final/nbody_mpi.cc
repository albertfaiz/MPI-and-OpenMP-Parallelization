#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <mpi.h>

static int N = 1024;
static const int D = 3;
static int ND = N * D;
static const double G = 0.5;
static const double dt = 5e-4;
static const int T = 5000;
static const double t_max = static_cast<double>(T) * dt;
static const double x_min = 0., x_max = 1.;
static const double v_min = 0., v_max = 0.;
static const double m_0 = 1.;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;

static int rank, n_ranks;
static int N_beg, N_end, N_local;
static int ND_beg, ND_end, ND_local;
static std::vector<int> counts, displs, countsD, displsD;

using Vec = std::vector<double>;
using Vecs = std::vector<Vec>;

static std::mt19937 gen;
static std::uniform_real_distribution<> ran(0., 1.);

template <typename T>
void save(const std::vector<T>& vec, const std::string& filename, const std::string& header = " ") {
    std::ofstream file(filename);
    if (file.is_open()) {
        if (!header.empty()) file << "# " << header << std::endl;
        for (const auto& elem : vec) file << elem << " ";
        file << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

void setup_parallelism() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    N_local = N / n_ranks;
    int remainder = N % n_ranks;
    N_beg = rank * N_local + std::min(rank, remainder);
    N_end = N_beg + N_local + (rank < remainder ? 1 : 0);
    N_local = N_end - N_beg;
    ND_beg = N_beg * D;
    ND_end = N_end * D;
    ND_local = ND_end - ND_beg;
    counts.resize(n_ranks);
    displs.resize(n_ranks);
    countsD.resize(n_ranks);
    displsD.resize(n_ranks);
    for (int i = 0; i < n_ranks; ++i) {
        int local_N = N / n_ranks + (i < remainder ? 1 : 0);
        counts[i] = local_N;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        countsD[i] = local_N * D;
        displsD[i] = displs[i] * D;
    }
    auto now = std::chrono::high_resolution_clock::now();
    auto now_int = std::chrono::time_point_cast<std::chrono::microseconds>(now).time_since_epoch().count();
    gen.seed(now_int ^ rank);
}

std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    const double dx = x_max - x_min;
    const double dv = v_max - v_min;
    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = ran(gen) * dx + x_min;
        v[i] = ran(gen) * dv + v_min;
    }
    MPI_Allgatherv(MPI_IN_PLACE, ND_local, MPI_DOUBLE, x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, ND_local, MPI_DOUBLE, v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return std::make_tuple(x, v);
}

Vec acceleration(const Vec& x, const Vec& m) {
    Vec a(ND, 0.0);
    for (int i = N_beg; i < N_end; ++i) {
        const int iD = i * D;
        for (int j = 0; j < N; ++j) {
            const int jD = j * D;
            double dx[D], dx2 = epsilon2;
            for (int k = 0; k < D; ++k) {
                dx[k] = x[jD + k] - x[iD + k];
                dx2 += dx[k] * dx[k];
            }
            const double Gm_dx3 = G * m[j] / (dx2 * std::sqrt(dx2));
            for (int k = 0; k < D; ++k) {
                a[iD + k] += Gm_dx3 * dx[k];
            }
        }
    }
    MPI_Allgatherv(MPI_IN_PLACE, ND_local, MPI_DOUBLE, a.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return a;
}

std::tuple<Vec, Vec> timestep(const Vec& x0, const Vec& v0, const Vec& m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
    for (int i = ND_beg; i < ND_end; ++i) {
        v1[i] = a0[i] * dt + v0[i];
        x1[i] = v1[i] * dt + x0[i];
    }
    MPI_Allgatherv(MPI_IN_PLACE, ND_local, MPI_DOUBLE, x1.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, ND_local, MPI_DOUBLE, v1.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return std::make_tuple(x1, v1);
}

int main(int argc, char** argv) {
    setup_parallelism();
    auto start = std::chrono::high_resolution_clock::now();
    if (argc > 1 && rank == 0) {
        N = std::atoi(argv[1]);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ND = N * D;
    N_local = N / n_ranks;
    int remainder = N % n_ranks;
    N_beg = rank * N_local + std::min(rank, remainder);
    N_end = N_beg + N_local + (rank < remainder ? 1 : 0);
    N_local = N_end - N_beg;
    ND_beg = N_beg * D;
    ND_end = N_end * D;
    ND_local = ND_end - ND_beg;
    counts.resize(n_ranks);
    displs.resize(n_ranks);
    countsD.resize(n_ranks);
    displsD.resize(n_ranks);
    for (int i = 0; i < n_ranks; ++i) {
        int local_N = N / n_ranks + (i < remainder ? 1 : 0);
        counts[i] = local_N;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        countsD[i] = local_N * D;
        displsD[i] = displs[i] * D;
    }
    Vec t(T + 1);
    for (int i = 0; i <= T; ++i) t[i] = double(i) * dt;
    Vec m(N, m_0);
    Vecs x(T + 1), v(T + 1);
    std::tie(x[0], v[0]) = initial_conditions();
    for (int n = 0; n < T; ++n) {
        std::tie(x[n + 1], v[n + 1]) = timestep(x[n], v[n], m);
    }
    Vec KE(T + 1);
    for (int n = 0; n <= T; ++n) {
        double KE_n = 0.;
        for (int i = N_beg; i < N_end; ++i) {
            double v2 = 0.;
            for (int j = 0; j < D; ++j) {
                const int k = i * D + j;
                v2 += v[n][k] * v[n][k];
            }
            KE_n += 0.5 * m[i] * v2;
        }
        if (rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, &KE_n, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&KE_n, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        KE[n] = KE_n;
    }
    if (rank == 0) {
        save(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
        save(t, "time_" + std::to_string(N) + ".txt", "Time");
        std::cout << "Total Kinetic Energy = [" << KE[0];
        const int Tskip = T / 50;
        for (int n = 1; n <= T; n += Tskip) std::cout << " " << KE[n];
        std::cout << "]" << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.;
    if (rank == 0) std::cout << "Runtime = " << elapsed << " s for N = " << N << std::endl;
    MPI_Finalize();
    return 0;
}
