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
static double *m, *x, *v, *a, *x_next, *v_next;
static MPI_Win win_m, win_x, win_v, win_a, win_x_next, win_v_next;

using Vec = std::vector<double>;

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
    if (rank == 0) {
        MPI_Win_allocate_shared(N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
    } else {
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
    }
    auto now = std::chrono::high_resolution_clock::now();
    auto now_int = std::chrono::time_point_cast<std::chrono::microseconds>(now).time_since_epoch().count();
    gen.seed(now_int ^ rank);
}

void initial_conditions() {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < N; ++i) m[i] = m_0;
    const double dx = x_max - x_min;
    const double dv = v_max - v_min;
    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = ran(gen) * dx + x_min;
        v[i] = ran(gen) * dv + v_min;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void acceleration() {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = ND_beg; i < ND_end; ++i) a[i] = 0.0;
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
    MPI_Barrier(MPI_COMM_WORLD);
}

void timestep() {
    acceleration();
    for (int i = ND_beg; i < ND_end; ++i) {
        v_next[i] = a[i] * dt + v[i];
        x_next[i] = v_next[i] * dt + x[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::swap(x, x_next);
    std::swap(v, v_next);
}

double kinetic_energy() {
    double KE_n = 0.;
    for (int i = N_beg; i < N_end; ++i) {
        double v2 = 0.;
        for (int j = 0; j < D; ++j) {
            const int k = i * D + j;
            v2 += v[k] * v[k];
        }
        KE_n += 0.5 * m[i] * v2;
    }
    double KE_total;
    MPI_Reduce(&KE_n, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return KE_total;
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
    Vec t(T + 1);
    for (int i = 0; i <= T; ++i) t[i] = double(i) * dt;
    Vec KE(T + 1);
    initial_conditions();
    for (int n = 0; n < T; ++n) {
        timestep();
        if (rank == 0) KE[n] = kinetic_energy();
    }
    if (rank == 0) KE[T] = kinetic_energy();
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
    MPI_Win_free(&win_v_next);
    MPI_Win_free(&win_x_next);
    MPI_Win_free(&win_a);
    MPI_Win_free(&win_v);
    MPI_Win_free(&win_x);
    MPI_Win_free(&win_m);
    MPI_Finalize();
    return 0;
}
