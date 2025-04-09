#include <mpi.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>
#include <thread>
#include <mutex>

static int N = 128;
static const int D = 3;
static int ND = N * D;
static const double G = 0.5;
static const double dt = 1e-3;
static const int T = 300;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;
static const int num_threads = 4;

static int rank, n_ranks;
static int N_beg, N_end, ND_beg, ND_end, N_local, ND_local;

using Vec = std::vector<double>;
using Vecs = std::vector<Vec>;

std::mt19937 gen;
std::uniform_real_distribution<> ran(0.0, 1.0);
std::mutex mtx;

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

void initial_conditions_thread(int start, int end, Vec& x, Vec& v, double dx, double dv) {
    std::mt19937 local_gen;
    local_gen.seed(gen() ^ (start + rank));
    std::uniform_real_distribution<> local_ran(0.0, 1.0);
    for (int i = start; i < end; ++i) {
        x[i] = local_ran(local_gen) * dx;
        v[i] = local_ran(local_gen) * dv;
    }
}

std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    double dx = 1.0, dv = 0.0;
    std::vector<std::thread> threads;
    int chunk_size = ND_local / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start = ND_beg + t * chunk_size;
        int end = (t == num_threads - 1) ? ND_end : start + chunk_size;
        threads.emplace_back(initial_conditions_thread, start, end, std::ref(x), std::ref(v), dx, dv);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    return {x, v};
}

void acceleration_thread(int start, int end, const Vec& x, const Vec& m, Vec& a) {
    for (int i = start; i < end; ++i) {
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
                a[iD + k] += factor * dx[k];
            }
        }
    }
}

Vec acceleration(const Vec& x, const Vec& m) {
    Vec a(ND, 0.0);
    std::vector<std::thread> threads;
    int chunk_size = (N_end - N_beg) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start = N_beg + t * chunk_size;
        int end = (t == num_threads - 1) ? N_end : start + chunk_size;
        threads.emplace_back(acceleration_thread, start, end, std::cref(x), std::cref(m), std::ref(a));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, a.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    return a;
}

void timestep_thread(int start, int end, const Vec& x0, const Vec& v0, const Vec& a0, Vec& x1, Vec& v1) {
    for (int i = start; i < end; ++i) {
        v1[i] = v0[i] + a0[i] * dt;
        x1[i] = x0[i] + v1[i] * dt;
    }
}

std::tuple<Vec, Vec> timestep(const Vec& x0, const Vec& v0, const Vec& m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
    std::vector<std::thread> threads;
    int chunk_size = ND_local / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start = ND_beg + t * chunk_size;
        int end = (t == num_threads - 1) ? ND_end : start + chunk_size;
        threads.emplace_back(timestep_thread, start, end, std::cref(x0), std::cref(v0), std::cref(a0), std::ref(x1), std::ref(v1));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x1.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v1.data(), ND_local, MPI_DOUBLE, MPI_COMM_WORLD);
    return {x1, v1};
}

void ke_thread(int start, int end, int n, const Vecs& v, const Vec& m, double& ke) {
    double local_ke = 0.0;
    for (int i = start; i < end; ++i) {
        double v2 = 0.0;
        for (int k = 0; k < D; ++k)
            v2 += v[n][i * D + k] * v[n][i * D + k];
        local_ke += 0.5 * m[i] * v2;
    }
    std::lock_guard<std::mutex> lock(mtx);
    ke += local_ke;
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
        save(t, "time_MPI_THREADED_" + std::to_string(N) + ".txt", "Time (MPI+Threads)");
    Vec m(N, 1.0);
    Vecs x(T+1), v(T+1);
    std::tie(x[0], v[0]) = initial_conditions();
    for (int n = 0; n < T; ++n)
        std::tie(x[n+1], v[n+1]) = timestep(x[n], v[n], m);
    Vec local_KE(T+1, 0.0);
    for (int n = 0; n <= T; ++n) {
        double ke = 0.0;
        std::vector<std::thread> threads;
        int chunk_size = (N_end - N_beg) / num_threads;
        for (int t = 0; t < num_threads; t++) {
            int start = N_beg + t * chunk_size;
            int end = (t == num_threads - 1) ? N_end : start + chunk_size;
            threads.emplace_back(ke_thread, start, end, n, std::cref(v), std::cref(m), std::ref(ke));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        local_KE[n] = ke;
    }
    Vec global_KE(T+1, 0.0);
    MPI_Reduce(local_KE.data(), global_KE.data(), T+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        save(global_KE, "KE_MPI_THREADED_" + std::to_string(N) + ".txt", "Kinetic Energy (MPI+Threads)");
        std::cout << "Total KE (first timestep) = " << global_KE[0] << "\n";
    }
    MPI_Finalize();
    return 0;
}