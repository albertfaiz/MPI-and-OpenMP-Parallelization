// nbody_omp.cc
// OpenMP parallelized version of the N-body simulation.
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <omp.h>

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

// Global seed (only used to compute per-thread seeds)
static const unsigned base_seed = 12345;

// Save a vector to a file.
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
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// Generate initial conditions in parallel.
std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    double dx = x_max - x_min;
    double dv = v_max - v_min;
#pragma omp parallel for
    for (int i = 0; i < ND; ++i) {
        int tid = omp_get_thread_num();
        std::mt19937 local_gen(base_seed + tid + i); // combine thread id and index for variability
        x[i] = (local_gen() / (double)local_gen.max()) * dx + x_min;
        v[i] = (local_gen() / (double)local_gen.max()) * dv + v_min;
    }
    return {x, v};
}

// Compute acceleration in parallel.
Vec acceleration(const Vec &x, const Vec &m) {
    Vec a(ND, 0.0);
#pragma omp parallel for
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
            double factor = G * m[j] / denom;
            for (int k = 0; k < D; ++k)
#pragma omp atomic
                a[iD + k] += factor * dx[k];
        }
    }
    return a;
}

// Compute one time step update.
std::tuple<Vec, Vec> timestep(const Vec &x0, const Vec &v0, const Vec &m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
#pragma omp parallel for
    for (int i = 0; i < ND; ++i) {
        v1[i] = v0[i] + a0[i] * dt;
        x1[i] = x0[i] + v1[i] * dt;
    }
    return {x1, v1};
}

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    if (argc > 1) {
        N = std::atoi(argv[1]);
        ND = N * D;
    }
    Vec t(T+1);
    for (int i = 0; i <= T; ++i)
        t[i] = i * dt;
    Vec m(N, m_0);
    Vecs x(T+1), v(T+1);
    std::tie(x[0], v[0]) = initial_conditions();
    for (int n = 0; n < T; ++n) {
        std::tie(x[n+1], v[n+1]) = timestep(x[n], v[n], m);
    }
    Vec KE(T+1, 0.0);
    for (int n = 0; n <= T; ++n) {
        double KE_n = 0.0;
        for (int i = 0; i < N; ++i) {
            double v2 = 0.0;
            for (int j = 0; j < D; ++j)
                v2 += v[n][i*D + j] * v[n][i*D + j];
            KE_n += 0.5 * m[i] * v2;
        }
        KE[n] = KE_n;
    }
    save(KE, "KE_OMP_" + std::to_string(N) + ".txt", "Kinetic Energy (OpenMP)");
    save(t, "time_OMP_" + std::to_string(N) + ".txt", "Time");
    std::cout << "Total Kinetic Energy (initial): " << KE[0] << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000.0;
    std::cout << "Runtime = " << elapsed << " s for N = " << N << std::endl;
    return 0;
}
