// nbody_omp.cc
// OpenMP parallelized n-body simulation
// Compile with: clang++ -fopenmp -O3 -std=c++17 nbody_omp.cc -o nbody_omp
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

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

// Generate initial conditions with OpenMP parallelization
std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
#ifdef _OPENMP
#pragma omp parallel
    {
        unsigned thread_seed = seed ^ omp_get_thread_num();
        std::mt19937 gen(thread_seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
#pragma omp for
        for (int i = 0; i < ND; i++) {
             x[i] = dist(gen);
             v[i] = 0.0;
        }
    }
#else
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < ND; i++) {
         x[i] = dist(gen);
         v[i] = 0.0;
    }
#endif
    return {x, v};
}

Vec acceleration(const Vec &x, const Vec &m) {
    Vec a(ND, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; i++) {
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < ND; i++) {
         v1[i] = v0[i] + a0[i] * dt;
         x1[i] = x0[i] + v1[i] * dt;
    }
    return {x1, v1};
}

int main(int argc, char **argv) {
    auto start = std::chrono::high_resolution_clock::now();
    if (argc > 1) {
         N = std::atoi(argv[1]);
         ND = N * D;
    }
    Vec t(T + 1);
    for (int i = 0; i <= T; i++)
         t[i] = i * dt;
    Vec m(N, m0);
    std::vector<Vec> x(T + 1), v(T + 1);
    std::tie(x[0], v[0]) = initial_conditions();
    for (int n = 0; n < T; n++)
         std::tie(x[n + 1], v[n + 1]) = timestep(x[n], v[n], m);
    Vec KE(T + 1, 0.0);
    for (int n = 0; n <= T; n++) {
         double KE_n = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:KE_n)
#endif
         for (int i = 0; i < N; i++) {
              double v2 = 0.0;
              int base = i * D;
              for (int k = 0; k < D; k++)
                   v2 += v[n][base + k] * v[n][base + k];
              KE_n += 0.5 * m[i] * v2;
         }
         KE[n] = KE_n;
    }
    save(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
    save(t, "time_" + std::to_string(N) + ".txt", "Time");
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "OpenMP simulation complete. Time: " << elapsed << " s for N = " << N << "\n";
    return 0;
}
