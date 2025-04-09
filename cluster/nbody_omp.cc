// nbody_omp.cc - OpenMP Version
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>
#include <omp.h>

static int N = 128;
static const int D = 3;
static int ND = N * D;
static const double G = 0.5;
static const double dt = 1e-3;
static const int T = 300;
static const double t_max = T * dt;
static const double x_min = 0.0, x_max = 1.0;
static const double v_min = 0.0, v_max = 0.0;
static const double m_0 = 1.0;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;

using Vec = std::vector<double>;
using Vecs = std::vector<Vec>;

static std::mt19937 gen(12345); // Fixed seed for reproducibility
static std::uniform_real_distribution<> ran(0.0, 1.0);

template <typename T>
void save(const std::vector<T>& vec, const std::string& filename, const std::string& header = ""){
    std::ofstream file(filename);
    if(file.is_open()){
        if(!header.empty())
            file << "# " << header << std::endl;
        for(auto &elem : vec)
            file << elem << " ";
        file << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    double dx = x_max - x_min, dv = v_max - v_min;
    #pragma omp parallel for
    for (int i = 0; i < ND; ++i) {
        // Each thread uses a local seed based on its index and thread number.
        unsigned local_seed = 12345 + i + omp_get_thread_num();
        std::mt19937 local_gen(local_seed);
        x[i] = ran(local_gen) * dx + x_min;
        v[i] = ran(local_gen) * dv + v_min;
    }
    return {x, v};
}

Vec acceleration(const Vec& x, const Vec& m) {
    Vec a(ND, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int iD = i * D;
        double diff[D];
        for (int j = 0; j < N; ++j) {
            int jD = j * D;
            double r2 = epsilon2;
            for (int k = 0; k < D; ++k) {
                diff[k] = x[jD + k] - x[iD + k];
                r2 += diff[k] * diff[k];
            }
            double denom = r2 * std::sqrt(r2);
            double factor = G * m[j] / denom;
            for (int k = 0; k < D; ++k) {
                a[iD + k] += factor * diff[k];
            }
        }
    }
    return a;
}

std::tuple<Vec, Vec> timestep(const Vec& x0, const Vec& v0, const Vec& m) {
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
    double start = omp_get_wtime();
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
        double ke = 0.0;
        for (int i = 0; i < N; ++i) {
            double v2 = 0.0;
            for (int j = 0; j < D; ++j) {
                v2 += v[n][i*D + j] * v[n][i*D + j];
            }
            ke += 0.5 * m[i] * v2;
        }
        KE[n] = ke;
    }
    save(KE, "KE_OMP_" + std::to_string(N) + ".txt", "Kinetic Energy (OpenMP)");
    save(t, "time_OMP_" + std::to_string(N) + ".txt", "Time");
    std::cout << "Total Kinetic Energy (initial): " << KE[0] << std::endl;
    double end = omp_get_wtime();
    std::cout << "Runtime = " << (end - start) << " s for N = " << N << std::endl;
    return 0;
}
