// nbody.cc
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>

static int N = 128;           // Number of masses
static const int D = 3;       // Dimensionality
static int ND = N * D;        // Size of state vectors
static const double G = 0.5;
static const double dt = 1e-3;
static const int T = 300;     // Number of time steps
static const double x_min = 0.0, x_max = 1.0;
static const double v_min = 0.0, v_max = 0.0;
static const double m_0 = 1.0;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;

using Vec = std::vector<double>;
using Vecs = std::vector<Vec>;

static std::mt19937 gen(std::random_device{}());
static std::uniform_real_distribution<> ran(0.0, 1.0);

template <typename T>
void save(const std::vector<T>& vec, const std::string& filename, const std::string& header = "") {
    std::ofstream file(filename);
    if(file.is_open()){
        if(!header.empty())
            file << "# " << header << "\n";
        for(const auto &elem : vec)
            file << elem << " ";
        file << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << "\n";
    }
}

std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    double dx = x_max - x_min, dv = v_max - v_min;
    for (int i = 0; i < ND; i++) {
        x[i] = ran(gen) * dx + x_min;
        v[i] = ran(gen) * dv + v_min;
    }
    return {x, v};
}

Vec acceleration(const Vec &x, const Vec &m) {
    Vec a(ND, 0.0);
    for (int i = 0; i < N; i++) {
        int iD = i * D;
        for (int j = 0; j < N; j++) {
            int jD = j * D;
            double dx[D], dx2 = epsilon2;
            for (int k = 0; k < D; k++) {
                dx[k] = x[jD + k] - x[iD + k];
                dx2 += dx[k] * dx[k];
            }
            double denom = dx2 * std::sqrt(dx2);
            double factor = G * m[j] / denom;
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
    for (int i = 0; i < ND; i++) {
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
    Vec t(T + 1);
    for (int i = 0; i <= T; i++) {
        t[i] = i * dt;
    }
    Vec m(N, m_0);
    Vecs x(T + 1), v(T + 1);
    std::tie(x[0], v[0]) = initial_conditions();
    for (int n = 0; n < T; n++) {
        std::tie(x[n + 1], v[n + 1]) = timestep(x[n], v[n], m);
    }
    Vec KE(T + 1, 0.0);
    for (int n = 0; n <= T; n++) {
        double KE_n = 0.0;
        for (int i = 0; i < N; i++) {
            double v2 = 0.0;
            for (int k = 0; k < D; k++) {
                v2 += v[n][i * D + k] * v[n][i * D + k];
            }
            KE_n += 0.5 * m[i] * v2;
        }
        KE[n] = KE_n;
    }
    save(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
    save(t, "time_" + std::to_string(N) + ".txt", "Time");
    std::cout << "Total Kinetic Energy (first time step): " << KE[0] << "\n";
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    std::cout << "Runtime = " << elapsed << " s for N = " << N << "\n";
    return 0;
}
