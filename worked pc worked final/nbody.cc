// nbody.cc
// Serial n-body simulation code
// Compile with: g++ -O3 -std=c++17 nbody.cc -o nbody
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdlib>

static int N = 128;             // Number of masses (can be overridden via command line)
static const int D = 3;         // Dimensionality (3D)
static int ND = N * D;          // Size of state vectors
static const double G = 0.5;    // Gravitational constant
static const double dt = 1e-3;  // Time step size
static const int T = 300;       // Number of timesteps (modify if needed, e.g., 4096)
static const double epsilon = 0.01;       // Softening parameter
static const double epsilon2 = epsilon * epsilon;
static const double m0 = 1.0;   // Mass value

using Vec = std::vector<double>;

// Save a vector to a file
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

// Generate random initial conditions
std::tuple<Vec, Vec> initial_conditions() {
    Vec x(ND), v(ND);
    std::mt19937 gen(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < ND; i++) {
         x[i] = dist(gen);
         v[i] = 0.0;
    }
    return {x, v};
}

// Compute acceleration for each mass
Vec acceleration(const Vec &x, const Vec &m) {
    Vec a(ND, 0.0);
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

// Perform one timestep update
std::tuple<Vec, Vec> timestep(const Vec &x0, const Vec &v0, const Vec &m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
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
    std::cout << "Serial simulation complete. Time: " << elapsed << " s for N = " << N << "\n";
    return 0;
}
