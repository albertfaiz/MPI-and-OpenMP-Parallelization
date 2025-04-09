// nbody_mpi_omp.cc
// Hybrid MPI+OpenMP n-body simulation
// Compile with: mpicxx -fopenmp -DUSE_MPI -O3 -std=c++17 nbody_mpi_omp.cc -o nbody_mpi_omp
// Run with: export OMP_NUM_THREADS=4; mpiexec -n 2 ./nbody_mpi_omp 1024
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <chrono>
#include <random>
#include <cmath>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// Global simulation parameters
static int N = 128;             // Total number of masses (can be overridden via command line)
static const int D = 3;         // Dimensionality (3D)
static int ND = N * D;          // Size of state vectors (global)
static const double G = 0.5;    // Gravitational constant
static const double dt = 1e-3;  // Time step size
static const int T = 300;       // Number of timesteps (modify as needed)
static const double epsilon = 0.01;       // Softening parameter
static const double epsilon2 = epsilon * epsilon;
static const double m0 = 1.0;   // Mass value

using Vec = std::vector<double>;

// Domain decomposition variables (only used when MPI is enabled)
#ifdef USE_MPI
int rank, n_ranks;         // MPI process rank and total number of processes
int local_N;               // Number of masses for this MPI rank
int global_N;              // Total number of masses (== N)
int local_start;           // Starting mass index for this rank
std::vector<int> counts;   // Number of masses per rank
std::vector<int> displs;   // Displacement (in mass index) for each rank
std::vector<int> countsD;  // Number of state elements per rank (each mass has D entries)
std::vector<int> displsD;  // Displacement (in state vector index) for each rank

// Initialize domain decomposition: divide N among MPI ranks
void setup_domain_decomposition(int N_total) {
    global_N = N_total;
    counts.resize(n_ranks);
    displs.resize(n_ranks);
    countsD.resize(n_ranks);
    displsD.resize(n_ranks);
    int base = N_total / n_ranks;
    int remainder = N_total % n_ranks;
    for (int i = 0; i < n_ranks; i++) {
        counts[i] = base + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
        countsD[i] = counts[i] * D;
        displsD[i] = displs[i] * D;
    }
    local_N = counts[rank];
    local_start = displs[rank];
}
#endif

// Initialize positions and velocities.
// Global vectors x and v are allocated to size ND; each MPI rank initializes only its local portion.
void initialize_conditions(Vec &x, Vec &v) {
    // Seed random number generator using current time (and rank if using MPI)
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
#ifdef USE_MPI
    seed ^= rank;
#endif
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
#ifdef USE_MPI
    int local_start_idx = local_start * D;
    int local_end_idx = (local_start + local_N) * D;
#else
    int local_start_idx = 0;
    int local_end_idx = ND;
#endif

    for (int i = local_start_idx; i < local_end_idx; i++) {
        x[i] = dist(gen);
        v[i] = 0.0;
    }
}

// Compute accelerations for masses in the local range.
void compute_acceleration(const Vec &x, const Vec &masses, Vec &a) {
#ifdef USE_MPI
    int local_start_idx = local_start * D;
    int local_end_idx = (local_start + local_N) * D;
#else
    int local_start_idx = 0;
    int local_end_idx = ND;
#endif

    for (int i = local_start_idx; i < local_end_idx; i++)
        a[i] = 0.0;
    
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = local_start; i < local_start + local_N; i++) {
        int iD = i * D;
        for (int j = 0; j < global_N; j++) {
            int jD = j * D;
            double dx[D];
            double dist2 = epsilon2;
            for (int k = 0; k < D; k++) {
                dx[k] = x[jD + k] - x[iD + k];
                dist2 += dx[k] * dx[k];
            }
            double inv_dist3 = 1.0 / (dist2 * sqrt(dist2));
            double factor = G * masses[j] * inv_dist3;
            for (int k = 0; k < D; k++) {
                a[iD + k] += factor * dx[k];
            }
        }
    }
}

// Perform one timestep update.
void timestep(Vec &x, Vec &v, const Vec &masses, Vec &a) {
#ifdef USE_MPI
    int local_start_idx = local_start * D;
    int local_end_idx = (local_start + local_N) * D;
#else
    int local_start_idx = 0;
    int local_end_idx = ND;
#endif

    compute_acceleration(x, masses, a);
    
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = local_start_idx; i < local_end_idx; i++) {
        v[i] += a[i] * dt;
        x[i] += v[i] * dt;
    }
}

// Compute the kinetic energy for the local masses.
double compute_kinetic_energy(const Vec &v, const Vec &masses) {
    double KE_local = 0.0;
#ifdef USE_MPI
    int count = local_N;
    for (int i = 0; i < count; i++) {
        double v2 = 0.0;
        int idx = (local_start + i) * D;
        for (int k = 0; k < D; k++) {
            v2 += v[idx + k] * v[idx + k];
        }
        KE_local += 0.5 * masses[local_start + i] * v2;
    }
#else
    int count = N;
    for (int i = 0; i < count; i++) {
        double v2 = 0.0;
        int idx = i * D;
        for (int k = 0; k < D; k++) {
            v2 += v[idx + k] * v[idx + k];
        }
        KE_local += 0.5 * masses[i] * v2;
    }
#endif
    return KE_local;
}

void save_vector(const Vec &vec, const std::string &filename, const std::string &header="") {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error opening file " << filename << "\n";
        return;
    }
    if (!header.empty())
        ofs << "# " << header << "\n";
    for (const auto &val : vec)
        ofs << val << " ";
    ofs << "\n";
    ofs.close();
}

int main(int argc, char **argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
#endif

    if (argc > 1)
        N = std::atoi(argv[1]);
    ND = N * D;

#ifdef USE_MPI
    setup_domain_decomposition(N);
    global_N = N;
#endif

    Vec t(T + 1, 0.0);
    for (int i = 0; i <= T; i++)
         t[i] = i * dt;
    Vec masses(N, m0);
    Vec x(ND, 0.0);
    Vec v(ND, 0.0);
    Vec a(ND, 0.0);
    Vec KE(T + 1, 0.0);

    initialize_conditions(x, v);
#ifdef USE_MPI
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
#endif
    double KE_local = compute_kinetic_energy(v, masses);
#ifdef USE_MPI
    double KE_total = 0.0;
    MPI_Reduce(&KE_local, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
         KE[0] = KE_total;
#else
    KE[0] = KE_local;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int t_step = 0; t_step < T; t_step++) {
         timestep(x, v, masses, a);
#ifdef USE_MPI
         MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
         MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
#endif
         KE_local = compute_kinetic_energy(v, masses);
#ifdef USE_MPI
         MPI_Reduce(&KE_local, &KE_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
         if (rank == 0)
              KE[t_step + 1] = KE_total;
#else
         KE[t_step + 1] = KE_local;
#endif
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
#ifdef USE_MPI
    if (rank == 0) {
         std::cout << "Hybrid MPI+OpenMP simulation complete.\n";
         std::cout << "Total simulation time: " << elapsed << " s for N = " << N << "\n";
         save_vector(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
         save_vector(t, "time_" + std::to_string(N) + ".txt", "Time");
    }
    MPI_Finalize();
#else
    std::cout << "Simulation complete.\n";
    std::cout << "Total simulation time: " << elapsed << " s for N = " << N << "\n";
    save_vector(KE, "KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
    save_vector(t, "time_" + std::to_string(N) + ".txt", "Time");
#endif

    return 0;
}
