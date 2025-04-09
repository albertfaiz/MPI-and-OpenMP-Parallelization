// nbody_mpi_shared.cc - Shared-Memory MPI Version
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>

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

int rank, n_ranks;

template <typename T>
void save(const std::vector<T>& vec, const std::string &filename, const std::string &header=""){
    if(rank==0){
        std::ofstream file(filename);
        if(file.is_open()){
            if(!header.empty())
                file << "# " << header << std::endl;
            for(auto &elem: vec)
                file << elem << " ";
            file << std::endl;
            file.close();
        } else {
            std::cerr << "Unable to open file " << filename << std::endl;
        }
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    if(argc > 1){
        N = std::atoi(argv[1]);
        ND = N * D;
    }
    // Allocate shared memory windows for positions and velocities
    double *x_shared, *v_shared;
    MPI_Win win_x, win_v;
    MPI_Aint size = ND * sizeof(double);
    int disp_unit = sizeof(double);
    if(rank == 0){
        MPI_Win_allocate_shared(size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &x_shared, &win_x);
        MPI_Win_allocate_shared(size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &v_shared, &win_v);
        // Initialize shared arrays on rank 0
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> ran(0.0, 1.0);
        double dx = x_max - x_min, dv = v_max - v_min;
        for (int i = 0; i < ND; i++){
            x_shared[i] = ran(gen) * dx + x_min;
            v_shared[i] = ran(gen) * dv + v_min;
        }
    } else {
        MPI_Win_allocate_shared(0, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &x_shared, &win_x);
        MPI_Win_allocate_shared(0, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &v_shared, &win_v);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // For demonstration, perform one timestep update (in a full simulation, you would loop over T steps)
    std::vector<double> x_current(x_shared, x_shared + ND);
    std::vector<double> v_current(v_shared, v_shared + ND);
    std::vector<double> a(ND, 0.0);
    for (int i = 0; i < N; i++){
        int iD = i * D;
        double diff[D];
        for (int j = 0; j < N; j++){
            int jD = j * D;
            double r2 = epsilon2;
            for (int k = 0; k < D; k++){
                diff[k] = x_current[jD + k] - x_current[iD + k];
                r2 += diff[k] * diff[k];
            }
            double denom = r2 * std::sqrt(r2);
            double factor = G * m_0 / denom; // using constant mass m_0
            for (int k = 0; k < D; k++){
                a[iD + k] += factor * diff[k];
            }
        }
    }
    // Update velocities and positions
    for (int i = 0; i < ND; i++){
        v_current[i] += a[i] * dt;
        x_current[i] += v_current[i] * dt;
    }
    // Use MPI shared memory window locks to update the global shared arrays
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_x);
    std::copy(x_current.begin(), x_current.end(), x_shared);
    MPI_Win_unlock(0, win_x);
    
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_v);
    std::copy(v_current.begin(), v_current.end(), v_shared);
    MPI_Win_unlock(0, win_v);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "Shared-memory update complete." << std::endl;
        // For demonstration, compute a dummy “kinetic energy” (you would compute KE properly in your simulation)
        std::vector<double> KE(1, 0.0);
        for (int i = 0; i < N; i++){
            double v2 = 0.0;
            for (int j = 0; j < D; j++){
                v2 += v_shared[i*D+j] * v_shared[i*D+j];
            }
            KE[0] += 0.5 * m_0 * v2;
        }
        save(KE, "KE_MPI_SHARED_" + std::to_string(N) + ".txt", "Kinetic Energy (Shared-MPI)");
    }
    MPI_Win_free(&win_x);
    MPI_Win_free(&win_v);
    MPI_Finalize();
    return 0;
}
