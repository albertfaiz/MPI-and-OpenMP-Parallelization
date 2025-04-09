// nbody_mpi_shared.cc
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <cmath>

static int N = 128;
static const int D = 3;
static int ND = N * D;
static const double G = 0.5;
static const double dt = 1e-3;
static const int T = 300;
static const double x_min = 0.0, x_max = 1.0;
static const double v_min = 0.0, v_max = 0.0;
static const double m_0 = 1.0;
static const double epsilon = 0.01;
static const double epsilon2 = epsilon * epsilon;

using Vec = std::vector<double>;

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    if(argc > 1){
        N = std::atoi(argv[1]);
        ND = N * D;
    }
    MPI_Aint size = ND * sizeof(double);
    int disp_unit = sizeof(double);
    double *x_shared, *v_shared;
    MPI_Win win_x, win_v;
    
    // Allocate shared memory windows
    if(rank == 0){
        MPI_Win_allocate_shared(size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &x_shared, &win_x);
        MPI_Win_allocate_shared(size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &v_shared, &win_v);
    } else {
        MPI_Win_allocate_shared(0, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &x_shared, &win_x);
        MPI_Win_shared_query(win_x, 0, &size, &disp_unit, &x_shared);
        MPI_Win_allocate_shared(0, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &v_shared, &win_v);
        MPI_Win_shared_query(win_v, 0, &size, &disp_unit, &v_shared);
    }
    
    // Initialize shared arrays on rank 0
    if(rank == 0){
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> ran(0.0, 1.0);
        double dx = x_max - x_min, dv = v_max - v_min;
        for (int i = 0; i < ND; i++){
            x_shared[i] = ran(gen) * dx + x_min;
            v_shared[i] = ran(gen) * dv + v_min;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // For demonstration, perform one timestep update on shared arrays.
    std::vector<double> local_a(ND, 0.0);
    for (int i = 0; i < N; i++){
        int iD = i * D;
        for (int j = 0; j < N; j++){
            int jD = j * D;
            double dx[D], dx2 = epsilon2;
            for (int k = 0; k < D; k++){
                dx[k] = x_shared[jD + k] - x_shared[iD + k];
                dx2 += dx[k] * dx[k];
            }
            double denom = dx2 * std::sqrt(dx2);
            double factor = G * m_0 / denom;  // masses are all m_0
            for (int k = 0; k < D; k++){
                local_a[iD+k] += factor * dx[k];
            }
        }
    }
    // Update velocities and positions
    for (int i = 0; i < ND; i++){
        v_shared[i] += local_a[i] * dt;
        x_shared[i] += v_shared[i] * dt;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "Shared-memory update complete.\n";
    }
    
    MPI_Win_free(&win_x);
    MPI_Win_free(&win_v);
    
    MPI_Finalize();
    return 0;
}
