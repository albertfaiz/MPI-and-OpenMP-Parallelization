// nbody_mpi.cc - MPI Version
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <cmath>
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
using Vecs = std::vector<Vec>;

int rank, n_ranks;
std::mt19937 gen;
std::uniform_real_distribution<> ran(0.0, 1.0);

// Arrays for work distribution:
std::vector<int> counts, displs, countsD, displsD;
int N_beg, N_end, ND_beg, ND_end;

template <typename T>
void save(const std::vector<T>& vec, const std::string& filename, const std::string& header="") {
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

void setup_parallelism(){
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    gen.seed(seed ^ rank);
    counts.resize(n_ranks);
    displs.resize(n_ranks);
    countsD.resize(n_ranks);
    displsD.resize(n_ranks);
    int base = N / n_ranks;
    int rem = N % n_ranks;
    for (int i=0; i<n_ranks; i++){
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = i * base + (i < rem ? i : rem);
        countsD[i] = counts[i] * D;
        displsD[i] = displs[i] * D;
    }
    N_beg = displs[rank];
    N_end = N_beg + counts[rank];
    ND_beg = N_beg * D;
    ND_end = N_end * D;
}

std::tuple<Vec, Vec> initial_conditions() {
    // Each process initializes its local portion
    Vec x_local(ND_end - ND_beg), v_local(ND_end - ND_beg);
    double dx = x_max - x_min, dv = v_max - v_min;
    for (int i = ND_beg; i < ND_end; i++){
        x_local[i - ND_beg] = ran(gen) * dx + x_min;
        v_local[i - ND_beg] = ran(gen) * dv + v_min;
    }
    // Allocate full arrays and fill with local data
    Vec x(ND, 0.0), v(ND, 0.0);
    for (int i = ND_beg; i < ND_end; i++){
        x[i] = x_local[i - ND_beg];
        v[i] = v_local[i - ND_beg];
    }
    // Gather data from all processes
    MPI_Allgatherv(MPI_IN_PLACE, ND_end-ND_beg, MPI_DOUBLE,
                   x.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, ND_end-ND_beg, MPI_DOUBLE,
                   v.data(), countsD.data(), displsD.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return {x, v};
}

Vec acceleration(const Vec &x, const Vec &m) {
    Vec a(ND, 0.0);
    for (int i = 0; i < N; i++){
        int iD = i * D;
        double diff[D];
        for (int j = 0; j < N; j++){
            int jD = j * D;
            double r2 = epsilon2;
            for (int k = 0; k < D; k++){
                diff[k] = x[jD + k] - x[iD + k];
                r2 += diff[k]*diff[k];
            }
            double denom = r2 * std::sqrt(r2);
            double factor = G * m[j] / denom;
            for (int k = 0; k < D; k++){
                a[iD + k] += factor * diff[k];
            }
        }
    }
    return a;
}

std::tuple<Vec, Vec> timestep(const Vec &x0, const Vec &v0, const Vec &m) {
    Vec a0 = acceleration(x0, m);
    Vec x1(ND), v1(ND);
    for (int i = 0; i < ND; i++){
        v1[i] = v0[i] + a0[i] * dt;
        x1[i] = x0[i] + v1[i] * dt;
    }
    return {x1, v1};
}

int main(int argc, char** argv){
    setup_parallelism();
    if(argc > 1){
        N = std::atoi(argv[1]);
        ND = N * D;
    }
    std::vector<double> t(T+1);
    for (int i = 0; i <= T; i++)
        t[i] = i * dt;
    Vec m(N, m_0);
    Vecs x(T+1), v(T+1);
    std::tie(x[0], v[0]) = initial_conditions();
    for (int n = 0; n < T; n++){
        std::tie(x[n+1], v[n+1]) = timestep(x[n], v[n], m);
    }
    Vec KE(T+1, 0.0);
    // Each process computes KE only over its local masses
    for (int n = 0; n <= T; n++){
        double local_ke = 0.0;
        for (int i = N_beg; i < N_end; i++){
            double v2 = 0.0;
            for (int j = 0; j < D; j++){
                v2 += v[n][i*D+j]*v[n][i*D+j];
            }
            local_ke += 0.5 * m[i] * v2;
        }
        KE[n] = local_ke;
    }
    std::vector<double> KE_total(T+1, 0.0);
    MPI_Reduce(KE.data(), KE_total.data(), T+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        save(KE_total, "KE_MPI_" + std::to_string(N) + ".txt", "Kinetic Energy (MPI)");
        save(t, "time_MPI_" + std::to_string(N) + ".txt", "Time");
        std::cout << "Total KE (first timestep) = " << KE_total[0] << std::endl;
    }
    MPI_Finalize();
    return 0;
}
