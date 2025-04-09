#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>

struct Particle {
    double x, y, z;     // Position
    double vx, vy, vz;  // Velocity
    double mass;
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Simulation parameters
    int N = (argc > 1) ? std::atoi(argv[1]) : 128; // Number of particles
    double dt = 0.001;                             // Time step
    int steps = 301;                               // Number of time steps
    const double G = 6.67430e-11;                  // Gravitational constant
    const double softening = 1e-9;                 // Softening length

    // Initialize random number generator
    std::mt19937 gen(rank);
    std::uniform_real_distribution<> ran(0.0, 1.0);

    // Allocate shared memory for particles
    MPI_Win win;
    Particle* particles;
    MPI_Aint win_size = (rank == 0) ? N * sizeof(Particle) : 0;
    int disp_unit = sizeof(double);
    MPI_Win_allocate_shared(win_size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &particles, &win);

    // Initialize particles on rank 0
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            particles[i].x = ran(gen) * 1e18; particles[i].y = ran(gen) * 1e18; particles[i].z = ran(gen) * 1e18;
            particles[i].vx = ran(gen) * 1e3; particles[i].vy = ran(gen) * 1e3; particles[i].vz = ran(gen) * 1e3;
            particles[i].mass = 1.989e30;
        }
    }

    // Synchronize to ensure all ranks can access the shared memory
    MPI_Win_fence(0, win);

    // Output file for kinetic energy and time
    std::ofstream file;
    if (rank == 0) {
        std::string filename = "KE_MPI_SHARED_" + std::to_string(N) + ".txt";
        file.open(filename);
        file << "# Kinetic Energy\n";

        // Save time data
        std::ofstream time_file("time_MPI_SHARED_" + std::to_string(N) + ".txt");
        time_file << "# Time\n";
        for (int i = 0; i < steps; i++) {
            time_file << (i * dt) << " ";
        }
        time_file << std::endl;
        time_file.close();
    }

    // Time loop
    for (int step = 0; step < steps; step++) {
        // Compute forces
        int local_N = N / size;
        if (rank == size - 1) local_N += N % size;
        int start = rank * (N / size);
        int end = start + local_N;

        std::vector<double> ax(N, 0.0), ay(N, 0.0), az(N, 0.0);
        for (int i = start; i < end; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;
                double r = std::sqrt(dx * dx + dy * dy + dz * dz + softening);
                double force = G * particles[i].mass * particles[j].mass / (r * r * r);
                ax[i] += force * dx;
                ay[i] += force * dy;
                az[i] += force * dz;
            }
        }

        // Synchronize forces across ranks
        MPI_Win_fence(0, win);
        MPI_Allreduce(MPI_IN_PLACE, ax.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, ay.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, az.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Win_fence(0, win);

        // Update velocities and positions
        for (int i = start; i < end; i++) {
            particles[i].vx += ax[i] * dt / particles[i].mass;
            particles[i].vy += ay[i] * dt / particles[i].mass;
            particles[i].vz += az[i] * dt / particles[i].mass;
            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;
            particles[i].z += particles[i].vz * dt;
        }

        // Synchronize particles
        MPI_Win_fence(0, win);

        // Compute kinetic energy
        double local_ke = 0.0;
        for (int i = start; i < end; i++) {
            local_ke += 0.5 * particles[i].mass * (particles[i].vx * particles[i].vx +
                                                    particles[i].vy * particles[i].vy +
                                                    particles[i].vz * particles[i].vz);
        }
        double total_ke = 0.0;
        MPI_Reduce(&local_ke, &total_ke, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Output kinetic energy
        if (rank == 0) {
            file << total_ke << " ";
        }
    }

    if (rank == 0) {
        file << std::endl;
        file.close();
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}