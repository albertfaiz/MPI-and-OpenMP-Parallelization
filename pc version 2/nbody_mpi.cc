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

    // Initialize particles (distributed across ranks)
    int local_N = N / size;
    if (rank == size - 1) {
        local_N += N % size;
    }
    std::vector<Particle> particles(local_N);
    for (auto& p : particles) {
        p.x = ran(gen) * 1e18; p.y = ran(gen) * 1e18; p.z = ran(gen) * 1e18;
        p.vx = ran(gen) * 1e3; p.vy = ran(gen) * 1e3; p.vz = ran(gen) * 1e3;
        p.mass = 1.989e30;
    }

    // Output file for kinetic energy and time
    std::ofstream file;
    if (rank == 0) {
        std::string filename = "KE_MPI_" + std::to_string(N) + ".txt";
        file.open(filename);
        file << "# Kinetic Energy\n";

        // Save time data
        std::ofstream time_file("time_MPI_" + std::to_string(N) + ".txt");
        time_file << "# Time\n";
        for (int i = 0; i < steps; i++) {
            time_file << (i * dt) << " ";
        }
        time_file << std::endl;
        time_file.close();
    }

    // Time loop
    for (int step = 0; step < steps; step++) {
        // Gather all particles
        std::vector<Particle> all_particles(N);
        int* recvcounts = new int[size];
        int* displs = new int[size];
        for (int i = 0; i < size; i++) {
            recvcounts[i] = (i == size - 1) ? (N / size + N % size) : (N / size);
            recvcounts[i] *= sizeof(Particle) / sizeof(double);
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
        }
        MPI_Allgatherv(particles.data(), local_N * sizeof(Particle) / sizeof(double), MPI_DOUBLE,
                       all_particles.data(), recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        delete[] recvcounts;
        delete[] displs;

        // Compute forces and update velocities
        for (int i = 0; i < local_N; i++) {
            double ax = 0.0, ay = 0.0, az = 0.0;
            for (int j = 0; j < N; j++) {
                if (j == (rank * (N / size) + i)) continue;
                double dx = all_particles[j].x - particles[i].x;
                double dy = all_particles[j].y - particles[i].y;
                double dz = all_particles[j].z - particles[i].z;
                double r = std::sqrt(dx * dx + dy * dy + dz * dz + softening);
                double force = G * particles[i].mass * all_particles[j].mass / (r * r * r);
                ax += force * dx;
                ay += force * dy;
                az += force * dz;
            }
            particles[i].vx += ax * dt / particles[i].mass;
            particles[i].vy += ay * dt / particles[i].mass;
            particles[i].vz += az * dt / particles[i].mass;
        }

        // Update positions
        for (int i = 0; i < local_N; i++) {
            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;
            particles[i].z += particles[i].vz * dt;
        }

        // Compute local kinetic energy
        double local_ke = 0.0;
        for (const auto& p : particles) {
            local_ke += 0.5 * p.mass * (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
        }

        // Reduce to get total kinetic energy
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

    MPI_Finalize();
    return 0;
}