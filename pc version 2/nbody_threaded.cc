#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>

struct Particle {
    double x, y, z;     // Position
    double vx, vy, vz;  // Velocity
    double mass;
};

std::mutex mtx; // For thread-safe updates

void compute_forces(int start, int end, const std::vector<Particle>& particles, std::vector<double>& ax, std::vector<double>& ay, std::vector<double>& az, double G, double softening, int N) {
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
}

double compute_ke(int start, int end, const std::vector<Particle>& particles) {
    double ke = 0.0;
    for (int i = start; i < end; i++) {
        ke += 0.5 * particles[i].mass * (particles[i].vx * particles[i].vx +
                                         particles[i].vy * particles[i].vy +
                                         particles[i].vz * particles[i].vz);
    }
    return ke;
}

int main(int argc, char* argv[]) {
    // Simulation parameters
    int N = (argc > 1) ? std::atoi(argv[1]) : 128; // Number of particles
    double dt = 0.001;                             // Time step
    int steps = 301;                               // Number of time steps
    const double G = 6.67430e-11;                  // Gravitational constant
    const double softening = 1e-9;                 // Softening length
    const int num_threads = 4;                     // Number of threads

    // Initialize random number generator
    std::mt19937 gen(0);
    std::uniform_real_distribution<> ran(0.0, 1.0);

    // Initialize particles
    std::vector<Particle> particles(N);
    for (auto& p : particles) {
        p.x = ran(gen) * 1e18; p.y = ran(gen) * 1e18; p.z = ran(gen) * 1e18;
        p.vx = ran(gen) * 1e3; p.vy = ran(gen) * 1e3; p.vz = ran(gen) * 1e3;
        p.mass = 1.989e30; // Solar mass
    }

    // Output file for kinetic energy
    std::ofstream file("KE_THREADED_" + std::to_string(N) + ".txt");
    file << "# Kinetic Energy\n";

    // Time loop
    for (int step = 0; step < steps; step++) {
        // Compute forces
        std::vector<double> ax(N, 0.0), ay(N, 0.0), az(N, 0.0);
        std::vector<std::thread> threads;
        int chunk_size = N / num_threads;
        for (int t = 0; t < num_threads; t++) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? N : start + chunk_size;
            threads.emplace_back(compute_forces, start, end, std::cref(particles), std::ref(ax), std::ref(ay), std::ref(az), G, softening, N);
        }
        for (auto& thread : threads) {
            thread.join();
        }

        // Update velocities and positions
        for (int i = 0; i < N; i++) {
            particles[i].vx += ax[i] * dt / particles[i].mass;
            particles[i].vy += ay[i] * dt / particles[i].mass;
            particles[i].vz += az[i] * dt / particles[i].mass;
            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;
            particles[i].z += particles[i].vz * dt;
        }

        // Compute kinetic energy
        double ke = 0.0;
        threads.clear();
        for (int t = 0; t < num_threads; t++) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? N : start + chunk_size;
            threads.emplace_back([&ke, start, end, &particles]() {
                double local_ke = compute_ke(start, end, particles);
                std::lock_guard<std::mutex> lock(mtx);
                ke += local_ke;
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        file << ke << " ";
    }

    file << std::endl;
    file.close();
    return 0;
}