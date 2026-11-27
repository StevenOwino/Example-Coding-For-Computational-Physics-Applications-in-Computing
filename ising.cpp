/* Here’s a self-contained C++20 reference program implementing both algorithms for the zero-field, ferromagnetic 2D Ising model with periodic boundaries. It prints CSV measurements for each method. For Wolff, one reported “sweep” means cluster updates continue until at least N = L^2 spins have been flipped; this is a convention, not the same work as one Metropolis sweep. */

/* Compile and run near criticality with J = $k_{\mathrm B}, so $\beta_c$ $\approx$ 0.4406868: 
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic ising.cpp -o ising
./ising 32 0.4406868 1000 5000 12345 > measurements.csv */


/* This is a CPU reference implementation, not the hybrid CPU/GPU version. The CSV measurements are successive samples, so they are correlated; estimate autocorrelation and statistical errors before comparing the algorithms’ efficiency.c */


#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class Ising2D {
public:
    Ising2D(int size, double beta, std::uint64_t seed)
        : L(size),
          N(size * size),
          beta(beta),
          spin(N),
          inCluster(N, 0),
          rng(seed),
          siteDist(0, N - 1),
          uniform(0.0, 1.0) {
        if (L < 3) {
            throw std::invalid_argument("L must be at least 3.");
        }
        if (beta < 0.0) {
            throw std::invalid_argument("beta must be nonnegative.");
        }

        // Random initial configuration.
        for (int& s : spin) {
            s = (uniform(rng) < 0.5) ? -1 : 1;
        }

        M = 0;
        for (int s : spin) {
            M += s;
        }
        E = totalEnergy();
    }

    // One Metropolis sweep: N randomly selected single-spin proposals.
    void metropolisSweep() {
        for (int attempt = 0; attempt < N; ++attempt) {
            const int i = siteDist(rng);
            const int s = spin[i];
            const int sum = neighborSum(i);
            const double deltaE = 2.0 * s * sum; // J = 1

            if (deltaE <= 0.0 || uniform(rng) < std::exp(-beta * deltaE)) {
                spin[i] = -s;
                E += deltaE;
                M -= 2 * s;
            }
        }
    }

    // One Wolff cluster update; returns the number of spins flipped.
    int wolffUpdate() {
        const int seed = siteDist(rng);
        const int seedSpin = spin[seed];
        const double pAdd = 1.0 - std::exp(-2.0 * beta);

        ++clusterMark;
        if (clusterMark == 0) { // Extremely unlikely integer wraparound.
            std::fill(inCluster.begin(), inCluster.end(), 0);
            clusterMark = 1;
        }

        std::vector<int> cluster;
        cluster.reserve(64);
        cluster.push_back(seed);
        inCluster[seed] = clusterMark;

        // Grow the cluster through aligned nearest neighbors.
        for (std::size_t head = 0; head < cluster.size(); ++head) {
            const int i = cluster[head];

            for (int j : neighbors(i)) {
                if (inCluster[j] != clusterMark &&
                    spin[j] == seedSpin &&
                    uniform(rng) < pAdd) {
                    inCluster[j] = clusterMark;
                    cluster.push_back(j);
                }
            }
        }

        // Sum the energy changes before flipping. Internal cluster bonds
        // cancel in the sum, leaving only boundary-bond contributions.
        double deltaE = 0.0;
        for (int i : cluster) {
            deltaE += 2.0 * spin[i] * neighborSum(i);
        }

        for (int i : cluster) {
            M -= 2 * spin[i];
            spin[i] = -spin[i];
        }
        E += deltaE;

        return static_cast<int>(cluster.size());
    }

    double energyPerSpin() const {
        return E / N;
    }

    double absMagnetizationPerSpin() const {
        return std::abs(static_cast<double>(M)) / N;
    }

    int siteCount() const {
        return N;
    }

private:
    int L;
    int N;
    double beta;
    std::vector<int> spin;
    std::vector<std::uint32_t> inCluster;
    std::uint32_t clusterMark = 0;
    std::mt19937_64 rng;
    std::uniform_int_distribution<int> siteDist;
    std::uniform_real_distribution<double> uniform;
    double E = 0.0;
    int M = 0;

    int index(int x, int y) const {
        return x * L + y;
    }

    // Four neighbors with periodic boundary conditions.
    std::vector<int> neighbors(int i) const {
        const int x = i / L;
        const int y = i % L;

        return {
            index((x + 1) % L, y),
            index((x + L - 1) % L, y),
            index(x, (y + 1) % L),
            index(x, (y + L - 1) % L)
        };
    }

    int neighborSum(int i) const {
        int sum = 0;
        for (int j : neighbors(i)) {
            sum += spin[j];
        }
        return sum;
    }

    // Count each nearest-neighbor bond once (right and down only).
    double totalEnergy() const {
        double energy = 0.0;
        for (int x = 0; x < L; ++x) {
            for (int y = 0; y < L; ++y) {
                const int i = index(x, y);
                const int right = index(x, (y + 1) % L);
                const int down = index((x + 1) % L, y);
                energy -= spin[i] * (spin[right] + spin[down]);
            }
        }
        return energy; // J = 1
    }
};

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " L beta burn_sweeps samples seed\n";
        return 1;
    }

    try {
        const int L = std::stoi(argv[1]);
        const double beta = std::stod(argv[2]);
        const int burnSweeps = std::stoi(argv[3]);
        const int samples = std::stoi(argv[4]);
        const std::uint64_t seed =
            static_cast<std::uint64_t>(std::stoull(argv[5]));

        if (burnSweeps < 0 || samples <= 0) {
            throw std::invalid_argument(
                "burn_sweeps must be nonnegative and samples must be positive.");
        }

        // J = k_B = 1, so beta_c = 0.5 * ln(1 + sqrt(2)).
        const double betaCritical = 0.5 * std::log(1.0 + std::sqrt(2.0));
        std::cerr << "L=" << L << ", beta=" << beta
                  << ", beta_c=" << betaCritical << '\n';

        std::cout << "algorithm,sample,energy_per_spin,abs_magnetization_per_spin\n";
        std::cout << std::setprecision(10);

        // Metropolis run.
        Ising2D metro(L, beta, seed);
        for (int t = 0; t < burnSweeps; ++t) {
            metro.metropolisSweep();
        }
        for (int sample = 0; sample < samples; ++sample) {
            metro.metropolisSweep();
            std::cout << "Metropolis," << sample << ','
                      << metro.energyPerSpin() << ','
                      << metro.absMagnetizationPerSpin() << '\n';
        }

        // Wolff run with an independent seed.
        Ising2D wolff(L, beta, seed + 1);
        for (int t = 0; t < burnSweeps; ++t) {
            int flipped = 0;
            while (flipped < wolff.siteCount()) {
                flipped += wolff.wolffUpdate();
            }
        }
        for (int sample = 0; sample < samples; ++sample) {
            int flipped = 0;
            while (flipped < wolff.siteCount()) {
                flipped += wolff.wolffUpdate();
            }
            std::cout << "Wolff," << sample << ','
                      << wolff.energyPerSpin() << ','
                      << wolff.absMagnetizationPerSpin() << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

