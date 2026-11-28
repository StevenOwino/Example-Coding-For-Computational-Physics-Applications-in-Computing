/* Example of Kawasaki Dynamics for a 2D nearest-neighbor Ising model with periodic boundaries. Each attempted move exchanges a randomly selected spin with one of its four neighbors, so total magnetization is conserved. */


/* Compile and run: g++ -O2 -std=c++17 kawasaki.cpp -o kawasaki
./kawasaki */

/* The code uses $k_{\mathrm B}$ = 1, so $\beta$ = $\frac{1}{T}$. 
The reported Magnetization should remain Constant throughout the run. */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
    const int L = 20;                  // Lattice width and height
    const int N = L * L;
    const double J = 1.0;             // Ferromagnetic coupling
    const double temperature = 2.0;
    const double beta = 1.0 / temperature;
    const int sweeps = 1000;           // One sweep = N attempted exchanges

    if (L < 3 || temperature <= 0.0) {
        std::cerr << "Use L >= 3 and temperature > 0.\n";
        return 1;
    }

    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> site(0, N - 1);
    std::uniform_int_distribution<int> direction(0, 3);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    // Initialize with equal numbers of +1 and -1 spins.
    // For odd N, the total magnetization will be -1.
    std::vector<int> spins(N, -1);
    for (int i = 0; i < N / 2; ++i) {
        spins[i] = 1;
    }
    std::shuffle(spins.begin(), spins.end(), rng);

    auto index = [L](int x, int y) {
        x = (x + L) % L;
        y = (y + L) % L;
        return x * L + y;
    };

    // Each bond is counted once: right and down neighbors only.
    auto total_energy = [&]() {
        double E = 0.0;
        for (int x = 0; x < L; ++x) {
            for (int y = 0; y < L; ++y) {
                const int s = spins[index(x, y)];
                E -= J * s * spins[index(x + 1, y)];
                E -= J * s * spins[index(x, y + 1)];
            }
        }
        return E;
    };

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (int attempt = 0; attempt < N; ++attempt) {
            const int i = site(rng);
            const int x = i / L;
            const int y = i % L;

            int nx = x;
            int ny = y;

            switch (direction(rng)) {
                case 0: nx = x + 1; break;
                case 1: nx = x - 1; break;
                case 2: ny = y + 1; break;
                case 3: ny = y - 1; break;
            }

            const int j = index(nx, ny);

            // Exchanging identical spins does not change the configuration.
            if (spins[i] == spins[j]) {
                continue;
            }

            // Sum the neighbors of each site, excluding the other selected
            // site. Their mutual bond, if present, is unchanged by exchange.
            const int neighbors_i =
                spins[index(x + 1, y)] + spins[index(x - 1, y)] +
                spins[index(x, y + 1)] + spins[index(x, y - 1)] -
                spins[j];

            const int neighbors_j =
                spins[index(nx + 1, ny)] + spins[index(nx - 1, ny)] +
                spins[index(nx, ny + 1)] + spins[index(nx, ny - 1)] -
                spins[i];

            // Energy change for exchanging unlike spins.
            const double delta_E =
                J * (spins[i] - spins[j]) * (neighbors_i - neighbors_j);

            // Metropolis acceptance rule.
            if (delta_E <= 0.0 ||
                uniform01(rng) < std::exp(-beta * delta_E)) {
                std::swap(spins[i], spins[j]);
            }
        }

        int magnetization = 0;
        for (int s : spins) {
            magnetization += s;
        }

        std::cout << "sweep " << std::setw(5) << sweep + 1
                  << "  E = " << std::setw(10) << total_energy()
                  << "  M = " << magnetization << '\n';
    }

    return 0;
}

