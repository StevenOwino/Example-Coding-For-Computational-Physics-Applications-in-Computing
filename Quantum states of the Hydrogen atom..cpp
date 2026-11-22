// For the non--relativistic Hydrogen atom, the Energy depends only on the Principal Quantum Number (n). For each (n), the allowed values are (l = 0), (n-1),                  and (m_l = -l...l). Thus the spatial degeneracy of shell (n) is (n^2); including the electrons' two spin states, it is (2n^2).

// For example, the (n = 3) shell contains 3s, 3p, and 3d subshells, with (1 + 3 + 5 = 9 spatial states, or 18 states when spin is included; The bars are a schematic shell display, not an orbital--shape plot.

// Schematic energy-level marker only; this does not plot an orbital shape.(It prints repeated dashes as a simple shell-level marker; it does not calculate or draw an orbitals' spatial shape)


#include <iostream>
#include <string>

std::string subshellName(int l) {
    // Conventional spectroscopic labels
    const std::string labels = "spdfghik";
    if (l < static_cast<int>(labels.size())) {
        return std::string(1, labels[l]);
    }
    return "l=" + std::to_string(l);
}

int main() {
    int maxN;
    std::cout << "Display shells up to n = ";
    if (!(std::cin >> maxN) || maxN < 1) {
        std::cerr << "Please enter a positive integer.\n";
        return 1;
    }

    std::cout << "\nHydrogen atom: nonrelativistic Coulomb levels\n";
    std::cout << "Energy depends only on n; spin is not included in the listed states.\n\n";

    for (int n = 1; n <= maxN; ++n) {
        const int spatialDegeneracy = n * n;
        const int spinDegeneracy = 2 * spatialDegeneracy;

        // Qualitative energy-level display: higher n is less tightly bound.
        std::cout << "n = " << n << "  E_n = -13.6 / " << n << "^2 eV"
                  << "  spatial degeneracy = " << spatialDegeneracy
                  << "  including spin = " << spinDegeneracy << '\n';

        std::cout << "  ";
        for (int i = 0; i < n; ++i) {
            std::cout << "----";
        }
        std::cout << '\n';

        int shellCount = 0;

        for (int l = 0; l < n; ++l) {
            const int subshellDegeneracy = 2 * l + 1;

            std::cout << "  " << n << subshellName(l)
                      << " (l = " << l << "): m_l = ";

            for (int m = -l; m <= l; ++m) {
                if (m > -l) {
                    std::cout << ", ";
                }
                std::cout << m;
            }

            std::cout << "  [" << subshellDegeneracy << " spatial states]\n";
            shellCount += subshellDegeneracy;
        }

        std::cout << "  Check: ";
        for (int l = 0; l < n; ++l) {
            if (l > 0) {
                std::cout << " + ";
            }
            std::cout << (2 * l + 1);
        }
        std::cout << " = " << shellCount << " spatial states\n\n";
    }

    std::cout << "Note: including spin doubles each spatial-state count.\n";
    return 0;
}
