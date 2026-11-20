#include <cmath>
#include <fstream>
#include <iostream>

int main() {
    const double wn = 2.0;       // Natural frequency (rad/s)
    const double zeta = 0.35;    // Damping ratio for the damped case
    const double zetaOver = 1.5; // Damping ratio for the overdamped case
    const double dt = 0.01;
    const double tMax = 10.0;

    std::ofstream data("responses.dat");
    if (!data) {
        std::cerr << "Could not create responses.dat\n";
        return 1;
    }

    for (double t = 0.0; t <= tMax; t += dt) {
        const double undamped = std::cos(wn * t);

        const double wd = wn * std::sqrt(1.0 - zeta * zeta);
        const double damped =
            std::exp(-zeta * wn * t) *
            (std::cos(wd * t) +
             (zeta / std::sqrt(1.0 - zeta * zeta)) * std::sin(wd * t));

        const double criticallyDamped =
            (1.0 + wn * t) * std::exp(-wn * t);

        const double root = std::sqrt(zetaOver * zetaOver - 1.0);
        const double r1 = -zetaOver * wn + wn * root;
        const double r2 = -zetaOver * wn - wn * root;
        const double overdamped =
            (-r2 * std::exp(r1 * t) + r1 * std::exp(r2 * t)) / (r1 - r2);

        data << t << ' ' << undamped << ' ' << damped << ' '
             << criticallyDamped << ' ' << overdamped << '\n';
    }
    data.close();

    std::ofstream plot("plot.gp");
    plot << "set terminal pngcairo size 1000,650\n"
         << "set output 'damping_graph.png'\n"
         << "set title 'Free Response for Different Damping Cases'\n"
         << "set xlabel 'Time (s)'\n"
         << "set ylabel 'Displacement x(t)'\n"
         << "set grid\n"
         << "set key outside\n"
         << "plot 'responses.dat' using 1:2 with lines title 'Undamped', \\\n"
         << "     'responses.dat' using 1:3 with lines title 'Damped (ζ = 0.35)', \\\n"
         << "     'responses.dat' using 1:4 with lines title 'Critically damped (ζ = 1)', \\\n"
         << "     'responses.dat' using 1:5 with lines title 'Overdamped (ζ = 1.5)'\n";
    plot.close();

    std::cout << "Created responses.dat and plot.gp\n"
              << "Run: gnuplot plot.gp\n"
              << "The graph will be saved as damping_graph.png\n";
    return 0;
}
