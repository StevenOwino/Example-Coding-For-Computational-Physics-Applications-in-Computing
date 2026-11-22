// Illustrative graph of Millikan-style measurements, drawing a cubic-spline curve through the data. The sample values are in units of the elementary charge (e)

#include <fstream>
#include <iostream>

int main() {
    // Synthetic illustrative data: drop number and measured charge in units of e.
    const double data[][2] = {
        {1, 1.02}, {2, 1.98}, {3, 3.03}, {4, 2.01}, {5, 3.98},
        {6, 1.01}, {7, 2.99}, {8, 4.02}, {9, 2.02}, {10, 3.01}
    };
    const int count = sizeof(data) / sizeof(data[0]);

    std::ofstream points("millikan_data.dat");
    if (!points) {
        std::cerr << "Could not create millikan_data.dat\n";
        return 1;
    }

    for (const auto& row : data) {
        points << row[0] << ' ' << row[1] << '\n';
    }
    points.close();

std::ofstream plot("millikan_plot.gp");
if (!plot) {
    std::cerr << "Could not create millikan_plot.gp\n";
    return 1;
}

plot << R"GNUPLOT(
set terminal pngcairo size 1600,1000 enhanced font "Arial,20"
set output "millikan_plot.png"

set title "Millikan Oil-Drop Experiment (Illustrative Data)" font ",24"
set xlabel "Oil-drop measurement number"
set ylabel "Measured charge (units of e)"
set xrange [0:11]
set yrange [0.5:4.5]
set xtics 1
set ytics 1
set grid
set key top left

# Horizontal reference lines at integer multiples of e
set arrow from 0,1 to 11,1 nohead dashtype 2 linecolor rgb "#999999"
set arrow from 0,2 to 11,2 nohead dashtype 2 linecolor rgb "#999999"
set arrow from 0,3 to 11,3 nohead dashtype 2 linecolor rgb "#999999"
set arrow from 0,4 to 11,4 nohead dashtype 2 linecolor rgb "#999999"

plot "millikan_data.dat" using 1:2 with points \
     pt 7 ps 2 lc rgb "#145DA0" title "Illustrative measurements", \
     "" using 1:2 smooth csplines with lines \
     lw 2 lc rgb "#E87500" title "Cubic spline"
)GNUPLOT";


plot.close();


    std::cout << "Created millikan_data.dat and millikan_plot.gp\n";
    std::cout << "Run: gnuplot millikan_plot.gp\n";
    return 0;
}
