#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "csaps.h"

struct NaturalCubicSpline {
    std::vector<double> x, y, second;

    NaturalCubicSpline(const std::vector<double>& xs,
                       const std::vector<double>& ys)
        : x(xs), y(ys), second(xs.size(), 0.0) {
        const int n = static_cast<int>(x.size());
        if (n < 3) {
            throw std::runtime_error("Need at least 3 data points.");
        }

        // Solve the tridiagonal system for natural-spline second derivatives.
        std::vector<double> lower(n, 0.0), diagonal(n, 0.0);
        std::vector<double> upper(n, 0.0), rhs(n, 0.0);

        diagonal[0] = 1.0;
        diagonal[n - 1] = 1.0;

        for (int i = 1; i < n - 1; ++i) {
            const double h0 = x[i] - x[i - 1];
            const double h1 = x[i + 1] - x[i];

            lower[i] = h0;
            diagonal[i] = 2.0 * (h0 + h1);
            upper[i] = h1;
            rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h1
                          - (y[i] - y[i - 1]) / h0);
        }

        // Thomas algorithm.
        for (int i = 1; i < n; ++i) {
            const double factor = lower[i] / diagonal[i - 1];
            diagonal[i] -= factor * upper[i - 1];
            rhs[i] -= factor * rhs[i - 1];
        }

        second[n - 1] = rhs[n - 1] / diagonal[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            second[i] =
                (rhs[i] - upper[i] * second[i + 1]) / diagonal[i];
        }
    }

    double operator()(double q) const {
        int i;

        if (q <= x.front()) {
            i = 0;
        } else if (q >= x.back()) {
            i = static_cast<int>(x.size()) - 2;
        } else {
            int low = 0;
            int high = static_cast<int>(x.size()) - 1;

            while (high - low > 1) {
                const int mid = (low + high) / 2;
                if (x[mid] > q) high = mid;
                else low = mid;
            }
            i = low;
        }

        const double h = x[i + 1] - x[i];
        const double a = (x[i + 1] - q) / h;
        const double b = (q - x[i]) / h;

        return a * y[i] + b * y[i + 1]
             + ((a * a * a - a) * second[i]
             +  (b * b * b - b) * second[i + 1]) * h * h / 6.0;
    }
};

int main() {
    constexpr int dataCount = 18;
    constexpr int plotCount = 500;
    constexpr double pi = 3.14159265358979323846;
    constexpr double noiseStdDev = 0.20;
    constexpr double smooth = 0.90; // csaps smoothing parameter

    auto trueFunction = [](double x) {
        return std::sin(2.0 * pi * x);
    };

    // Generate reproducible noisy observations.
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, noiseStdDev);

    std::vector<double> xData(dataCount);
    std::vector<double> yData(dataCount);

    csaps::DoubleArray csapsX(dataCount);
    csaps::DoubleArray csapsY(dataCount);

    for (int i = 0; i < dataCount; ++i) {
        const double x = static_cast<double>(i) / (dataCount - 1);
        const double y = trueFunction(x) + noise(rng);

        xData[i] = x;
        yData[i] = y;
        csapsX(i) = x;
        csapsY(i) = y;
    }

    // Natural cubic interpolation passes through all noisy data points.
    NaturalCubicSpline interpolatingSpline(xData, yData);

    // Cubic smoothing spline from csaps-cpp.
    csaps::UnivariateCubicSmoothingSpline smoothingSpline(
        csapsX, csapsY, smooth
    );

    std::ofstream pointsFile("noisy_points.csv");
    if (!pointsFile) {
        std::cerr << "Could not open noisy_points.csv\n";
        return 1;
    }

    for (int i = 0; i < dataCount; ++i) {
        pointsFile << xData[i] << ',' << yData[i] << '\n';
    }
    pointsFile.close();

    std::ofstream dataFile("spline_plot.csv");
    if (!dataFile) {
        std::cerr << "Could not open spline_plot.csv\n";
        return 1;
    }

    // csaps-cpp evaluates a DoubleArray, so pass all plot x-values at once.
    csaps::DoubleArray query(plotCount);
    for (int i = 0; i < plotCount; ++i) {
        query(i) = static_cast<double>(i) / (plotCount - 1);
    }
    const csaps::DoubleArray smoothValues = smoothingSpline(query);

    dataFile << "x,true,interpolating,smoothing\n";
    for (int i = 0; i < plotCount; ++i) {
        const double x = query(i);

        dataFile << x << ','
                 << trueFunction(x) << ','
                 << interpolatingSpline(x) << ','
                 << smoothValues(i) << '\n';
    }
    dataFile.close();

    // Gnuplot script for the graph.
    std::ofstream plotFile("plot_splines.gp");
    if (!plotFile) {
        std::cerr << "Could not open plot_splines.gp\n";
        return 1;
    }

    plotFile << R"(
set datafile separator ","
set title "Noisy Data and Cubic Splines"
set xlabel "x"
set ylabel "y"
set grid
set key outside
plot "noisy_points.csv" using 1:2 with points pt 7 ps 1.2 title "Noisy data", \
     "spline_plot.csv" using 1:2 every ::1 with lines lw 2 title "True function", \
     "spline_plot.csv" using 1:3 every ::1 with lines lw 2 title "Cubic interpolation", \
     "spline_plot.csv" using 1:4 every ::1 with lines lw 2 title "Cubic smoothing spline"
pause -1
)";
    plotFile.close();

    // Requires gnuplot installed and available on PATH.
    return std::system("gnuplot plot_splines.gp");
}
