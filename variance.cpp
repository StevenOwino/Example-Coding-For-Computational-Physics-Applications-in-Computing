/* The code below computes the Bartlett covariance matrix for an exponential autocorrelation model, then sums its entries to estimate the variance of \widehat{\tau}_{\mathrm{int}}(W)=\frac12+\sum_{t=1}^{W}\widehat{\rho}(t). 
τ
  
int
​	
 (W)= 
2
1
​	
 +∑ 
t=1
W
​	
  
ρ
​	
 (t). It writes a CSV comparing that finite-window Bartlett sum with the Madras–Sokal large-WW approximation, \operatorname{Var}(\widehat{\tau}_{\mathrm{int}}) \approx \frac{2(2W+1)}{N}\tau_{\mathrm{int}}^2.Var( 
τ
  
int
​	
 )≈ 
N
2(2W+1)
​	
 τ 
int
2
​	
 . This approximation is intended for a window much larger than the integrated autocorrelation time, while still small relative to the series length. */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    // Model: rho(t) = exp(-|t| / tauExp).
    constexpr double tauExp = 10.0;
    constexpr int N = 1'000'000;
    constexpr int Wmax = 200;

    // Truncate Bartlett's infinite sum well into the exponential tail.
    const int Kmax = static_cast<int>(std::ceil(50.0 * tauExp + Wmax));

    auto rho = [](int t) {
        return std::exp(-std::abs(t) / tauExp);
    };

    // For this discrete exponential ACF:
    // tau_int = 1/2 + sum_{t=1}^\infty rho(t).
    const double tauInt =
        0.5 + 1.0 / (std::exp(1.0 / tauExp) - 1.0);

    // Bartlett covariance of rho-hat(i) and rho-hat(j):
    //
    // Cov(rhohat(i), rhohat(j)) ~= (1/N) sum_{k=1}^\infty A(i,k) A(j,k)
    //
    // where A(i,k) = rho(k+i) + rho(k-i) - 2 rho(i) rho(k).
    std::vector<std::vector<double>> cov(
        Wmax + 1, std::vector<double>(Wmax + 1, 0.0));

    for (int i = 1; i <= Wmax; ++i) {
        for (int j = 1; j <= Wmax; ++j) {
            double sum = 0.0;

            for (int k = 1; k <= Kmax; ++k) {
                const double ai =
                    rho(k + i) + rho(k - i) - 2.0 * rho(i) * rho(k);
                const double aj =
                    rho(k + j) + rho(k - j) - 2.0 * rho(j) * rho(k);

                sum += ai * aj;
            }

            cov[i][j] = sum / N;
        }
    }

    std::ofstream out("variance.csv");
    if (!out) {
        std::cerr << "Could not open variance.csv for writing\n";
        return 1;
    }

    out << "W,bartlett_variance,madras_sokal_variance\n";
    out << std::setprecision(12);

    // Var(sum_{t=1}^W rhohat(t)): add the new row/column at each W.
    double bartlettVariance = 0.0;

    for (int W = 1; W <= Wmax; ++W) {
        bartlettVariance += cov[W][W];

        for (int i = 1; i < W; ++i) {
            bartlettVariance += 2.0 * cov[i][W];
        }

        const double madrasSokalVariance =
            (2.0 * (2.0 * W + 1.0) / N) * tauInt * tauInt;

        out << W << ','
            << bartlettVariance << ','
            << madrasSokalVariance << '\n';
    }

    std::cout << "Wrote variance.csv\n"
              << "tau_int = " << tauInt << '\n'
              << "Compare the columns for W much larger than tau_int.\n";
}

