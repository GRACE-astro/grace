/**
 * @file spherical_harmonics.hh
 * @brief Real spherical harmonics on Gauss-Legendre collocation grid.
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 */

#ifndef GRACE_UTILS_SPHERICAL_HARMONICS_HH
#define GRACE_UTILS_SPHERICAL_HARMONICS_HH

#include <vector>
#include <cmath>
#include <cassert>
#include <array>
#include <algorithm>

namespace grace {

/**
 * @brief Compute Gauss-Legendre nodes and weights on [-1,1].
 */
inline void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(n);
    weights.resize(n);
    for (int i = 0; i < n; ++i) {
        double x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        for (int iter = 0; iter < 100; ++iter) {
            double p0 = 1.0, p1 = x;
            for (int k = 1; k < n; ++k) {
                double p2 = ((2*k + 1)*x*p1 - k*p0) / (k + 1);
                p0 = p1; p1 = p2;
            }
            double pp = n * (x*p1 - p0) / (x*x - 1.0);
            double dx = -p1 / pp;
            x += dx;
            if (std::abs(dx) < 1e-15) break;
        }
        // recompute P_n and P'_n at final x
        double p0 = 1.0, p1 = x;
        for (int k = 1; k < n; ++k) {
            double p2 = ((2*k + 1)*x*p1 - k*p0) / (k + 1);
            p0 = p1; p1 = p2;
        }
        double pp = n * (x*p1 - p0) / (x*x - 1.0);
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x*x) * pp*pp);
    }
}

/**
 * @brief Spectral basis of real spherical harmonics on a
 *        Gauss-Legendre (theta) x uniform (phi) collocation grid.
 *
 * Coefficient indexing for l, m:
 *   m = 0:      idx = l*l
 *   m > 0, cos: idx = l*l + 2*m - 1
 *   m > 0, sin: idx = l*l + 2*m
 *
 * Total coefficients: (l_max+1)^2
 */
class spherical_harmonic_basis_t {
public:
    explicit spherical_harmonic_basis_t(int l_max)
        : l_max_(l_max)
        , n_coeffs_((l_max+1)*(l_max+1))
        , n_theta_(l_max + 1)
        , n_phi_(2*l_max + 1)
        , n_points_(n_theta_ * n_phi_)
    {
        setup_grid();
        precompute_matrices();
    }

    int l_max()    const { return l_max_; }
    int n_coeffs() const { return n_coeffs_; }
    int n_theta()  const { return n_theta_; }
    int n_phi()    const { return n_phi_; }
    int n_points() const { return n_points_; }

    static int coeff_index(int l, int m, bool sine) {
        if (m == 0) return l*l;
        return l*l + 2*m - 1 + (sine ? 1 : 0);
    }

    double theta(int it) const { return theta_[it]; }
    double phi(int ip)   const { return phi_[ip]; }
    double cos_theta(int it) const { return cos_theta_[it]; }
    double sin_theta(int it) const { return sin_theta_[it]; }
    int point_index(int it, int ip) const { return it * n_phi_ + ip; }

    /// Quadrature weight for surface integral: w_GL(it) * 2pi/n_phi
    double weight(int it, int /*ip*/) const {
        return gl_weights_[it] * 2.0 * M_PI / n_phi_;
    }

    /// Evaluate h and angular derivatives from spectral coefficients.
    /// Output arrays must be pre-sized to n_points.
    void evaluate(
        const double* a,
        double* h, double* dh_dth, double* dh_dph,
        double* d2h_dth2 = nullptr,
        double* d2h_dthdph = nullptr,
        double* d2h_dph2 = nullptr
    ) const {
        // matrix-vector multiplies: h = Y * a, dh/dth = dY_dth * a, etc.
        for (int p = 0; p < n_points_; ++p) {
            double v = 0, vt = 0, vp = 0;
            double vtt = 0, vtp = 0, vpp = 0;
            int base = p * n_coeffs_;
            for (int c = 0; c < n_coeffs_; ++c) {
                v  += Y_[base + c]       * a[c];
                vt += dY_dth_[base + c]  * a[c];
                vp += dY_dph_[base + c]  * a[c];
                if (d2h_dth2) {
                    vtt += d2Y_dth2_[base + c]   * a[c];
                    vtp += d2Y_dthdph_[base + c]  * a[c];
                    vpp += d2Y_dph2_[base + c]    * a[c];
                }
            }
            h[p] = v; dh_dth[p] = vt; dh_dph[p] = vp;
            if (d2h_dth2) { d2h_dth2[p] = vtt; d2h_dthdph[p] = vtp; d2h_dph2[p] = vpp; }
        }
    }

    /// Evaluate h at a single arbitrary (theta, phi) from coefficients.
    double evaluate_single(const double* a, double th, double ph) const {
        auto [P, norm] = compute_legendre_and_norm(th);
        double val = 0;
        for (int l = 0; l <= l_max_; ++l) {
            val += a[coeff_index(l, 0, false)] * norm[l][0] * P[l][0];
            for (int m = 1; m <= l; ++m) {
                double Nlm = norm[l][m] * P[l][m];
                val += a[coeff_index(l, m, false)] * Nlm * std::cos(m * ph);
                val += a[coeff_index(l, m, true)]  * Nlm * std::sin(m * ph);
            }
        }
        return val;
    }

    /// Inverse transform: a_lm = integral of f * Y_lm dOmega via quadrature.
    void inverse(const double* f, double* a) const {
        for (int c = 0; c < n_coeffs_; ++c) {
            double sum = 0;
            for (int it = 0; it < n_theta_; ++it) {
                for (int ip = 0; ip < n_phi_; ++ip) {
                    int p = point_index(it, ip);
                    sum += weight(it, ip) * f[p] * Y_[p * n_coeffs_ + c];
                }
            }
            a[c] = sum;
        }
    }

    /// Access to precomputed Y_lm matrix (row-major, n_points x n_coeffs).
    const double* Y_matrix() const { return Y_.data(); }
    const double* dY_dth_matrix() const { return dY_dth_.data(); }
    const double* dY_dph_matrix() const { return dY_dph_.data(); }

private:
    int l_max_, n_coeffs_, n_theta_, n_phi_, n_points_;
    std::vector<double> theta_, phi_, cos_theta_, sin_theta_;
    std::vector<double> gl_weights_;
    // Matrices [n_points x n_coeffs], row-major
    std::vector<double> Y_, dY_dth_, dY_dph_;
    std::vector<double> d2Y_dth2_, d2Y_dthdph_, d2Y_dph2_;

    void setup_grid() {
        std::vector<double> gl_nodes;
        gauss_legendre(n_theta_, gl_nodes, gl_weights_);
        // GL nodes are cos(theta), ordered from ~+1 to ~-1
        theta_.resize(n_theta_);
        cos_theta_.resize(n_theta_);
        sin_theta_.resize(n_theta_);
        for (int i = 0; i < n_theta_; ++i) {
            cos_theta_[i] = gl_nodes[i];
            theta_[i] = std::acos(gl_nodes[i]);
            sin_theta_[i] = std::sin(theta_[i]);
        }
        phi_.resize(n_phi_);
        for (int j = 0; j < n_phi_; ++j) {
            phi_[j] = 2.0 * M_PI * j / n_phi_;
        }
    }

    /**
     * @brief Compute associated Legendre P_l^m(cos th) and normalization
     *        factors for all l, m at a given theta.
     */
    using leg_table_t = std::vector<std::vector<double>>;
    std::pair<leg_table_t, leg_table_t>
    compute_legendre_and_norm(double th) const {
        double x = std::cos(th);
        double sth = std::sin(th);
        int L = l_max_;
        // P[l][m] = P_l^m(x) with Condon-Shortley phase
        leg_table_t P(L+2, std::vector<double>(L+2, 0.0));
        P[0][0] = 1.0;
        for (int m = 1; m <= L; ++m)
            P[m][m] = -(2*m - 1) * sth * P[m-1][m-1];
        for (int m = 0; m <= L; ++m) {
            if (m + 1 <= L)
                P[m+1][m] = (2*m + 1) * x * P[m][m];
            for (int l = m + 2; l <= L; ++l)
                P[l][m] = ((2*l - 1)*x*P[l-1][m] - (l + m - 1)*P[l-2][m]) / (l - m);
        }
        // normalization: N[l][m]
        leg_table_t N(L+1, std::vector<double>(L+1, 0.0));
        for (int l = 0; l <= L; ++l) {
            for (int m = 0; m <= l; ++m) {
                double ratio = 1.0;
                for (int k = l - m + 1; k <= l + m; ++k)
                    ratio *= k; // ratio = (l+m)!/(l-m)!
                double fac = (2*l + 1) / (4.0 * M_PI) / ratio;
                N[l][m] = std::sqrt(fac) * (m > 0 ? std::sqrt(2.0) : 1.0);
            }
        }
        return {P, N};
    }

    void precompute_matrices() {
        size_t sz = n_points_ * n_coeffs_;
        Y_.assign(sz, 0.0);
        dY_dth_.assign(sz, 0.0);
        dY_dph_.assign(sz, 0.0);
        d2Y_dth2_.assign(sz, 0.0);
        d2Y_dthdph_.assign(sz, 0.0);
        d2Y_dph2_.assign(sz, 0.0);

        for (int it = 0; it < n_theta_; ++it) {
            double th = theta_[it];
            double x = cos_theta_[it];
            double sth = sin_theta_[it];
            auto PN = compute_legendre_and_norm(th);
            auto& P = PN.first;
            auto& N = PN.second;

            // dP_l^m / dtheta
            // = [l*cos(th)*P_l^m - (l+m)*P_{l-1}^m] / sin(th)
            // for l=m: = m*cos(th)*P_m^m / sin(th) (since P_{m-1}^m = 0)
            auto dP = [&](int l, int m) -> double {
                if (l < m) return 0.0;
                double Plm1 = (l - 1 >= m) ? P[l-1][m] : 0.0;
                return (l * x * P[l][m] - (l + m) * Plm1) / sth;
            };

            // d2P_l^m / dtheta2 from the associated Legendre ODE:
            // = -cot(th) dP/dth - [l(l+1) - m^2/sin^2(th)] P
            auto d2P = [&](int l, int m) -> double {
                return -(x/sth) * dP(l, m)
                       - (l*(l+1) - m*m/(sth*sth)) * P[l][m];
            };

            for (int ip = 0; ip < n_phi_; ++ip) {
                double ph = phi_[ip];
                int pt = point_index(it, ip);

                for (int l = 0; l <= l_max_; ++l) {
                    // m = 0
                    {
                        int c = coeff_index(l, 0, false);
                        double Ylm  = N[l][0] * P[l][0];
                        double dYth = N[l][0] * dP(l, 0);
                        double d2Yth2 = N[l][0] * d2P(l, 0);
                        Y_[pt*n_coeffs_ + c]            = Ylm;
                        dY_dth_[pt*n_coeffs_ + c]       = dYth;
                        dY_dph_[pt*n_coeffs_ + c]       = 0.0;
                        d2Y_dth2_[pt*n_coeffs_ + c]     = d2Yth2;
                        d2Y_dthdph_[pt*n_coeffs_ + c]   = 0.0;
                        d2Y_dph2_[pt*n_coeffs_ + c]     = 0.0;
                    }
                    // m > 0
                    for (int m = 1; m <= l; ++m) {
                        double Nlm = N[l][m];
                        double Plm = P[l][m];
                        double dPlm = dP(l, m);
                        double d2Plm = d2P(l, m);
                        double cm = std::cos(m * ph), sm = std::sin(m * ph);

                        int cc = coeff_index(l, m, false); // cosine
                        int cs = coeff_index(l, m, true);  // sine

                        Y_[pt*n_coeffs_ + cc]          = Nlm * Plm * cm;
                        Y_[pt*n_coeffs_ + cs]          = Nlm * Plm * sm;

                        dY_dth_[pt*n_coeffs_ + cc]     = Nlm * dPlm * cm;
                        dY_dth_[pt*n_coeffs_ + cs]     = Nlm * dPlm * sm;

                        dY_dph_[pt*n_coeffs_ + cc]     = -m * Nlm * Plm * sm;
                        dY_dph_[pt*n_coeffs_ + cs]     =  m * Nlm * Plm * cm;

                        d2Y_dth2_[pt*n_coeffs_ + cc]   = Nlm * d2Plm * cm;
                        d2Y_dth2_[pt*n_coeffs_ + cs]   = Nlm * d2Plm * sm;

                        d2Y_dthdph_[pt*n_coeffs_ + cc] = -m * Nlm * dPlm * sm;
                        d2Y_dthdph_[pt*n_coeffs_ + cs] =  m * Nlm * dPlm * cm;

                        d2Y_dph2_[pt*n_coeffs_ + cc]   = -m*m * Nlm * Plm * cm;
                        d2Y_dph2_[pt*n_coeffs_ + cs]   = -m*m * Nlm * Plm * sm;
                    }
                }
            }
        }
    }
};

} // namespace grace

#endif // GRACE_UTILS_SPHERICAL_HARMONICS_HH
