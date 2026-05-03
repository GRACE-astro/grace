/**
 * @file grace_weakhub_table.hh
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @brief Weakhub library interpolation and lookup
 * @date 2026-02-12
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *                                    
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *   
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *   
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */

#ifndef GRACE_PHYSICS_WEAKHUB_TABLE_HH
#define GRACE_PHYSICS_WEAKHUB_TABLE_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/physics/eos/physical_constants.hh>
#include <grace/config/config_parser.hh>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <array>
#include <string>

namespace grace { namespace weakhub {

struct interp_outputs {
    std::array<double,5> kappa_a_en{{0.0,0.0,0.0,0.0,0.0}};
    std::array<double,5> kappa_a_num{{0.0,0.0,0.0,0.0,0.0}};
    std::array<double,5> kappa_s{{0.0,0.0,0.0,0.0,0.0}};
};

struct device_handle {
    int n_species_table = 0; // 3 or 6 in the original tables
    int nrho = 0, ntemp = 0, nye = 0, nymu = 0;
    double logrho_min = 0.0, logrho_max = 0.0;
    double logtemp_min = 0.0, logtemp_max = 0.0;
    double ye_min = 0.0, ye_max = 0.0;
    double logymu_min = 0.0, logymu_max = 0.0;
    bool valid = false;

    Kokkos::View<double*> logrho_axis;
    Kokkos::View<double*> logtemp_axis;
    Kokkos::View<double*> ye_axis;
    Kokkos::View<double*> logymu_axis;
    Kokkos::View<double*> kappa_a_en_table;
    Kokkos::View<double*> kappa_a_num_table;
    Kokkos::View<double*> kappa_s_table;

    GRACE_HOST_DEVICE inline int flat_index(int ispec, int irho, int itemp, int iye, int iymu) const {
        return ispec + n_species_table * (irho + nrho * (itemp + ntemp * (iye + nye * iymu)));
    }

    GRACE_HOST_DEVICE inline void clamp_state(double& rho_cgs, double& temp_mev, double& yle, double& ymu) const {
        if (!valid) return;
        const double tiny = 1.0e-300;
        double lrho = Kokkos::log(rho_cgs > tiny ? rho_cgs : tiny);
        double ltemp = Kokkos::log(temp_mev > tiny ? temp_mev : tiny);
        lrho = Kokkos::fmax(logrho_min, Kokkos::fmin(lrho, logrho_max));
        ltemp = Kokkos::fmax(logtemp_min, Kokkos::fmin(ltemp, logtemp_max));
        rho_cgs = Kokkos::exp(lrho);
        temp_mev = Kokkos::exp(ltemp);
        yle = Kokkos::fmax(ye_min, Kokkos::fmin(yle, ye_max));
        if (nymu > 1) {
            double lymu = Kokkos::log(ymu > tiny ? ymu : tiny);
            lymu = Kokkos::fmax(logymu_min, Kokkos::fmin(lymu, logymu_max));
            ymu = Kokkos::exp(lymu);
        } else {
            ymu = 0.0;
        }
    }

    GRACE_HOST_DEVICE inline void find_bracket(const Kokkos::View<double*>& axis, int n, double x, int& i0, double& w) const {
        if (n <= 1) { i0 = 0; w = 0.0; return; }
        if (x <= axis(0)) { i0 = 0; w = 0.0; return; }
        if (x >= axis(n-1)) { i0 = n-2; w = 1.0; return; }
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (axis(mid) <= x) lo = mid; else hi = mid;
        }
        i0 = lo;
        const double x0 = axis(lo), x1 = axis(lo+1);
        w = (x1 > x0) ? ((x - x0) / (x1 - x0)) : 0.0;
        w = Kokkos::fmax(0.0, Kokkos::fmin(w, 1.0));
    }

    GRACE_HOST_DEVICE inline double interp_table(const Kokkos::View<double*>& table, int ispec,
                                               double lrho, double ltemp, double ye, double lymu) const {
        int ir = 0, it = 0, iy = 0, im = 0;
        double wr = 0.0, wt = 0.0, wy = 0.0, wm = 0.0;
        find_bracket(logrho_axis, nrho, lrho, ir, wr);
        find_bracket(logtemp_axis, ntemp, ltemp, it, wt);
        find_bracket(ye_axis, nye, ye, iy, wy);
        if (nymu > 1) find_bracket(logymu_axis, nymu, lymu, im, wm);

        double out = 0.0;
        const int mz = (nymu > 1 ? 2 : 1);
        for (int dm = 0; dm < mz; ++dm) {
            const double cm = (nymu > 1 ? (dm ? wm : 1.0 - wm) : 1.0);
            const int jm = im + (nymu > 1 ? dm : 0);
            for (int dy = 0; dy < 2; ++dy) {
                const double cy = dy ? wy : 1.0 - wy;
                const int jy = iy + dy;
                for (int dt = 0; dt < 2; ++dt) {
                    const double ct = dt ? wt : 1.0 - wt;
                    const int jt = it + dt;
                    for (int dr = 0; dr < 2; ++dr) {
                        const double cr = dr ? wr : 1.0 - wr;
                        const int jr = ir + dr;
                        out += cm * cy * ct * cr * table(flat_index(ispec, jr, jt, jy, jm));
                    }
                }
            }
        }
        return out;
    }

  GRACE_HOST_DEVICE inline interp_outputs lookup(double rho_cgs, double temp_mev, double yle, double ymu) const {
      interp_outputs out;
      if (!valid) return out;
      clamp_state(rho_cgs, temp_mev, yle, ymu);
      const double lrho = Kokkos::log(rho_cgs);
      const double ltemp = Kokkos::log(temp_mev);
      const double lymu = (nymu > 1 ? Kokkos::log(ymu > 1.0e-300 ? ymu : 1.0e-300) : 0.0);

      if (n_species_table == 3) {
          for (int s = 0; s < 2; ++s) {
              out.kappa_a_en[s]  = interp_table(kappa_a_en_table,  s, lrho, ltemp, yle, lymu);
              out.kappa_a_num[s] = interp_table(kappa_a_num_table, s, lrho, ltemp, yle, lymu);
              out.kappa_s[s]     = interp_table(kappa_s_table,     s, lrho, ltemp, yle, lymu);
          }
          out.kappa_a_en[4]  = interp_table(kappa_a_en_table,  2, lrho, ltemp, yle, lymu);
          out.kappa_a_num[4] = interp_table(kappa_a_num_table, 2, lrho, ltemp, yle, lymu);
          out.kappa_s[4]     = interp_table(kappa_s_table,     2, lrho, ltemp, yle, lymu);
      } else {
          for (int s = 0; s < 5; ++s) {
              out.kappa_a_en[s]  = interp_table(kappa_a_en_table,  s, lrho, ltemp, yle, lymu);
              out.kappa_a_num[s] = interp_table(kappa_a_num_table, s, lrho, ltemp, yle, lymu);
              out.kappa_s[s]     = interp_table(kappa_s_table,     s, lrho, ltemp, yle, lymu);
          }
      }
      for (int s = 0; s < 5; ++s) {
          if (!(out.kappa_a_en[s] > 0.0) || !Kokkos::isfinite(out.kappa_a_en[s])) out.kappa_a_en[s] = 1.0e-60;
          if (!(out.kappa_a_num[s] > 0.0) || !Kokkos::isfinite(out.kappa_a_num[s])) out.kappa_a_num[s] = 1.0e-60;
          if (!(out.kappa_s[s] > 0.0) || !Kokkos::isfinite(out.kappa_s[s])) out.kappa_s[s] = 1.0e-60;
      }
      return out;
    }
};

bool weakhub_enabled_from_params();
void initialize_weakhub_from_params();
const device_handle& get_device_handle();
bool is_initialized();

}} // namespace grace::weakhub

#endif
