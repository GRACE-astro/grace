/**
 * @file ppm_reconstruction.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief PPM reconstruction
 * @version 1.0
 * @date 2026-05-26
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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

#ifndef GRACE_UTILS_PPM_RECONSTRUCTION_HH 
#define GRACE_UTILS_PPM_RECONSTRUCTION_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/limiters.hh>
#include <grace/utils/matrix_helpers.tpp>

#include <grace/data_structures/variable_properties.hh>

namespace grace {

struct ppm_unlimited_profile_method1_t {
    template <typename ViewT>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    void operator()(ViewT& u,
                    VEC(int const i, int const j, int const k),
                    double& alp_m, double& alp_p, double& a_i, 
                    int8_t idir) const
    {
        auto const U = [&](int I) {
            return u(i + I*(idir==0),
                     j + I*(idir==1),
                     k + I*(idir==2)  );
        };

        a_i = U(0) ; 

        // Van Leer limiter 
        MCbeta lim{};
        auto const delta_a = [&](int off) {
            double const dL = U(off)   - U(off-1);
            double const dC = 0.5*(U(off+1) - U(off-1));
            double const dR = U(off+1) - U(off);
            return lim(dL, dC, dR);   // MC needs all three
        };

        double const dam = delta_a(-1);
        double const da  = delta_a( 0);
        double const dap = delta_a(+1);

        double const aip = 0.5*(U( 1) + U( 0)) - (1.0/6.0)*(dap - da );
        double const aim = 0.5*(U( 0) + U(-1)) - (1.0/6.0)*(da  - dam);

        alp_p = aip - a_i;
        alp_m = aim - a_i;
    }
};

template <int Order = 4>
struct ppm_unlimited_profile_method2_t {
    static_assert(Order == 4 || Order == 6, "only 4th or 6th order");
    static constexpr int Stencil = (Order == 4) ? 2 : 3;   // ghost cells needed

    template <typename ViewT>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    void operator()(ViewT& u,
                    VEC(int const i, int const j, int const k),
                    double& alp_m, double& alp_p,
                    int8_t idir) const
    {
        auto const U = [&](int I) {
            return u(i + I*(idir==0),
                    j + I*(idir==1),
                    k + I*(idir==2));
        };

        double aip, aim;
        if constexpr (Order == 4) {
            aip = (7.0/12.0)*(U( 1) + U( 0)) - (1.0/12.0)*(U( 2) + U(-1));
            aim = (7.0/12.0)*(U( 0) + U(-1)) - (1.0/12.0)*(U( 1) + U(-2));
        } else {  // Order == 6
            aip = (37.0/60.0)*(U( 1) + U( 0))
                - ( 8.0/60.0)*(U( 2) + U(-1))
                + ( 1.0/60.0)*(U( 3) + U(-2));
            aim = (37.0/60.0)*(U( 0) + U(-1))
                - ( 8.0/60.0)*(U( 1) + U(-2))
                + ( 1.0/60.0)*(U( 2) + U(-3));
        }

        alp_p = aip - U(0);
        alp_m = aim - U(0);
    }
};

/**
 * @brief Woodward-Colella PPM reconstruction
 * 4th order PPM reconstruction
 */
struct ppm_reconstructor_t {

    template< typename ViewT >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (
          ViewT& u
        , VEC( int const i
             , int const j
             , int const k)
        , double& uL
        , double& uR
        , int8_t idir )
    {
        double alp_p, alp_m, a_i ; 
        ppm_unlimited_profile_method1_t slp{} ; 
        slp(u,i,j,k,alp_m,alp_p,a_i,idir) ; 

        if ( alp_p * alp_m >= 0 ) {
            alp_p = alp_m = 0 ; 
        } else if ( SQR(alp_p) > 4 * SQR(alp_m) ) {
            alp_p = - 2 * alp_m ; 
        } else if ( SQR(alp_m) > 4* SQR(alp_p) ) {
            alp_m = - 2 * alp_p ; 
        }

        uL = a_i + alp_m ; 
        uR = a_i + alp_p ; 
    }

} ; 

/**
 * @brief Sekora-Colella PPM reconstruction with extremum-preserving limiter
 * 4th order, uses the new ppm limiter that preserves accuracy 
 * at extrema
 */
struct sc_reconstructor_t {

    

} ; 

}

#endif 