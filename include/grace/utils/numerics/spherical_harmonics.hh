/**
 * @file spherical_harmonics.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-03-27
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_UTILS_SPHERICAL_HARMONICS_HH
#define GRACE_UTILS_SPHERICAL_HARMONICS_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include <array> 

#include <Kokkos_Core.hpp>

namespace grace {


    namespace utils{

 
        GRACE_HOST_DEVICE
        static double factorial(int n)
        {
        double returnval = 1;
        for (int i = n; i >= 1; i--)
        {
            returnval *= i;
        }
        return returnval;
        }

        static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        double combination(int n, int m)
        {
        // Binomial coefficient is undefined if these conditions do not hold
        assert(n >= 0);
        assert(m >= 0);
        assert(m <= n);

        return factorial(n) / (factorial(m) * factorial(n-m));
        }


        static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        int imin(int a, int b)
        {
        return a < b ? a : b;
        }


        static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        int imax(int a, int b)
        {
        return a > b ? a : b;
        }

       /**
         * @brief Obtain the value of a spin-weighted spherical harmonic s_Y_lm(th,ph) 
         * 
         * @param s - spin weight
         * @param l - number
         * @param m 
         * @param th - polar coordinate on the sphere
         * @param ph - azimuthal coordinate on the sphere
         * @param reY - [ref] real part of the harmonic
         * @param imY - [ref] imag part of the harmonic
         * @return void
         */

        void GRACE_HOST_DEVICE
        multipole_spherical_harmonic(int s, int l, int m,
                                        const double th, const double ph,
                                        double& reY, double& imY)
        {

        double all_coeff = 0, sum = 0;
        all_coeff = Kokkos::pow(-1.0, m);
        all_coeff *= Kokkos::sqrt(factorial(l+m)*factorial(l-m)*(2*l+1) / (4.*M_PI*factorial(l+s)*factorial(l-s)));
        sum = 0.;
        for (int i = imax(m - s, 0); i <= imin(l + m, l - s); i++)
        {
            double sum_coeff = combination(l-s, i) * combination(l+s, i+s-m);
            sum += sum_coeff * Kokkos::pow(-1.0, l-i-s) * Kokkos::pow(Kokkos::cos(th/2.), 2 * i + s - m) *
            Kokkos::pow(sin(th/2.), 2*(l-i)+m-s);
        }
        reY = all_coeff*sum*Kokkos::cos(m*ph);
        imY = all_coeff*sum*Kokkos::sin(m*ph);
        }

    }
}

#endif /* GRACE_UTILS_SPHERICAL_HARMONICS_HH */