/**
 * @file advanced_riemann_solvers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-28
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
#ifndef GRACE_UTILS_ADVANCED_RIEMANN_SOLVERS_HH
#define GRACE_UTILS_ADVANCED_RIEMANN_SOLVERS_HH

#include <grace_config.h>

#include <grace/utils/math.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device.h> 
#include <grace/utils/metric_utils.hh>
#include <grace/data_structures/macros.hh>

#include <Kokkos_Core.hpp> 

namespace grace {


template< int idir > 
struct hllc_riemann_solver_t {
    using tetrad_t = std::array<std::array<double,4>,4> ; 

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    transform_velocities_to_tetrad_frame( double const u0
                                        , std::array<double,3>& v
                                        , std::array<double,3>& uD ) const 
    {
        std::array<double,4> umu { 
              u0
            , v[0]/u0 
            , v[1]/u0
            , v[2]/u0
        } ; 

        double const u0_tetrad = metric.contract_4dvec_4dcovec(inertial_cotetrad[0],umu) ;
        for(int ii=0; ii<3;++ii) {
            uD[ii] = metric.contract_4dvec_4dcovec(inertial_cotetrad[1+ii],umu) ;
            // In the tetrad frame lower and upper indices are the same
            v[ii]  = uD[ii] / u0_tetrad ; 
        }
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    transform_fluxes_to_eulerian_frame( grace::grmhd_cons_array_t const& cons 
                                      , std::array<double,5>& f) 
    {
        int scalar_vars_indices[] = {DENSL,TAUL,YESL,ENTSL} ; 
        for (int ivar=0; ivar<4; ++ivar) {
            f[ivar] = metric.alp() * (
                  inertial_tetrad[0][idir] * cons[ivar]
                + inertial_tetrad[idir][idir] * f[ivar] ; 
            ) ; 
        }

        std::array<double,3> stilde = {cons[STXL], cons[STYL], cons[STZL]} ; 
        std::array<double,3> fstilde = {f[STXL], f[STYL], f[STZL]} ; 
        for( int ivdir=0; ivdir<3; ++ivdir) { 
            std::array<double,3>
            double const eS = metric.contract_vec_covec(
                  stilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ; 
            double const eF = metric.contract_vec_covec(
                  fstilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ;
            f[STXL+ivdir] = metric.alp() * (
                  inertial_tetrad[0][idir] * eS 
                + inertial_tetrad[idir][idir] * eF ; 
            ) ; 
        }
    }



 private:
    grace::metric_array_t metric ; 

    tetrad_t inertial_tetrad, inertial_cotetrad ; 

} ; 


template< int idir >
void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
get_tetrad_basis( grace::metric_array_t const& metric
                , hllc_riemann_solver_t::tetrad_t& tetrad 
                , hllc_riemann_solver_t::tetrad_t& cotetrad) ;

template<>
void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
get_tetrad_basis<1>( grace::metric_array_t const& metric
                   , hllc_riemann_solver_t::tetrad_t& tetrad 
                   , hllc_riemann_solver_t::tetrad_t& cotetrad)
{
    int ww = 0 ; 

    /* e_{(t)}^\mu                       */
    tetrad[0][ww]   = 1./metric.alp() ; 
    cotetrad[0][ww] = -metric.alp()   ; ww++;

    #pragma unroll 3
    for( int ii=0; ii<3; ++ii) {
        tetrad[0][ww]   = metric.beta(ii) ; 
        cotetrad[0][ww] = 0; ww++;
    }

    /* e_{(x)}^\mu                       */
    ww = 0 ; 
    double const Bhat = 1./Kokkos::sqrt(metric.invgamma(0)) ; 
    tetrad[1][ww]   = 0 ;  
    cotetrad[1][ww] = Bhat*metric.beta(0) ; ww++;  

    #pragma unroll 3
    for( int ii=0; ii<3; ++ii) {
        tetrad[1][ww]   = B * metric.invgamma(ii) ;  
        cotetrad[1][ww] = B * utils::delta(0,ii)  ; ww++;
    }

    
    /* e_{(y)}^\mu                       */
    ww = 0 ; 
    auto const betaL = metric.lower({metric.beta(0),metric.beta(1),metric.beta(2)}) ; 
    double const Dhat = 1./Kokkos::sqrt( metric.gamma(5)*(metric.gamma(3)*metric.gamma(5) 
                                       - math::int_pow<2>(metric.gamma(4)))) ; 

    tetrad[2][ww]   = 0 ; 
    cotetrad[2][ww] = Dhat *(betaL[1] * metric.gamma(5) - betaL[2] * metric.gamma(4)) ; ww++;

    tetrad[2][ww]   = 0 ; 
    cotetrad[2][ww] = Dhat *(metric.gamma(1)*metric.gamma(5)-metric.gamma(2)*metric.gamma(4))  ; ww++;   
    
    tetrad[2][ww]   = Dhat * metric.gamma(5) ;  
    cotetrad[2][ww] = Dhat *( metric.gamma(3)*metric.gamma(5) 
                            - math::int_pow<2>(metric.gamma(4))) ; ww++;

    tetrad[2][ww]   = -Dhat * metric.gamma(4) ;    
    cotetrad[2][ww] = 0 ; 

    /* e_{(z)}^\mu                       */
    ww = 0 ;
    double const Chat = 1./Kokkos::sqrt(metric.invgamma(5)) ; 
    tetrad[3][ww]   = 0 ; 
    cotetrad[3][ww] = Chat*betaL[2] ; ww++;

    int gammacomp[3] = {2,4,5} ; // XZ YZ ZZ
    for(int ii=0; ii<3; ++ii){
        tetrad[3][ww]   = Chat*utils::delta(ii,2) ; 
        cotetrad[3][ww] = Chat*metric.gamma(gammacomp[ii]); ww++;
    }

}

}

#endif 