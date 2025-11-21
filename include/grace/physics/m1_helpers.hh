/**
 * @file m1_helpers.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2024-11-21
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
#ifndef GRACE_PHYSICS_M1_HELPERS_HH
#define GRACE_PHYSICS_M1_HELPERS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>

#include <grace/system/grace_system.hh>

#include <grace/data_structures/grace_data_structures.hh>

#include <grace/parallel/mpi_wrappers.hh>

#include <grace/utils/metric_utils.hh>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace grace {

enum m1_var_idx_loc : int {
    ERADL,
    FXL,
    FYL,
    FZL,
    N_M1_VARS
} ; 

enum m1_eas_idx_loc : int {
    KAL,
    KSL,
    ETAL,
    N_M1_EAS
} ; 

using m1_prims_array_t = std::array<double,N_M1_VARS+3> ; // velocity included here 

using m1_cons_array_t = std::array<double,N_M1_VARS> ;

using m1_eas_array_t = std::arrat<double,N_M1_EAS> ; 

struct m1_closure_t {
    using vec_t = std::array<double,3> ;
    //! Ctor without zeta
    m1_closure_t(
        double _E,
        vec_t const& _FD,
        vec_t const& _vU,
        metric_array_t _metric
    ) : E(_E), FD(_FD), vU(_vU), metric(_metric)
    {
        FU = metric.raise(FD) ;
        F2 = metric.square_vec(FU) + TINY ; 
        F = sqrt(F2) ; 
        // v here is actually z, we convert now 
        double const z2 = metric.square_vec(vU) ; 
        W2 = (1+z2) ; 
        W = sqrt(W2) ; 
        vU[0]/=W ; vU[1]/=W; vU[2]/=W ; 

        vdotF = FD[0] * vU[0] + FD[1] * vU[1] + FD[2] * vU[2] + TINY ; 
        vD = metric.lower(vU) ; 
        v2 = metric.square_vec(vU) ; 

        fhD = vec_t({
            FD[0]/F,
            FD[1]/F,
            FD[2]/F
        }) ; 
        fhU = metric.raise(fhD) ; 
        Fdotfh = F ; 
    }
    //! Ctor with zeta 
    m1_closure_t(
        double _zeta,
        double _E,
        vec_t const& _FD,
        vec_t const& _vU,
        metric_array_t _metric
    ) : E(_E), zeta(_zeta), FD(_FD), vU(_vU), metric(_metric)
    {
        FU = metric.raise(FD) ;
        F2 = metric.square_vec(FU) + TINY ; 
        F = sqrt(F2) ; 
        // v here is actually z, we convert now 
        double const z2 = metric.square_vec(vU) ; 
        W2 = (1+z2) ; 
        W = sqrt(W2) ; 
        vU[0]/=W ; vU[1]/=W; vU[2]/=W ; 

        vdotF = FD[0] * vU[0] + FD[1] * vU[1] + FD[2] * vU[2] + TINY ; 
        vD = metric.lower(vU) ; 
        v2 = metric.square_vec(vU) ; 

        fhD = vec_t({
            FD[0]/F,
            FD[1]/F,
            FD[2]/F
        }) ; 
        fhU = metric.raise(fhD) ; 
        Fdotfh = F ;  
        // compute fluid-frame quantities and rad-pressure 
        update_closure(zeta, false) ; 
    }


    void update_closure(double zeta0, double update=true)
    {
        JJ0 = W2 * (E-2.0*vdotF) ; 
        JJthin = W2 * E * vdotfh * vdotfh ; 
        JJthick = (W2-1.)/(1.+2.*W2) * (4.*W2*vdotF + (3.-2.*W2)*E) ; 

        // Hˆ2 = cn 
        double const cn = W*JJ0 + W*(vdotF-E) ; 
        double const cv = W*JJ0 ;
        double const cF = -W ;

        double const cthickn = W * JJthick ; 
        double const cthickv = W*JJthick + W / ( 1. + 2*W2 ) * ((3.-2.*W2)*E + (2.*W2-1.)*vdotF) ; 
        double const cthickF = W * v2 ; 

        double const cthinn = W * JJthin ; 
        double const cthinv = cthinn ; 
        double const cthinfh = W * E * vdotfh ; 

        HH0 = SQR(cv) * v2 + SQR(cF) * F2 + 2. * cv * cF * vdotF - SQR(cn) ; 
        HHthickthick = SQR(cthickv) * v2 + SQR(cthickF) * F2 + 2. * cthickF * cthickv * vdotF - SQR(cthickn) ; 
        HHthinthin = SQR(cthinv) * v2 + sqr(cthinfh) + 2. * cthinv * cthinfh * vdotfh - SQR(cthinn) ; 
        HHthin = 2.*(cv * cthinv *v2 + cF * cthinfh * Fdotfh  + cthinfh * cv * vdotfh + cthinv * cF * vdotF - cthinn * cn);
        HHthick = 2. * (cv * cthickv * v2 + cF * cthickF * F2 + cthickF * cv * vdotF + cthickv * cF * vdotF - cthickn * cn) ; 
        HHthickthin = 2. * (cthinv * cthickv * v2 + cthinfh * cthickF * Fdotfh + cthinfh * cthickv * vdotfh + cthinv * cthickF * vdotF - cthinn * cthickn ) ; 

        // solve the closure 
        if ( v2 > 1e-15 and update ) {
            Eclosure = E ; 
            if ( zeta0 < 1e-5 or zeta0 > 1.0 ) {
                zeta = sqrt(F2/(E*E+TINY)) ;
                if ( F2<=TINY ) zeta = 0 ;
                zeta = min(zeta,1.) ; 
            } else {
                zeta = zeta0 ; 
            }
            auto _z_func = [this,=] (double const xi) {
                return this->z_func(xi) ; 
            } ;
            zeta = utils::brent(_z_func, 0.0, 1.0, 1e-15) ; 
        } else if (update) {
            // if v == 0 the two frames coincide 
            zeta = sqrt(F2/(E*E+TINY)) ;
            if ( F2<=TINY ) zeta = 0 ;
            zeta = min(zeta,1.) ; 
        }

        chi = closure_func(zeta) ; 
        double athin = 1.5 * chi - 0.5 ;
        double athick = 1.5 - 1.5 * chi ; 

        double const Jthick = 3./(1.+2.*W2) * ((2.*W2-1.)*E-2.*vdotF*W2) ; 

        vec_t tHt ; 
        #pragma unroll 3
        for( int ii=0; ii<3; ++ii)
            tHt[ii] = FU[ii]/W + vU[ii] * W / (2.*W2+1) * ((4.*W2+1.)*vdotF - 4.*W2*E) ; 
        
        J = JJ0 + JJthick * athick + JJthin * athin ; 

        Hdotn = cn + cthickn * athick + cthinn * athin ; 

        for( int ii=0; ii<3; ++ii ) {
            HU[ii] = -(cv + athin * cthinv + athick * cthickv) * vU[ii]
                     -(cF + athick * cthickF) * FU[ii]
                     - cthinfh * fhU[ii] ; 
        }

        double icomp=0; 
        for( int ii=0; ii<3; ++ii) {
            for( int jj=ii; jj<3; ++jj) {
                double const Pthick = Jthick/3. * (4.*W2*vU[ii]*vU[jj] + metric.invgamma(icomp)) 
                                    + W * (tHt[ii] * vU[jj] + vU[ii] * tHt[jj]) ; 
                double const Pthin = W2 * E * fhU[ii] * fhU[jj] ; 
                PUU[icomp] = athick * Pthick + athin * Pthin ; 
                icomp++ ; 
            }
        }

    }

    constexpr double TINY = 1e-50 ; 

    vec_t FD, fhD, vD, vU, FU, fhU, HU; 
    double E, J, F2, F, vdotF, vdotfh, Fdotfh, v2, W, W2, zeta; 
    std::array<double,6> PUU;
    
    metric_array_t metric ; 

    // bits and pieces 
    double chi ; 
    double Eclosure ; 
    double JJ0, JJthin, JJthick ; 
    double HH0, HHthickthick, HHthinthin, HHthin, HHthick, HHthickthin;
    double Hdotn ; 


    private:

    double z_func(double const& xi) {
        double const chil = closure_func(xi) ; 
        double const athin = 1.5 * chil - 0.5 ;
        double const athick = 1.5 - 1.5 * chil ; 
        double const Jl = JJ0 + athin * JJthin + athick * JJthick ; 
        double const H2l = HH0 + athick * HHthick + athin * HHthin 
                         + athick*athin * HHthickthin + SQR(athick) * HHthickthick 
                         + SQR(athin) * HHthinthin  ; 

        return (SQR(Jl*xi) - H2l) / (SQR(Eclosure)) ; 
    }

    double closure_func( double const& z ) {
        return 1./3. + SQR(z) * (6.-2.*z+6.*SQR(z))/15. ; 
    }

}


}

#endif 