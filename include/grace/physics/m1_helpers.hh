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

#include <grace/utils/rootfinding.hh>

#include <grace/physics/m1_subexpressions.hh>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace grace {

enum m1_var_idx_loc : int {
    ERADL=0,
    NRADL,
    FXL,
    FYL,
    FZL,
    ZXL,
    ZYL,
    ZZL,
    N_M1_VARS
} ; 

enum m1_eas_idx_loc : int {
    KAL=0,
    KSL,
    ETAL,
    ETANL,
    KANL,
    N_M1_EAS
} ; 

using m1_prims_array_t = std::array<double,N_M1_VARS> ; // velocity included here 

using m1_cons_array_t = std::array<double,N_M1_VARS> ;

using m1_eas_array_t = std::array<double,N_M1_EAS> ; 

#define FILL_M1_PRIMS_ARRAY(primsarr,vview,aview,q,ispec,...)\
primsarr[ERADL] = vview(__VA_ARGS__,ERAD1_+ispec*GRACE_N_M1_VARS,q);          \
primsarr[NRADL] = vview(__VA_ARGS__,NRAD1_+ispec*GRACE_N_M1_VARS,q);          \
primsarr[FXL] = vview(__VA_ARGS__,FRADX1_+ispec*GRACE_N_M1_VARS,q) ;          \
primsarr[FYL] = vview(__VA_ARGS__,FRADY1_+ispec*GRACE_N_M1_VARS,q) ;          \
primsarr[FZL] = vview(__VA_ARGS__,FRADZ1_+ispec*GRACE_N_M1_VARS,q) ;          \
primsarr[ZXL] = aview(__VA_ARGS__,ZVECX_,q) ;          \
primsarr[ZYL] = aview(__VA_ARGS__,ZVECY_,q) ;          \
primsarr[ZZL] = aview(__VA_ARGS__,ZVECZ_,q) 


struct m1_closure_t {
    using vec_t = std::array<double,3> ;

    GRACE_HOST_DEVICE m1_closure_t(
        m1_prims_array_t const& prims,
        metric_array_t _metric
    ) : E(prims[ERADL])
      , FD({prims[FXL],prims[FYL],prims[FZL]})
      , vU({prims[ZXL],prims[ZYL],prims[ZZL]})
      , metric(_metric)
    {
        initialize() ; 
    }
    //! Ctor without zeta
    GRACE_HOST_DEVICE m1_closure_t(
        double _E,
        vec_t const& _FD,
        vec_t const& _vU,
        metric_array_t _metric
    ) : E(_E), FD(_FD), vU(_vU), metric(_metric)
    {
        initialize() ; 
    }
    //! Ctor with zeta 
    GRACE_HOST_DEVICE m1_closure_t(
        double _zeta,
        double _E,
        vec_t const& _FD,
        vec_t const& _vU,
        metric_array_t _metric
    ) : E(_E), zeta(_zeta), FD(_FD), vU(_vU), metric(_metric)
    {
        initialize() ; 
        // compute fluid-frame quantities and rad-pressure 
        update_closure(zeta, false) ; 
    }


    void GRACE_HOST_DEVICE
    update_closure(m1_prims_array_t const& prims, double zeta0, bool update) {
        E = prims[ERADL]; 
        FD = vec_t{prims[FXL],prims[FYL],prims[FZL]}; 
        vU = vec_t{prims[ZXL],prims[ZYL],prims[ZZL]} ; 
        initialize() ; 
        update_closure(zeta0,update) ; 
    }

    void GRACE_HOST_DEVICE
    update_closure(double (&U)[4], double zeta0, bool update) {
        E = U[0]; 
        FD = vec_t{U[1],U[2],U[3]}; 
        initialize() ; 
        update_closure(zeta0,update) ; 
    }

    void GRACE_HOST_DEVICE
    update_closure(double _E, std::array<double,3> _F, double zeta0, bool update) {
        E = _E; 
        FD = _F; 
        initialize() ; 
        update_closure(zeta0,update) ; 
    }

    void GRACE_HOST_DEVICE
    update_closure(double zeta0, bool update=true)
    {
        double ff_params[12] ; 
        m1_closure_helpers(
            W, v2, E, vdotfh, vdotF, &ff_params
        ) ; 

        // solve the closure 
        if ( v2 > 1e-15 and update ) {
            Eclosure = E ; 
            if ( zeta0 < 1e-5 or zeta0 > 1.0 ) {
                double const E2 = fmax(E*E, f2_floor) ; 
                zeta = sqrt(F2/E2) ;
                if ( F2<=f2_floor ) zeta = 0 ;
                zeta = fmin(zeta,1.) ; 
            } else {
                zeta = zeta0 ; 
            }
            auto _z_func = [=,&ff_params,this] (double const xi) {
                chi = this->closure_func(xi) ; 
                double dthin =  1.5 * chi - 0.5 ;
                double dthick = 1.5 - 1.5 * chi ;  
                double res ; 
                m1_z_rootfind(
                    W, dthin, dthick, v2, E, F,
                    vdotfh, vdotF, xi, ff_params, &res
                ) ; 
                return res;
            } ;
            zeta = utils::brent(_z_func, 0.0, 1.0, 1e-15) ; 
        } else if (update) {
            // if v == 0 the two frames coincide 
            double const E2 = fmax(E*E, f2_floor) ; 
            zeta = sqrt(F2/E2) ;
            if ( F2<=f2_floor ) zeta = 0 ;
            zeta = fmin(zeta,1.) ; 
        }

        chi = closure_func(zeta) ; 
        double athin = 1.5 * chi - 0.5 ;
        double athick = 1.5 - 1.5 * chi ; 
        // check that pointer to data() is the same as pointer to double[3]
        double HDloc[3] ; //hacky change this! 
        m1_J_Hd(
            athin,athick,v2,FD.data(),vD.data(),fhD.data(),ff_params, &J, &(HDloc)
        ); 
        HD[0] = HDloc[0] ; HD[1] = HDloc[1] ; HD[2] = HDloc[2] ; 
        #if 0
        printf(
            "E %g Fx %g Fy %g Fz %g params {%g,%g,%g,%g,%g,%g,%g,%g,%g} J %g H {%g,%g,%g}\n",
            E, FD[0], FD[1], FD[2],
            ff_params[0], ff_params[1], ff_params[2], ff_params[3], ff_params[4],
            ff_params[5], ff_params[6], ff_params[7], ff_params[8], J,
            HD[0], HD[1], HD[2]
        );
        #endif 
        HU = metric.raise(HD) ; 
        Gamma = W * (E-vdotF)/fmax(J,f_floor) ; 
    }

    void GRACE_HOST_DEVICE
    compute_pressure() {
        chi = closure_func(zeta) ; 
        double athin = 1.5 * chi - 0.5 ;
        double athick = 1.5 - 1.5 * chi ; 
        m1_PUU(
            W, athin, athick, E, F, 
            FU.data(), vU.data(),
            vdotF, metric._ginv.data(), 
            &PUU
        ) ; 
    }

    void GRACE_HOST_DEVICE 
    implicit_update_func(
        m1_eas_array_t const& eas,
        double const (&U)[4],
        double const (&W)[4],/*undensitized!*/
        double (&S)[4],
        double dt, double dtfact, bool update_zeta=true
    ) 
    {
        E = U[0] ; FD = vec_t({U[1],U[2],U[3]}) ; 
        initialize() ;
        update_closure(zeta,update_zeta) ; 
        get_implicit_sources(eas,S) ; 
        for(int i=0; i<4; ++i) {
            S[i] = dt*dtfact*S[i] + W[i] - U[i] ; 
        }
    }

    void GRACE_HOST_DEVICE 
    implicit_update_dfunc(
        m1_eas_array_t const& eas,
        double const (&U)[4],
        double const (&W)[4],/*undensitized!*/
        double (&S)[4],
        double (&J)[4][4],
        double dt, double dtfact
    ) 
    {
        E = U[0] ; FD = vec_t({U[1],U[2],U[3]}) ; 
        initialize() ; 
        update_closure(zeta,false) ; /*note we don't update here*/

        get_implicit_sources(eas,S) ; 
        for(int i=0; i<4; ++i) {
            S[i] = dt*dtfact*S[i] + W[i] - U[i] ; 
        }
        get_implicit_jacobian(eas,J) ; 
        for(int i=0; i<4; ++i) {
            for( int j=0; j<4; ++j) {
                J[i][j] = dt*dtfact*J[i][j] - (i==j) ; 
            }
        }
    }

    void GRACE_HOST_DEVICE 
    get_implicit_update_initial_guess(m1_eas_array_t const& eas, double (&u)[4], double dt, double dtfact ) {
        double Jhat = ( J + dt*dtfact/W *  eas[ETAL] )/( 1. + dt*dtfact/W * eas[KAL] ) ; 
        double Hhat[3] = {
            (HD[0])/(1.+dt*dtfact/W * (eas[KAL] + eas[KSL])),
            (HD[1])/(1.+dt*dtfact/W * (eas[KAL] + eas[KSL])),
            (HD[2])/(1.+dt*dtfact/W * (eas[KAL] + eas[KSL]))
        } ; 
        // H_\alpha uˆ\alpha = 0 -> H_0 = - H_i u^i/u^0
        double Hhat_0 = 0 ; 
        for(int i=0; i<3; ++i) {
            Hhat_0 -= Hhat[i] * ( metric.alp()*vU[i] - metric.beta(i)) ; 
        }
        // transform back assuming zeta == 0 
        double Ethick, Fthick[3] ; 
        m1_fluid_to_lab_thick(
            W, vD.data(), metric.alp(), metric._beta.data(),
            Jhat, Hhat, Hhat_0, &(Ethick), &(Fthick)
        ); 
        for( int icomp=0; icomp<3; ++icomp) {
            u[icomp+1] = Fthick[icomp] ; 
        }
        u[0] = Ethick ; 
    } 

    void GRACE_HOST_DEVICE 
    get_implicit_sources(
        m1_eas_array_t const& eas, double (&S)[4]
    )
    {
        double dE, dF[3] ; 
        m1_source( 
            W, E, vD.data(), eas[KAL], eas[KSL], eas[ETAL],
            vdotF, metric.alp(), J, HD.data(), &(dE), &(dF)
        ) ; 
        S[0] = dE ; 
        S[1] = dF[0] ; 
        S[2] = dF[1] ; 
        S[3] = dF[2] ; 
    }

    void GRACE_HOST_DEVICE 
    get_implicit_jacobian(
        m1_eas_array_t const& eas, double (&J) [4][4]
    ) 
    {
        // get chi, dthin, dthick
        double const chil = closure_func(zeta) ; 
        double const dthin  = 1.5 * chil - 0.5 ;
        double const dthick = 1.5 - 1.5 * chil ; 
        m1_jacobian(
            W, dthin, dthick, v2, E, F, vU.data(), vD.data(),
            fhU.data(), fhD.data(), eas[KAL], eas[KSL],
            vdotfh, metric.alp(), &J
        ) ; 
    }

    void GRACE_HOST_DEVICE get_N_implicit_update(
        m1_prims_array_t const& prims, m1_eas_array_t const& eas, 
        double dt, double dtfact, double *N, double *dN
    )
    {
        *N = (prims[NRADL] + metric.alp() * dt * dtfact * eas[ETANL]) / (1.0 + metric.alp() * dt * dtfact * eas[KANL]/Gamma ) ; 
        *dN = (*N - prims[NRADL])/dt/dtfact ; 
    }

    static constexpr double TINY   = 1e-20 ; 
    static constexpr double DBLMIN = 1e-200 ; 

    double f2_floor, f_floor ;
    vec_t FD, fhD, HD, vD, vU, FU, fhU, HU; 
    double E, J, F2, F, vdotF, vdotfh, Fdotfh, v2, W, W2, zeta, Gamma; 
    double PUU[3][3];
    
    metric_array_t metric ; 

    // bits and pieces 
    double chi ; 
    double Eclosure ; 


    double GRACE_HOST_DEVICE
    closure_func( double const& z ) {
        return 1./3. + SQR(z) * (6.-2.*z+6.*SQR(z))/15. ; 
    }
    
    private:

    void GRACE_HOST_DEVICE 
    initialize() {
        f2_floor = fmax(TINY * E * E, DBLMIN) ; 
        f_floor  = fmax(TINY * E    , DBLMIN) ; 

        FU = metric.raise(FD) ;
        F2 = metric.square_vec(FU) ; 
        F  = sqrt(F2) ; 

        F2 = fmax(F2, f2_floor) ; 
        F  = fmax(F , f_floor ) ; 

        // v here is actually z, we convert now 
        double const z2 = metric.square_vec(vU) ; 
        W2 = (1.0+z2) ; 
        W = sqrt(W2) ; 
        vU[0]/=W ; vU[1]/=W; vU[2]/=W ; 

        vdotF = fmax(FD[0] * vU[0] + FD[1] * vU[1] + FD[2] * vU[2], f_floor) ; 

        vD = metric.lower(vU) ; 
        v2 = metric.square_vec(vU) ; 

        fhD = vec_t({
            FD[0]/F,
            FD[1]/F,
            FD[2]/F
        }) ; 
        fhU = metric.raise(fhD) ; 
        Fdotfh = F ; 
        vdotfh = vdotF/F ; 
    }

} ; 


struct m1_atmo_params_t {
    double E_fl, eps_fl ; 
    double E_fl_scaling, eps_fl_scaling ; 
    double eps_min, eps_max ; 
} ; 

struct m1_excision_params_t {
    bool excise_by_radius ; 
    double r_ex, alp_ex ; 
    double E_ex, eps_ex ;  
} ;

static m1_excision_params_t 
get_m1_excision_params() {
    m1_excision_params_t m1_excision_params ;
    m1_excision_params.excise_by_radius = ( grace::get_param<std::string>("grmhd", "excision", "excision_criterion") == "radius"); 
    m1_excision_params.r_ex = grace::get_param<double>("grmhd", "excision", "excision_radius");
    m1_excision_params.alp_ex = grace::get_param<double>("grmhd", "excision", "excision_lapse");
    m1_excision_params.E_ex = grace::get_param<double>("m1", "excision", "E_excision") ; 
    m1_excision_params.eps_ex = grace::get_param<double>("m1", "excision", "eps_excision") ; 
    return m1_excision_params ; 
}



static m1_atmo_params_t 
get_m1_atmo_params() {
    m1_atmo_params_t m1_atmo_params ; 
    m1_atmo_params.E_fl = grace::get_param<double>("m1", "atmosphere", "E_fl") ;
    m1_atmo_params.E_fl_scaling = grace::get_param<double>("m1", "atmosphere", "E_scaling") ;
    m1_atmo_params.eps_fl = grace::get_param<double>("m1", "atmosphere", "eps_fl") ;
    m1_atmo_params.eps_fl_scaling = grace::get_param<double>("m1", "atmosphere", "eps_scaling") ;
    m1_atmo_params.eps_min = grace::get_param<double>("m1", "atmosphere", "eps_min") ; 
    m1_atmo_params.eps_max = grace::get_param<double>("m1", "atmosphere", "eps_max") ; 
    return m1_atmo_params ; 
}

}

#endif
