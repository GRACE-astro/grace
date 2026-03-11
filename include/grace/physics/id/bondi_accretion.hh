/**
 * @file bondi_accretion.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-12-16
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
#ifndef GRACE_ID_BONDI_HH
#define GRACE_ID_BONDI_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

#include <grace/utils/rootfinding.hh>

#include "kerr_schild_subexpressions.hh"

#include "bondi_subexpressions.hh"

#include <Kokkos_Core.hpp>

namespace grace {

struct bondi_params_t {
    double rc     ; //!< Critical radius 
    double gamma  ; //!< Gamma 
    double K      ; //!< Adiabat 
    double M      ; //!< Mass (1)
    double j      ; //!< Specific ang mom 
    double j_pow  ; //!< Scaling of j with theta 
    double bh_spin; //!< Spin of black hole 
    double t_min  ; //!< Lower bound of temp 
    double t_max  ; //!< Upper bound of temp 
    //! derived qtities 
    double n;
    double uc;
    double Tc;
} ; 

//! Find temperature where residual sign changes 
//! inspired by Athena !!! 
static double KOKKOS_INLINE_FUNCTION 
find_temp_range(double r, bondi_params_t const& par) 
{
    double t_min{par.t_min}, t_max{par.t_max} ; 

    auto const rootfun = [&] (double T) {
        double out ; 
        bondi_T__r(par.M,par.rc,par.n,T,r,par.Tc,par.uc,&out);
        return out ; 
    } ; 

    double const ratio = 0.3819660112501051 ; 
    int const max_iter = 40                 ;
    
    double t_mid = t_min + ratio * (t_max-t_min) ; 
    double res_mid = rootfun(t_mid) ; 

    double t_new, res_new ; 
    bool go_right = true ; 
    for( int it=0; it<max_iter; ++it) {
        if ( res_mid < 0 ) {
            return t_mid ; 
        } 

        if ( go_right ) {
            t_new = t_mid + ratio * (t_max-t_mid); 
            res_new = rootfun(t_new) ; 

            if ( res_new < res_mid ) {
                t_min = t_mid ; 
                t_mid = t_new ; 
                res_mid = res_new ;
            } else {
                t_max = t_new ; 
                go_right = false ; 
            }
        } else {
            t_new = t_mid - ratio * (t_mid - t_min) ; 
            res_new = rootfun(t_new) ;
            if ( res_new < res_mid ) {
                t_max = t_mid ; 
                t_mid = t_new ; 
                res_mid = res_new ; 
            } else {
                t_min = t_new; 
                go_right = true ; 
            }
        }
    }
    return -1; 
}


static double KOKKOS_INLINE_FUNCTION 
get_temperature_bondi(double r, bondi_params_t const& par ) {
    double t_res = find_temp_range(r,par) ; 
    if ( t_res < 0 ) {
        Kokkos::abort("could not find temperature range") ; 
    }

    auto const rootfun = [=] (double T) {
        double out ; 
        bondi_T__r(par.M,par.rc,par.n,T,r,par.Tc,par.uc,&out);
        return out ; 
    };

    if ( r < par.rc ) {
        return utils::brent(rootfun, par.t_min, t_res, 1e-15) ; 
    } else if ( r > par.rc ) {
        return utils::brent(rootfun, t_res, par.t_max, 1e-15) ; 
    } else {
        return par.Tc ; 
    }
}

template < typename eos_t >
struct bondi_id_t {
    using state_t = grace::var_array_t ; 
    using view_t = Kokkos::View<double*, grace::default_space> ; 
    static constexpr size_t npoints = 1000 ; 

    bondi_id_t(
          eos_t _eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
    ) : eos(_eos), _pcoords(pcoords)
    {
        using namespace Kokkos ;
        excision_params = get_excision_params() ;

        par.rc    = get_param<double>("grmhd","bondi_flow","r_c") ; 
        par.K     = get_param<double>("grmhd","bondi_flow","K")   ;
        par.gamma = get_param<double>("grmhd","bondi_flow","gamma")   ;
        par.M     = 1. ; 
        // fixme! 
        par.j     = get_param<double>("grmhd","bondi_flow","spec_ang_mom") ; 
        par.j_pow = get_param<double>("grmhd","bondi_flow","spec_ang_mom_scaling") ; 

        par.bh_spin = get_param<double>("grmhd","bondi_flow","spin")   ;
        par.t_min = get_param<double>("grmhd","bondi_flow","temp_min")   ;
        par.t_max = get_param<double>("grmhd","bondi_flow","temp_max")   ;

        par.n = 1./(par.gamma-1.) ; 

        // Compute temperature and radial 4 velocity at the sonic  
        // point 
        bondi_uc_Tc(par.M, par.rc, par.n, &par.uc, &par.Tc) ; 
        GRACE_INFO("Into Bondi initial data, solving on radial grid.") ; 
        GRACE_INFO("Setup: r_c: {} T_c: {} u_c: {}", par.rc, par.Tc, par.uc) ; 

    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 
        double r_exc = excision_params.r_ex; 

        // TODO we assume Schwarzschild now 
        double xyz[3] ; 
        #if 0 
        coords.get_physical_coordinates(i,j,k,q,xyz) ; 
        double rtp[3] ; 
        coords.cart_to_sph(xyz,rtp) ; 
        #endif 
        xyz[0] = _pcoords(i,j,k,0,q) ; 
        xyz[1] = _pcoords(i,j,k,1,q) ; 
        xyz[2] = _pcoords(i,j,k,2,q) ; 

        // transform cks to bl coords
        double r,theta,phi ; 
        kerr_schild_to_boyer_lindquist(xyz,0.0,r_exc,&r,&theta,&phi) ; 
        // find temperature 
        double T = get_temperature_bondi(r, par) ; 

        // compute other quantities 
        double uBL[4] = {0,0,0,0}; 
        bondi_ur_rho_p__r(
            par.rc,par.n,par.K,T,r,par.Tc,par.uc,&(uBL[1]),&id.rho,&id.press
        ) ; 
        
        // for now no B field 
        id.bx = id.by = id.bz = 0.0;
        
        // four metric 
        double g4dd[4][4], g4uu[4][4] ; 
        kerr_schild_four_metric(xyz,0.0,r_exc,&g4dd,&g4uu) ; 

        // radial vector
        // TODO change this to be BL to CKS 
        double uKS[4] ; 
        //transform_vector_bl2ks(uBL,xyz,0.0,1e-6,&uKS) ; 
        uKS[1] = uBL[1] * xyz[0]/r ; 
        uKS[2] = uBL[1] * xyz[1]/r ;
        uKS[3] = uBL[1] * xyz[2]/r ;
        // get metric 
        double guu[6] ; 
        kerr_schild_adm_metric(
            xyz,0.0,r_exc,
            &id.gxx, &id.gxy, &id.gxz, &id.gyy, &id.gyz, &id.gzz,
            &guu[0], &guu[1], &guu[2], &guu[3], &guu[4], &guu[5], 
            &id.alp, &id.betax, &id.betay, &id.betaz, 
            &id.kxx, &id.kxy, &id.kxz, &id.kyy, &id.kyz, &id.kzz    
        ) ; 

        double betad[3] = {
            id.gxx * id.betax + id.gxy * id.betay + id.gxz * id.betaz,
            id.gxy * id.betax + id.gyy * id.betay + id.gyz * id.betaz,
            id.gxz * id.betax + id.gyz * id.betay + id.gzz * id.betaz
        } ; 
        double A = -SQR(id.alp) + (id.betax*betad[0] + id.betay*betad[1] + id.betaz*betad[2]) ;  // gtt 
        double B = (betad[0] * uKS[1] + betad[1] * uKS[2] + betad[2] * uKS[3]) ;  // g_ti u^i 
        double C = id.gxx * uKS[1] * uKS[1] + id.gyy * uKS[2] * uKS[2] + id.gzz * uKS[3] * uKS[3] +
            2.0 * ( id.gxy * uKS[1] * uKS[2] + id.gxz * uKS[1] * uKS[3] + id.gyz * uKS[2] * uKS[3] ) + 1.0;  // 1 + g_{ij} u^i u^j 
        double discrim = fmax(B*B - A*C,0);

        // pick future-directed
        uKS[0] = (-B - sqrt(discrim))/A ; 

        // get 3 velocity 
        id.vx = uKS[1]/(id.alp * uKS[0]) + id.betax/id.alp ; 
        id.vy = uKS[2]/(id.alp * uKS[0]) + id.betay/id.alp ; 
        id.vz = uKS[3]/(id.alp * uKS[0]) + id.betaz/id.alp ; 

        // ye
        id.ye = 0.0 ; 

        if ( excision_params.excise_by_radius and r <= ( 1e-12 + r_exc ) ) {
            id.rho = excision_params.rho_ex ; 
            id.bx = id.by = id.bz = 0.0;
            id.vx = id.vy = id.vz = 0.0;
            double temp = excision_params.temp_ex ; 
            unsigned int eoserr ; 
            id.press = eos.press__temp_rho_ye_impl(temp,id.rho,id.ye,eoserr) ; 
        }

        return id ; 
    }

    eos_t eos ; 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;

    bondi_params_t par ; 
    excision_params_t excision_params ; 

} ; 

}

#endif /*GRACE_ID_BONDI_HH*/