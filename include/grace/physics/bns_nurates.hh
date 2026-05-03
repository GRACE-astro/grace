/**
 * @file bns_nurates.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-03-26
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
#ifndef GRACE_PHYS_BNS_NURATES_HH
#define GRACE_PHYS_BNS_NURATES_HH

#include <grace_config.h>

#include <Kokkos_Core.hpp>

#include <bns_nurates.hpp>
#include <m1_opacities.hpp>

#include <grace/physics/eos/physical_constants.hh>
#include <grace/physics/eos/unit_system.hh>

namespace grace {

struct oned_quadrature_props_t 
{
    oned_quadrature_props_t() 
    : qx("nurates_quad_points",0), qw("nurates_quad_weights",0), n(0) 
    {}

    oned_quadrature_props_t(
        oned_quadrature_props_t const& other 
    ) = default ; 
    oned_quadrature_props_t(
        oned_quadrature_props_t&& other
    ) = default ; 

    fill(bns_nurates::MyQuadrature const& quad) {
        n = quad.nx ; 
        Kokkos::realloc(qx, n) ; 
        Kokkos::realloc(qw, n) ; 

        auto qxh = Kokkos::create_mirror_view(qx) ; 
        auto qwh = Kokkos::create_mirror_view(qw) ; 

        for( int i=0; i<n; ++i ) {
            qxh(i) = quad.points[i]  ; 
            qxw(i) = quad.weights[i] ; 
        }

        Kokkos::deep_copy(qx,qxh) ; 
        Kokkos::deep_copy(qw,qwh) ; 
    }

    Kokkos::View<double*, grace::default_space> qx, qw ; 
    int n ; 
} ; 


struct opacity_flags_t {
    OpacityFlags host ; 
    readonly_view_t<bool,5> device ;
} ; 

static inline oned_quadrature_props_t get_oned_quad() 
{
    using namespace bns_nurates ;

    

    // is 10 a reasonable number? 
    // should this ever change? 
    // who knows 
    MyQuadrature my_quad ; 
    my_quad.type = kGauleg ; 
    my_quad.alpha = -42 ; 
    my_quad.dim   = 1   ;
    my_quad.nx    = 10  ; 
    my_quad.ny    = my_quad.nz = 1   ;

    my_quad.x1 = 0  ;  
    my_quad.x2 = 1. ;

    my_quad.y1 = my_quad.y2 = -42 ; 
    my_quad.z1 = my_quad.z2 = -42 ; 

    GaussLegendre(&my_quad) ; 

    // copy over 
    oned_quadrature_props_t quad ; 
    quad.fill(my_quad)


    return quad ; 
}


template< typename eos_t > 
void get_bns_nurates()
{
    DECLARE_GRID_EXTENTS; 
    using namespace grace ;
    // code units to cgs 
    auto uconv = GEOM_units / CGS_units ; 

    auto& state = variable_list::get()::getstate() ; 
    auto& aux   = variable_list::get()::getaux()   ;
    auto eos =  eos::get().get_eos<eos_t>() ; 

    auto quadrature = get_oned_quad() ; 

    auto opacity_flags = get_opacity_flags() ; 
    auto opacity_pars  = get_opacity_pars()  ;

    readonly_view_t<bool,5> opacity_flags("opacity_flags") ; 
    readonly_view_t<bool,8> opacity_pars("opacity_pars")   ;

    fill_opacity_flags_and_pars(opacity_flags,opacity_pars) ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<4>> policy(
        {0,0,0,0}, {nx+2*ngz,ny+2*ngz,nz+2*ngz,nq}
    ) ; 

    Kokkos::parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "get_m1_opacities_bns_nurates"), policy,
        KOKKOS_LAMBDA( int const i, int const j, int const k, int const q) 
        {

            auto auxijk = Kokkos::subview(
                aux, i,j,k, Kokkos::ALL, q
            ) ; 
            auto stateijk = Kokkos::subview(
                state, i,j,k, Kokkos::ALL, q
            ) ;

            metric_array_t metric ; 
            FILL_METRIC_ARRAY(
                metric, state, q, i,j,k
            ) ; 

            bns_nurates::GreyOpacityParams eas_pars ; 

            eas_pars.opacity_pars  = opacity_pars  ; 
            eas_pars.opacity_falgs = opacity_flags ;

            // prims 
            double tempL = auxijk(TEMP_)  ; 
            double yeL   = auxijk(YE_  )  ;
            double rhoL  = auxijk(RHO_ )  ;
            eos_err_t err; 
            double mup, mun, Xa, Xh, Xn, Xp, Abar, Zbar ; 
            double mue = eos.mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye(
                mup,mun,Xa,Xh,Xn,Xp,Abar,Zbar,tempL,rhoL,yeL,err
            ) ; 
            
            // read in the hydro state 
            eas_pars.eos_pars.mu_e = mue ; // fixme with without restmass? 
            eas_pars.eos_pars.mu_p = mup ;
            eas_pars.eos_pars.mu_n = mun ;
            eas_pars.eos_pars.temp = tempL ; 
            eas_pars.eos_pars.yp   = yeL ; 
            eas_pars.eos_pars.yn   = 1.-yeL ; 
            eas_pars.eos_pars.nb   = rhoL * uconv.mass_density / eos.baryon_mass * 1e-21 ; // wtf is 1e-21?

            // set distr params to equilibrium (?)
            eas_pars.distr_pars = bns_nurates::NuEquilibriumParams(
                &eas_pars.eos_pars);

            // read in the M1 state 
            m1_prims_array_t pnue, pnua, pnux ; 
            FILL_M1_PRIMS_ARRAY(pnue,state,aux,q,0,VEC(i,j,k)) ; 
            FILL_M1_PRIMS_ARRAY(pnua,state,aux,q,1,VEC(i,j,k)) ; 
            FILL_M1_PRIMS_ARRAY(pnux,state,aux,q,2,VEC(i,j,k)) ; 
            // remove sqrtg 
            for( int iv=0; iv<5; ++iv ) {
                pnue[iv] /= metric.sqrtg(); 
                pnua[iv] /= metric.sqrtg(); 
                pnux[iv] /= metric.sqrtg(); 
            }

            m1_closure_t cle{pnue,metric}, cla{pnua,metric}, clx{pnux,metric} ; 
            cle.update_closure(0,true) ; 
            cla.update_closure(0,true) ; 
            clx.update_closure(0,true) ; 

            // 
            eas_pars.m1_pars.chi[bns_nurates::id_nue ] = cle.chi ; 
            eas_pars.m1_pars.chi[bns_nurates::id_anue] = cla.chi ; 
            eas_pars.m1_pars.chi[bns_nurates::id_nux ] = clx.chi ; 
            eas_pars.m1_pars.chi[bns_nurates::id_anux] = clx.chi ; 

            // MeV cm^-3
            eas_pars.m1_pars.J[bns_nurates::id_nue ] = cle.J * uconv.energy_density / constants::MeV_to_erg ; 
            eas_pars.m1_pars.J[bns_nurates::id_anue] = cla.J * uconv.energy_density / constants::MeV_to_erg ; 
            eas_pars.m1_pars.J[bns_nurates::id_nux ] = clx.J * uconv.energy_density / constants::MeV_to_erg ; 
            eas_pars.m1_pars.J[bns_nurates::id_anux] = clx.J * uconv.energy_density / constants::MeV_to_erg ; 

            // cm^-3 
            double oov = 1./(uconv.volume)
            eas_pars.m1_pars.n[bns_nurates::id_nue ] = pnue[NRADL] / cle.Gamma * oov ; 
            eas_pars.m1_pars.n[bns_nurates::id_anue] = pnua[NRADL] / cla.Gamma * oov ; 
            eas_pars.m1_pars.n[bns_nurates::id_nux ] = pnux[NRADL] / clx.Gamma * oov ; 
            eas_pars.m1_pars.n[bns_nurates::id_anux] = pnux[NRADL] / clx.Gamma * oov ; 

            double fconv = uconv.energy_density * uconv.velocity / constants::MeV_to_erg ;
            for( int ic=0; ic<4; ++ic) {
                // MeV cm^-2 s^-1 
                eas_pars.m1_pars.H[bns_nurates::id_nue ][ic] = cle.HU[ic] *  fconv ; 
                eas_pars.m1_pars.H[bns_nurates::id_anue][ic] = cla.HU[ic] *  fconv ; 
                eas_pars.m1_pars.H[bns_nurates::id_nux ][ic] = clx.HU[ic] *  fconv ; 
                eas_pars.m1_pars.H[bns_nurates::id_anux][ic] = clx.HU[ic] *  fconv ; 
            }

            // call opacities computation 
            auto eas = bns_nurates::ComputeM1Opacities(&dquad,&dquad,&eas_params) ; 

            // conversion 
            double kappa_conv = 1e7 * uconv.length ; 
            double eta_conv   = 1e21 / uconv.energy_density * uconv.time * constants::MeV_to_erg ; 
            double etan_conv  = 1e21 * uconv.volume * uconv.time ; 
            
            auxijk(KAPPAA1_)  = eas.kappa_a[bns_nurates::id_nue ] * kappa_conv ; 
            auxijk(KAPPAA2_) = eas.kappa_a[bns_nurates::id_anue] * kappa_conv ; 
            auxijk(KAPPAA3_) = (eas.kappa_a[bns_nurates::id_nux] + eas.kappa_a[bns_nurates::id_anux] )* kappa_conv ; 

            auxijk(KAPPAAN1_)  = eas.kappa_0_a[bns_nurates::id_nue ] * kappa_conv ; 
            auxijk(KAPPAAN2_) = eas.kappa_0_a[bns_nurates::id_anue] * kappa_conv ; 
            auxijk(KAPPAAN3_) = (eas.kappa_0_a[bns_nurates::id_nux] + eas.kappa_0_a[bns_nurates::id_anux] )* kappa_conv ; 

            auxijk(KAPPAS1_)  = eas.kappa_s[bns_nurates::id_nue ] * kappa_conv ; 
            auxijk(KAPPAS2_) = eas.kappa_s[bns_nurates::id_anue] * kappa_conv ; 
            auxijk(KAPPAS3_) = (eas.kappa_s[bns_nurates::id_nux] + eas.kappa_s[bns_nurates::id_anux] )* kappa_conv ; 

            auxijk(ETA1_) = eas.eta[bns_nurates::id_nue] * eta_conv ; 
            auxijk(ETA2_) = eas.eta[bns_nurates::id_anue] * eta_conv ; 
            auxijk(ETA3_) = ( eas.eta[bns_nurates::id_nux] + eas.eta[bns_nurates::id_anux] )* eta_conv ; 

            auxijk(ETAN1_) = eas.eta_0[bns_nurates::id_nue] * etan_conv ; 
            auxijk(ETAN2_) = eas.eta_0[bns_nurates::id_anue] * etan_conv ; 
            auxijk(ETAN3_) = ( eas.eta_0[bns_nurates::id_nux] + eas.eta_0[bns_nurates::id_anux] )* etan_conv ; 

        }) ; 

}


}

#endif
