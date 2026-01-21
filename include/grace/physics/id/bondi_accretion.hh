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

template < typename eos_t >
struct bondi_id_t {
    using state_t = grace::var_array_t ; 
    using view_t = Kokkos::View<double*, grace::default_space> ; 
    static constexpr size_t npoints = 1000 ; 

    bondi_id_t(
          eos_t _eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double _gamma
        , double _K
        , double _rc 
        , double _rmin
        , double _rmax
        , double _spin 
    ) : eos(_eos), _pcoords(pcoords), gamma(_gamma), n(1/(_gamma-1)), K(_K), rc(_rc), lrmin(log(_rmin)), lrmax(log(_rmax)), spin(_spin)
    {
        using namespace Kokkos ;
        excision_params = get_excision_params() ; 
        // Compute temperature and radial 4 velocity at the sonic  
        // point 
        bondi_uc_Tc(1.0/*Mass always 1*/, rc, n, &uc, &Tc) ; 
        GRACE_INFO("Into Bondi initial data, solving on radial grid.") ; 
        GRACE_INFO("Setup: r_c: {} T_c: {} u_c: {}", rc, Tc, uc) ; 
        // we solve the problem on a log r grid and store the solutions for 
        // interpolation 
        logr = view_t("log_r_Bondi", npoints) ; 
        logT = view_t("T_Bondi",npoints) ; 
        // fill radial coordinate 
        dlogr = (lrmax-lrmin)/npoints; 
        // view ptr 
        view_t alias_logr = logr ; 
        double _lrmin{lrmin}, _dlr{dlogr} ; 
        parallel_for(
            "Fill_r_Bondi",
            npoints,
            [=] GRACE_HOST_DEVICE ( int i) {
                alias_logr(i) = _lrmin + i * _dlr ; 
            }
        ) ;
        auto logr_h = create_mirror_view(logr) ; 
        deep_copy(logr_h, logr) ; 
        // solve the problem 
        // i_c is the index of the last point before the 
        // sonic radius
        int i_c = static_cast<int>((log(rc)-lrmin)/dlogr);
        auto logT_h = create_mirror_view(logT) ; 
        double _Tc(Tc), _n(n), _uc(uc) ; 
        // first we solve inwards of the sonic point 
        for( int i=i_c; i!=-1; i--) {
            double r = exp(logr_h(i)); 
            auto froot = [=] (double T, double& f, double& df) {
                    double const M = 1.0 ; 
                    bondi_T__r(_n,_uc,r,_Tc,T,M,_rc,&f,&df);
                } ;
            double Tg = i==i_c ? Tc*(1+1e-06) : exp(logT_h(i+1)); // guess 
            int rerr ; 
            double Tl = utils::rootfind_newton_raphson_unsafe(Tg,froot,30,1e-12,rerr) ; 
            if ( rerr != 1 ) {
                logT_h(i) = log(Tl) ; 
            } else {
                double f,df ; 
                froot(Tg,f,df);
                GRACE_INFO("Radius {} initial guess {} err {} fun {} dfun {}", r, Tg, rerr, f,df) ; 
                ERROR("Failed to converge to a root for temperature in inward bondi flow solve") ; 
            }
        }   
        // Then we solve outwards
        for( int i=i_c+1; i<npoints; i++) {
            double r = exp(logr_h(i)); 
            auto froot = [=] (double T, double& f, double& df) {
                    double const M = 1.0 ; 
                    bondi_T__r(_n,_uc,r,_Tc,T,M,_rc,&f,&df);
                } ;
            double Tg = i==i_c+1 ? Tc*(1-1e-06) : exp(logT_h(i-1)); // guess 
            int rerr ; 
            double Tl = utils::rootfind_newton_raphson_unsafe(Tg,froot,30,1e-12,rerr) ; 
            if ( rerr != 1 ) {
                logT_h(i) = log(Tl) ; 
            } else {
                double f,df ; 
                froot(Tg,f,df);
                GRACE_INFO("Radius {} initial guess {} err {} fun {} dfun {}", r, Tg, rerr, f,df) ; 
                ERROR("Failed to converge to a root for temperature in inward bondi flow solve") ; 
            }
        }
        deep_copy(logT,logT_h) ; 
    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 
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
        double r_eps = 1e-6; 
        kerr_schild_to_boyer_lindquist(xyz,0.0,r_eps,&r,&theta,&phi) ; 
        // log radius
        double _logr = log(r) ; 
        // find the index 
        if ( _logr < lrmin or _logr > lrmax ) {
            Kokkos::abort("Bondi grid does not cover the domain") ; 
        }
        // Interpolate temperature from solution grid 
        int ig = min(static_cast<int>((_logr-lrmin)/dlogr), npoints-2); 
        double const lambda = (_logr - logr(ig)) / dlogr;
        double _logT = (1.0 - lambda) * logT(ig) + lambda * logT(ig+1);
        // compute other quantities 
        double uBL[4] = {0,0,0,0}; 
        bondi_ur_rho_p__r(
            Tc,exp(_logT),n,rc,uc,exp(_logr),K,&(uBL[1]),&id.rho,&id.press
        ) ; 
        
        // for now no B field 
        id.bx = id.by = id.bz = 0.0;
        
        // four metric 
        double g4dd[4][4], g4uu[4][4] ; 
        r_eps = 1e-6 ; 
        kerr_schild_four_metric(xyz,0.0,r_eps,&g4dd,&g4uu) ; 

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
            xyz,0.0,0.25,
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

        if ( excision_params.excise_by_radius and r < excision_params.r_ex ) {
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

    double gamma, n, K, rc, uc, Tc, lrmin, lrmax, dlogr, spin ; 
    excision_params_t excision_params ; 
    view_t logr, logT ; 

} ; 

}

#endif /*GRACE_ID_BONDI_HH*/