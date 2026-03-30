/**
 * @file cloud.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-01-08
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
#ifndef GRACE_PHYSICS_ID_LORENE_BNS_HH
#define GRACE_PHYSICS_ID_LORENE_BNS_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/coordinates/coordinate_systems.hh>

/* LORENE includes */
#include <bin_ns.h>
#include <unites.h>

namespace grace {


template < typename eos_t >
struct lorene_bns_id_t {
    using state_t = grace::var_array_t ; 
    using sview_t = typename Kokkos::View<double ****, grace::default_space> ; 
    using vview_t = typename Kokkos::View<double *****, grace::default_space> ; 

    lorene_bns_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , std::string const& fname 
    ) : _pcoords(pcoords), _eos(eos)
    {
        DECLARE_GRID_EXTENTS;
        atmo_params = get_atmo_params() ;
        zero_shift = get_param<bool>("grmhd","lorene_bns","set_shift_to_zero") ; 
        reset_eps = get_param<bool>("grmhd","lorene_bns","reset_eps") ;  

        _edens = sview_t("edens_lorene", nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _vel = vview_t("vel_lorene", 3,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 

        _alp   = sview_t("lapse_lorene", nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _beta  = vview_t("shift_lorene", 3,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _g     = vview_t("metric_lorene", 6,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _k     = vview_t("ext_curv_lorene", 6,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ;


        // unit conversions 
        double const cSI = Lorene::Unites::c_si ; 
        double const GSI = Lorene::Unites::g_si ;
        double const MSI = Lorene::Unites::msol_si ;
        //double const mu0 = Lorene::Unites::mu_si ; 
        //double const eps0 = 1.0/(mu0*cSI*cSI) ; 

        units.length = GSI * MSI / (cSI*cSI) ; // m to Msun 
        units.time = units.length / cSI ; 
        units.mass = MSI ; 
        
        // kg/m^3 to Msun^-2 
        units.dens = MSI/(units.length*units.length*units.length) ; 
        units.vel = units.length/units.time / cSI ; 

        // from km to Msun 
        units.length *= 1e-03 ; 

        // read data 
        // 1) coordinates 
        const int nxg = nx + 2*ngz;
        const int nyg = ny + 2*ngz;
        const int nzg = nz + 2*ngz;
        size_t ncells = nxg*nyg*nzg*nq ; 

        auto unroll_idx = [=] (int idx) {
            size_t tmp = idx;

            const int i = tmp % nxg;
            tmp /= nxg;

            const int j = tmp % nyg;
            tmp /= nyg;

            const int k = tmp % nzg;
            tmp /= nzg;

            const int q = tmp;

            return std::make_tuple(i,j,k,q) ; 
        } ; 

        double *xc = new double[ncells] ; 
        double *yc = new double[ncells] ; 
        double *zc = new double[ncells] ; 

        
        #pragma omp parallel for
        for( size_t idx=0UL; idx<ncells; ++idx) {
            int i,j,k,iq ; 
            std::tie(i,j,k,iq) = unroll_idx(idx) ; 
                        
            auto xyz = grace::get_physical_coordinates(
                {static_cast<size_t>(i),static_cast<size_t>(j),static_cast<size_t>(k)}, iq, {0.5,0.5,0.5}, true
            ) ; 
            xc[idx] = xyz[0] * units.length ; 
            yc[idx] = xyz[1] * units.length ; 
            zc[idx] = xyz[2] * units.length ; 
        }

        // 2) call LORENE 
        spdlog::stopwatch sw ; 
        auto * bns = new Lorene::Bin_NS(
            ncells, xc,yc,zc, fname.c_str()
        ) ; 
        GRACE_VERBOSE("LORENE data read complete, time elapsed {} s", sw) ; 
        // temporary fix!! 
        Kpoly = bns->kappa_poly1 * pow((pow(cSI, 6.0) /
             ( pow(GSI, 3.0) * MSI * MSI *
               Lorene::Unites::rhonuc_si )),bns->gamma_poly1-1.);
        gamma_poly = bns->gamma_poly1 ; 

        GRACE_VERBOSE("LORENE K [GRACE units]: {}", Kpoly) ; 

        delete[] xc ;
        delete[] yc ;
        delete[] zc ;

        // 3) read fields into host buffers
        auto _he = Kokkos::create_mirror_view(_edens) ; 
        auto _hv = Kokkos::create_mirror_view(_vel) ; 

        auto _halp = Kokkos::create_mirror_view(_alp) ; 
        auto _hbeta = Kokkos::create_mirror_view(_beta) ; 
        auto _hg = Kokkos::create_mirror_view(_g) ; 
        auto _hk = Kokkos::create_mirror_view(_k) ; 
        
        #pragma omp parallel for
        for( size_t idx=0UL; idx<ncells; ++idx) {
            int i,j,k,q ; 
            std::tie(i,j,k,q) = unroll_idx(idx) ; 

            // ADM 
            _halp(i,j,k,q)    = bns->nnn[idx] ; 
            if (!zero_shift) {
                _hbeta(0,i,j,k,q) = -bns->beta_x[idx] ; 
                _hbeta(1,i,j,k,q) = -bns->beta_y[idx] ; 
                _hbeta(2,i,j,k,q) = -bns->beta_z[idx] ;
            }            

            double gdd[3][3] ; 
            _hg(0,i,j,k,q) = gdd[0][0] = bns->g_xx[idx] ; 
            _hg(1,i,j,k,q) = gdd[0][1] = gdd[1][0] = bns->g_xy[idx] ; 
            _hg(2,i,j,k,q) = gdd[0][2] = gdd[2][0] = bns->g_xz[idx] ; 
            _hg(3,i,j,k,q) = gdd[1][1] = bns->g_yy[idx] ; 
            _hg(4,i,j,k,q) = gdd[1][2] = gdd[2][1] = bns->g_yz[idx] ; 
            _hg(5,i,j,k,q) = gdd[2][2] = bns->g_zz[idx] ; 

            // note the curvature is not dimensionless! 
            _hk(0,i,j,k,q) = units.length * bns->k_xx[idx] ; 
            _hk(1,i,j,k,q) = units.length * bns->k_xy[idx] ; 
            _hk(2,i,j,k,q) = units.length * bns->k_xz[idx] ; 
            _hk(3,i,j,k,q) = units.length * bns->k_yy[idx] ; 
            _hk(4,i,j,k,q) = units.length * bns->k_yz[idx] ; 
            _hk(5,i,j,k,q) = units.length * bns->k_zz[idx] ; 

            // velocity 
            double velu[3] = {
                bns->u_euler_x[idx],
                bns->u_euler_y[idx],
                bns->u_euler_z[idx]
            } ; 

            double v2 = 0 ;  
            for( int ii=0; ii<3; ++ii) {
                for( int jj=0; jj<3; ++jj){
                    v2 += gdd[ii][jj] * velu[ii] * velu[jj] ; 
                }
            }
            double W = 1./sqrt(1.-v2) ; 
            // guard against garbage 
            if ( std::isnan(W) ) {
                double fact = sqrt((1.-1e-10)/v2) ; 
                velu[0] *= fact ; 
                velu[1] *= fact ; 
                velu[2] *= fact ; 
            }

            _hv(0,i,j,k,q) = velu[0] ; 
            _hv(1,i,j,k,q) = velu[1] ; 
            _hv(2,i,j,k,q) = velu[2] ; 

            // energy density 
            // I found several inconsistencies when extracting rho and eps individually
            // from lorene.. I guess this is the sane way of doing it since the total 
            // energy density couples directly to gravity? 
            _he(i,j,k,q) = bns->nbar[idx] / units.dens * ( 1. + bns->ener_spec[idx] ) ; 
        }

        delete bns ; 

        // 4) copy data to device 
        Kokkos::deep_copy(_edens,_he) ;
        Kokkos::deep_copy(_vel,_hv) ;
        Kokkos::deep_copy(_beta,_hbeta) ;
        Kokkos::deep_copy(_alp,_halp) ;
        Kokkos::deep_copy(_g,_hg) ;
        Kokkos::deep_copy(_k,_hk) ;

    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 
        //-----------------------------------------
        // hydro 
        //-----------------------------------------
        eos_err_t eos_err ;
        double edens = _edens(i,j,k,q) ; 
        id.rho = _eos.rho__energy_cold_impl(edens, eos_err) ; 
        id.eps = edens/id.rho - 1. ; 
        //-----------------------------------------
        // CM:
        // fixme what to do with eos_err here? I expect 
        // this will fire a lot of eps lower bound errors 
        // since the temperature is supposed to be exactly 
        // the minimum. We can ignore it probably. Maybe 
        // a better thing would be not to use the exact 
        // tmin but something a little above for cold tables?
        // 
        //-----------------------------------------
        //-----------------------------------------
        //-----------------------------------------
        double rho_atm  = atmo_params.rho_fl  ; 
        double ye_atm   = atmo_params.ye_fl   ;
        double temp_atm = atmo_params.temp_fl ;
        //-----------------------------------------
        //-----------------------------------------
        if ( id.rho < (1.+1e-3) * rho_atm || !Kokkos::isfinite(id.rho)) {
            id.rho   = rho_atm  ; 
            id.ye    = ye_atm   ; 
            id.temp  = temp_atm ; 
            // get the rest 
            double dummy ;
            id.press = _eos.press_eps_csnd2_entropy__temp_rho_ye(
                id.eps, dummy, id.entropy, id.temp, id.rho, id.ye, eos_err
            ) ; 
            // set velocities 
            id.vx = id.vy = id.vz = 0.0 ;
        } else {
            // set velocities 
            id.vx = _vel(0,i,j,k,q) ; 
            id.vy = _vel(1,i,j,k,q) ; 
            id.vz = _vel(2,i,j,k,q) ; 
            // ye 
            id.ye = _eos.ye_cold__rho(id.rho, eos_err) ; 
            //
            // reset eps if needed
            //
            // CM what this means is the following:
            //
            // if ! reset_eps, then we take 
            // eps = e_LORENE / rho - 1 
            // where rho was obtained from e and the table above.
            // this is consistent with the total energy density,
            // but it might be inconsistent with the table if a
            // bound was hit. 
            // 
            // if reset_eps, then we take
            // 
            // eps = eps_tab(rho, T_0, ye_beta_eq)
            //
            // where rho is again from above. Strictly speaking, 
            // this **can** break consistency with the solution. 

            // basically the choice is between shooting your right 
            // and left foot.

            if (reset_eps) {
                // assume "zero" temperature
                // for ideal gas t_atmo **must** be 
                // K rho_atmo^(Gamma-1) with K from the
                // ID for this to be self-consistent.
                double h, csnd2 ;
                id.temp = temp_atm ; 
                id.press = _eos.press_eps_csnd2_entropy__temp_rho_ye_impl(
                    id.eps, csnd2, id.entropy, id.temp, id.rho, id.ye, eos_err
                ) ; 
            } else {
                // get pressure and the rest assuming eps from
                // the ID is good enough
                double h, csnd2 ;
                id.press = _eos.press_h_csnd2_temp_entropy__eps_rho_ye(
                    h, csnd2, id.temp, id.entropy, id.eps, id.rho, id.ye, eos_err
                ) ; 
            }
            
        }

        // B field is set elsewhere    
        id.bx = id.by = id.bz = 0.0 ; 

        // metric 
        id.alp = _alp(i,j,k,q) ; 

        id.betax = _beta(0,i,j,k,q);
        id.betay = _beta(1,i,j,k,q);
        id.betaz = _beta(2,i,j,k,q);

        id.gxx = _g(0,i,j,k,q) ; 
        id.gxy = _g(1,i,j,k,q) ; 
        id.gxz = _g(2,i,j,k,q) ; 
        id.gyy = _g(3,i,j,k,q) ; 
        id.gyz = _g(4,i,j,k,q) ; 
        id.gzz = _g(5,i,j,k,q) ;
        
        id.kxx = _k(0,i,j,k,q) ; 
        id.kxy = _k(1,i,j,k,q) ; 
        id.kxz = _k(2,i,j,k,q) ; 
        id.kyy = _k(3,i,j,k,q) ; 
        id.kyz = _k(4,i,j,k,q) ; 
        id.kzz = _k(5,i,j,k,q) ;

        return id ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    atmo_params_t atmo_params                      ;  //!< Atmosphere properties 
    double Kpoly, gamma_poly ; 
    bool zero_shift, reset_eps ;    

    sview_t _edens, _alp ; 
    vview_t _vel, _g, _k, _beta ; 


    // unit conversions 
    struct unit_conversions {
        double length, time, mass, dens, vel ; 
    } units ; 
    
} ; 

}

#endif /* GRACE_PHYSICS_ID_LORENE_BNS_HH */