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

        _e   = sview_t("e_lorene", nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
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

        units.length = GSI * MSI / (cSI*cSI) ; // cm to Msun 
        units.time = units.length / cSI ; 
        units.mass = MSI ; 
        
        //units.Bfield = (1.0/units.length/sqrt(eps0*GSI/(cSI*cSI))) / 1e09 ; 

        // g/cm^3 to Msun^-2 
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
        auto * bns = new Lorene::Bin_NS(
            ncells, xc,yc,zc, fname.c_str()
        ) ; 
        GRACE_VERBOSE("LORENE data read complete.") ; 

        delete[] xc ;
        delete[] yc ;
        delete[] zc ;

        // 3) read fields into host buffers
        auto _he = Kokkos::create_mirror_view(_e) ; 
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
            _hbeta(0,i,j,k,q) = bns->beta_x[idx] ; 
            _hbeta(1,i,j,k,q) = bns->beta_y[idx] ; 
            _hbeta(2,i,j,k,q) = bns->beta_z[idx] ;

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
                bns->u_euler_x[idx] / units.vel,
                bns->u_euler_y[idx] / units.vel,
                bns->u_euler_z[idx] / units.vel
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
            _he(i,j,k,q) = bns->nbar[idx] * (1.0 + bns->ener_spec[idx]) / units.dens ; 
        }

        delete bns ; 

        // 4) copy data to device 
        Kokkos::deep_copy(_e,_he) ;
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

        // hydro 
        // recover rho from e density 
        double e = _e(i,j,k,q) ; 
        unsigned int eos_err ;
        double const rho = _eos.rho__energy_cold(e,eos_err) ; 
        double const rho_atm{1e-14} ; 
        if ( rho < (1.+1e-3) * rho_atm || !Kokkos::isfinite(rho)) {
            id.rho = rho_atm ; 
            // get ye at beta eq
            id.ye = _eos.ye_beta_eq__rho_cold(id.rho, eos_err) ;
            // get pressure from EOS
            id.press = _eos.press_cold__rho_ye(id.rho,id.ye,eos_err) ; 
            // set velocities 
            id.vx = id.vy = id.vz = 0.0 ;
        } else {
            id.rho = rho ; 
            // get ye at beta eq
            id.ye = _eos.ye_beta_eq__rho_cold(id.rho, eos_err) ;
            // get pressure from EOS
            id.press = _eos.press_cold__rho_ye(id.rho,id.ye,eos_err) ; 
            // set velocities 
            id.vx = _vel(0,i,j,k,q) ; 
            id.vy = _vel(1,i,j,k,q) ; 
            id.vz = _vel(2,i,j,k,q) ; 
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

    sview_t _e, _alp ; 
    vview_t _vel, _g, _k, _beta ; 


    // unit conversions 
    struct unit_conversions {
        double length, time, mass, dens, vel ; 
    } units ; 
    
} ; 

}

#endif /* GRACE_PHYSICS_ID_LORENE_BNS_HH */