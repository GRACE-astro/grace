/**
 * @file twopuncture.hh
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
#ifndef GRACE_PHYSICS_ID_TWO_PUNCTURE_HH
#define GRACE_PHYSICS_ID_TWO_PUNCTURE_HH

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

/* External includes */
#include <TwoPunctures.h>

namespace grace {

template < typename eos_t >
struct two_punctures_id_t {
    using state_t = grace::var_array_t ; 
    using sview_t = typename Kokkos::View<double ****, grace::default_space> ; 
    using vview_t = typename Kokkos::View<double *****, grace::default_space> ;

    two_punctures_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
    ) : _pcoords(pcoords), _eos(eos)
    {
        DECLARE_GRID_EXTENTS;

        _alp   = sview_t("lapse_tp", nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _beta  = vview_t("shift_tp", 3,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _g     = vview_t("metric_tp", 6,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ; 
        _k     = vview_t("ext_curv_tp", 6,nx+2*ngz,ny+2*ngz,nz+2*ngz,nq) ;

        // Initialize solver and read in parameters,
        // taken from Guercilena, Koeppel
        TP::TwoPunctures tp;

        tp.par_b = get_param<double>("grmhd","two_punctures","par_b") ; 

        tp.target_M_plus = get_param<double>("grmhd","two_punctures","target_M_plus") ;
        tp.par_P_plus[0] = get_param<double>("grmhd","two_punctures","par_P_plus_x") ;
        tp.par_P_plus[1] = get_param<double>("grmhd","two_punctures","par_P_plus_y") ;
        tp.par_P_plus[2] = get_param<double>("grmhd","two_punctures","par_P_plus_z") ;
        tp.par_S_plus[0] = get_param<double>("grmhd","two_punctures","par_S_plus_x") ;
        tp.par_S_plus[1] = get_param<double>("grmhd","two_punctures","par_S_plus_y") ;
        tp.par_S_plus[2] = get_param<double>("grmhd","two_punctures","par_S_plus_z") ;

        tp.target_M_minus = get_param<double>("grmhd","two_punctures","target_M_minus") ;
        tp.par_P_minus[0] = get_param<double>("grmhd","two_punctures","par_P_minus_x") ;
        tp.par_P_minus[1] = get_param<double>("grmhd","two_punctures","par_P_minus_y") ;
        tp.par_P_minus[2] = get_param<double>("grmhd","two_punctures","par_P_minus_z") ;
        tp.par_S_minus[0] = get_param<double>("grmhd","two_punctures","par_S_minus_x") ;
        tp.par_S_minus[1] = get_param<double>("grmhd","two_punctures","par_S_minus_y") ;
        tp.par_S_minus[2] = get_param<double>("grmhd","two_punctures","par_S_minus_z") ;

        tp.npoints_A   = get_param<size_t>("grmhd","two_punctures","npoints_A") ;
        tp.npoints_B   = get_param<size_t>("grmhd","two_punctures","npoints_B") ;
        tp.npoints_phi = get_param<size_t>("grmhd","two_punctures","npoints_phi") ;

        tp.TP_epsilon = get_param<double>("grmhd","two_punctures","TP_epsilon") ;

        GRACE_INFO("Running TwoPunctures") ; 
        tp.Run() ; 

        

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

        auto _halp = Kokkos::create_mirror_view(_alp) ; 
        auto _hbeta = Kokkos::create_mirror_view(_beta) ; 
        auto _hg = Kokkos::create_mirror_view(_g) ; 
        auto _hk = Kokkos::create_mirror_view(_k) ; 

        auto _hcoords = Kokkos::create_mirror_view(_pcoords) ;
        Kokkos::deep_copy(_hcoords,_pcoords) ; 
        GRACE_TRACE("nq {} hcoords {} alp {}", nq, _hcoords.extent(4), _halp.extent(3));
        //#pragma omp parallel for 
        for( size_t idx=0UL; idx<ncells; ++idx) {
            int i,j,k,q ; 
            std::tie(i,j,k,q) = unroll_idx(idx) ;
            // coordinates 
            double pos[3] = {
                _hcoords(i,j,k,0,q), _hcoords(i,j,k,1,q), _hcoords(i,j,k,2,q)
            } ; 
            // interpolate 
            double Q[TP::Z4VectorShortcuts::Qlen] ; 
            tp.Interpolate(pos,Q) ; 

            _halp(i,j,k,q) = Q[TP::Z4VectorShortcuts::lapse] ; 

            for( int a=0; a<6; ++a ) {
                _hg(a,i,j,k,q) = Q[TP::Z4VectorShortcuts::g11 + a] ;
                _hk(a,i,j,k,q) = Q[TP::Z4VectorShortcuts::K11 + a] ;
            }

            for( int a=0; a<3; ++a ) {
                // looking at the source it seems this is just zero
                _hbeta(a,i,j,k,q) = Q[TP::Z4VectorShortcuts::shift1 + a] ;
            }
        }

        Kokkos::deep_copy(_beta,_hbeta) ;
        Kokkos::deep_copy(_alp,_halp) ;
        Kokkos::deep_copy(_g,_hg) ;
        Kokkos::deep_copy(_k,_hk) ;

    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 

        id.rho = id.press = 0.0 ; 
        id.eps = 0.0 ; 
        id.temp = 0.0 ;
        id.entropy = 0.0 ; 
        id.vx = id.vy = id.vz = 0.0 ; 
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

    sview_t _alp ; 
    vview_t _g, _k, _beta ; 
} ; 

}


#endif 