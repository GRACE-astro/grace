/**
 * @file phys_bc_kernels.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-09-05
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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
#ifndef GRACE_AMR_PHYS_BC_KERNELS_HH
#define GRACE_AMR_PHYS_BC_KERNELS_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/physics/fd_subexpressions.hh>

#include <grace/coordinates/coordinate_systems.hh>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

struct reflect_bc_t 
{
    template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t is, int8_t js, int8_t ks), double f
      ) const
      {
        view(i,j,k) = f*view(is,js,ks) ; 
      }
} ;

template< size_t order >
struct extrap_bc_t 
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
      ) const ; 
} ; 

template<>
struct extrap_bc_t<0>
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
      ) const
      {
            view(VEC(i,j,k)) = view(VEC(i-dx,j-dy,k-dz)) ; 
      }; 
} ; 

template<>
struct extrap_bc_t<3>
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
        ) const
      {
            view(VEC(i,j,k)) =( 4*view(VEC(i-dx,j-dy,k-dz)) 
                              - 6*view(VEC(i-2*dx,j-2*dy,k-2*dz)) 
                              + 4*view(VEC(i-3*dx,j-3*dy,k-3*dz)) 
                              -   view(VEC(i-4*dx,j-4*dy,k-4*dz)) ); 
      }; 
} ;

struct sommerfeld_bc_t  
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view, view_t view_p,
            double r, double invh, double v, double f0, double s[3], double dt, double dtfact,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz),
            VEC( size_t nx, size_t ny, size_t nz), size_t ngz 
        ) const
      {
        double dudx,dudy,dudz; 
        if ( dx>0 ) {
            fd_der_2_x_l1(view_p,i,j,k,invh,&dudx) ; 
        } else if ( dx<0 ) {
            fd_der_2_x_r1(view_p,i,j,k,invh,&dudx) ; 
        } else {
            if ( i > 0 and i < nx + 2*ngz - 1 ) {
                fd_der_2_x(view_p,i,j,k,invh,&dudx) ; 
            } else if ( i == 0 ) {
                fd_der_2_x_r1(view_p,i,j,k,invh,&dudx) ; 
            } else {
                fd_der_2_x_l1(view_p,i,j,k,invh,&dudx) ; 
            }
        }
        if ( dy>0 ) {
            fd_der_2_y_l1(view_p,i,j,k,invh,&dudy) ; 
        } else if ( dy<0 ) {
            fd_der_2_y_r1(view_p,i,j,k,invh,&dudy) ; 
        } else {
            if ( j > 0 and j < ny + 2*ngz - 1 ) {
                fd_der_2_y(view_p,i,j,k,invh,&dudy) ;  
            } else if ( j ==  0 ) {
                fd_der_2_y_r1(view_p,i,j,k,invh,&dudy) ; 
            } else {
                fd_der_2_y_l1(view_p,i,j,k,invh,&dudy) ; 
            }
        }
        if ( dz>0 ) {
            fd_der_2_z_l1(view_p,i,j,k,invh,&dudz) ; 
        } else if ( dz<0 ) {
            fd_der_2_z_r1(view_p,i,j,k,invh,&dudz) ; 
        } else {
            if ( k > 0 and k < nz + 2*ngz - 1 ) {
                fd_der_2_z(view_p,i,j,k,invh,&dudz) ;  
            } else if ( k == 0 ) {
                fd_der_2_z_r1(view_p,i,j,k,invh,&dudz) ; 
            } else {
                fd_der_2_z_l1(view_p,i,j,k,invh,&dudz) ; 
            }
        }
        double dudt = -v*(s[0]*dudx + s[1]*dudy + s[2]*dudz) + (f0-view_p(i,j,k))*v/r;
        view(i,j,k) += dudt * dt * dtfact ;  // dt should be passed in
      }; 
} ; 

/**
 * @brief Apply outgoing boundary conditions.
 * \ingroup amr
 */
using outflow_bc_t = extrap_bc_t<0> ;

KOKKOS_INLINE_FUNCTION 
static void get_somm_props(int iv, double *v, double *f0) {
    #ifdef GRACE_ENABLE_Z4C_METRIC
    if ( iv == ALP_ or iv == KHAT_ ) {
        *v = sqrt(2) ; 
    } else {
        *v = 1.0 ; 
    }
    if ( iv == ALP_ or iv == GTXX_ or iv == GTYY_ or iv == GTZZ_ or iv == CHI_ ) {
        *f0 = 1.0;
    } else {
        *f0 = 0.0 ;
    }
    #else
    *f0=0.0 ; *v=1.;
    #endif 
} 

template< element_kind_t elem_kind, element_kind_t bc_kind, typename view_t >
struct phys_bc_op {

    readonly_view_t<std::size_t> qid   ;
    readonly_view_t<uint8_t> eid       ;
    readonly_twod_view_t<int8_t,3> dir ;
    readonly_twod_view_t<double,3> var_refl_fact ;

    readonly_view_t<bc_t> var_bcs      ; 

    readonly_twod_view_t<int,3> exloop       ;
    readonly_twod_view_t<int,3> offloop      ;
    
    view_t data, data_p ; 

    outflow_bc_t outflow_kernel ;
    extrap_bc_t<3> extrap_kernel ; 
    sommerfeld_bc_t sommerfeld_kernel ; 
    reflect_bc_t reflect_kernel ; 

    bool is_cbuf ; //!< If the data is cbuf set_data_ptr **must** be no-op

    var_staggering_t stag ; 

    // only one view involved, if nx needs to be 
    // halved, just do it here 
    size_t nx, ny, nz, ngz, nv ; 



    bool rx,ry,rz ; 

    double dt, dtfact ; 

    device_coordinate_system coords; 
    scalar_array_t<GRACE_NSPACEDIM> dx ; 

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        // NB this wont work if 
        // the phys boundaries
        // sit at an AMR boundary.
        // I don't see a way around 
        // the issue. 
        if (!is_cbuf) {
            data   = alias.get<stag>() ;
            data_p = alias.get_p<stag>() ;
            dt = alias._dt; dtfact = alias._dtfact;
        }

    }

    phys_bc_op(
        view_t _data, view_t _data_p, scalar_array_t<GRACE_NSPACEDIM> _dx, device_coordinate_system _coords,
        Kokkos::View<size_t*> _qid, 
        Kokkos::View<uint8_t*> _eid, 
        Kokkos::View<int8_t*[3]> _dir, 
        Kokkos::View<int*[3]> _ext,
        Kokkos::View<int*[3]> _off_l, 
        Kokkos::View<double*[3]> _var_rfact,  
        Kokkos::View<bc_t*> _var_bcs, 
         VEC(size_t _nx, size_t _ny, size_t _nz), size_t _ngz, size_t _nv, bool _is_cbuf, var_staggering_t _stag, bool _rx, bool _ry, bool _rz
    ) : qid(_qid),  eid(_eid), dir(_dir), var_refl_fact(_var_rfact), var_bcs(_var_bcs), exloop(_ext), offloop(_off_l), dx(_dx), coords(_coords),
        data(_data), data_p(_data_p),
        nx(_nx), ny(_ny), nz(_nz), ngz(_ngz), nv(_nv), is_cbuf(_is_cbuf), stag(_stag), rx(_rx), ry(_ry), rz(_rz)
    {
        outflow_kernel = outflow_bc_t{} ; extrap_kernel = extrap_bc_t<3>{} ; sommerfeld_kernel = sommerfeld_bc_t{} ; reflect_kernel = reflect_bc_t{} ; 
    }

    KOKKOS_INLINE_FUNCTION 
    void compute_zero_dir(int& lmin, int& lmax, int& idir, int& extent, uint8_t dir_idx, uint8_t eid) const {
        idir = +1;
        size_t _ncells[3] = {nx,ny,nz} ; 
        size_t const n = _ncells[dir_idx] ; 
        if constexpr (elem_kind == element_kind_t::CORNER) 
        {
            lmin = ((eid>>dir_idx) & 1) ? n + ngz : 0 ; 

            lmax = lmin + ngz;
            extent = ngz ; 
        } else if constexpr ((elem_kind == element_kind_t::EDGE) && (bc_kind == element_kind_t::FACE)) { 
            // supercalifragilistichespiralidoso 
            if(eid/4 == dir_idx) { 
                // along-edge -> full sweep
                lmin = ngz; lmax = n + ngz; idir = +1;
                extent = n ; 
            } else {
                // perpendicular -> ghost only
                if(eid < 4) {          // X-axis edges
                    lmin = ((eid>>((dir_idx+1)%2))&1) ? n + ngz : 0;
                } else if(eid < 8) {   // Y-axis edges
                    lmin = ((eid>>(dir_idx/2))&1) ? n + ngz : 0;
                } else {               // Z-axis edges
                    lmin = ((eid>>dir_idx)&1) ? n + ngz : 0;
                }
                lmax = lmin + ngz;
                extent = ngz ; 
            }

        } else {
            lmin = ngz; lmax = n + ngz ;
            extent = n ; 
        }
    }
    
    KOKKOS_INLINE_FUNCTION 
    void compute_bounds_impl(int8_t dir, int& lmin, int& lmax, int& idir, int& extent, uint8_t idx, uint8_t eid) const
    {
        size_t _ncells[3] = {nx,ny,nz} ; 
        if (dir < 0) {
            lmin = ngz - 1; lmax = -1; idir = -1; extent = ngz ; 
        } else if (dir > 0) {
            lmin = _ncells[idx] + ngz; lmax = _ncells[idx] + 2 * ngz; idir = +1;
            extent = ngz ; 
        } else {
            // anche se ti sembra che abbia un suono spaventoso 
            compute_zero_dir(lmin,lmax,idir,extent,idx,eid) ;  
        }
    };

    KOKKOS_INLINE_FUNCTION 
    void compute_bounds(
        const int8_t dir[3], int lmin[3], 
        int lmax[3], int idir[3], 
        int ext[3], int pdim[3], int npdim[3],
        uint8_t eid) const
    {
        int npc{0}, pc{0} ; 
        #pragma unroll 
        for( int ii=0; ii<3; ++ii) {
            compute_bounds_impl(dir[ii],lmin[ii],lmax[ii],idir[ii],ext[ii],ii,eid) ; 
            if ( dir[ii] != 0 ) {
                // dir nonzero -> ghostzone direction
                npdim[npc++] = ii ;
            } else {
                // dir zero -> we can parallelize 
                pdim[pc++] = ii ; 
            }
        }
    };

    #ifdef GRACE_ENABLE_Z4C_METRIC
    KOKKOS_INLINE_FUNCTION 
    void impose_algebraic_constraintz_z4c(const int ijk[3], size_t qid) const {
        auto sv = Kokkos::subview(
            data, 
            ijk[0],ijk[1],ijk[2],
            Kokkos::ALL(), qid
        );
        double gtxx = sv(GTXX_);
        double gtxy = sv(GTXY_);
        double gtxz = sv(GTXZ_);
        double gtyy = sv(GTYY_);
        double gtyz = sv(GTYZ_);
        double gtzz = sv(GTZZ_);

        double const detgt     = -(gtxz*gtxz*gtyy) + 2*gtxy*gtxz*gtyz - gtxx*(gtyz*gtyz) - gtxy*gtxy*gtzz + gtxx*gtyy*gtzz;
        double const cbrtdetgt = Kokkos::cbrt(detgt);

        gtxx/=cbrtdetgt;
        gtxy/=cbrtdetgt;
        gtxz/=cbrtdetgt;
        gtyy/=cbrtdetgt;
        gtyz/=cbrtdetgt;
        gtzz/=cbrtdetgt;

        double const gtXX=(-(gtyz*gtyz) + gtyy*gtzz) ;
        double const gtXY=(gtxz*gtyz - gtxy*gtzz)    ;
        double const gtXZ=(-(gtxz*gtyy) + gtxy*gtyz) ;
        double const gtYY=(-(gtxz*gtxz) + gtxx*gtzz) ;
        double const gtYZ=(gtxy*gtxz - gtxx*gtyz)    ;
        double const gtZZ=(-(gtxy*gtxy) + gtxx*gtyy) ; 

        double const Atxx = sv(ATXX_);
        double const Atxy = sv(ATXY_);
        double const Atxz = sv(ATXZ_);
        double const Atyy = sv(ATYY_);
        double const Atyz = sv(ATYZ_);
        double const Atzz = sv(ATZZ_);

        double const ATR = Atxx*gtXX + 2*Atxy*gtXY + 2*Atxz*gtXZ + Atyy*gtYY + 2*Atyz*gtYZ + Atzz*gtZZ ; 
        
        sv(ATXX_) -= 1./3. * gtxx * ATR ; 
        sv(ATXY_) -= 1./3. * gtxy * ATR ; 
        sv(ATXZ_) -= 1./3. * gtxz * ATR ; 
        sv(ATYY_) -= 1./3. * gtyy * ATR ; 
        sv(ATYZ_) -= 1./3. * gtyz * ATR ; 
        sv(ATZZ_) -= 1./3. * gtzz * ATR ; 

        sv(GTXX_) = gtxx ; 
        sv(GTXY_) = gtxy ; 
        sv(GTXZ_) = gtxz ; 
        sv(GTYY_) = gtyy ; 
        sv(GTYZ_) = gtyz ; 
        sv(GTZZ_) = gtzz ; 
    }
    #endif 

    KOKKOS_INLINE_FUNCTION 
    void apply_bc_impl(int iv, const int ijk[3], const int8_t _dir[3], size_t qid) const {
        auto sv = Kokkos::subview(
            data, 
            VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
            static_cast<size_t>(iv), qid 
        ) ;

        bool do_reflection = false ; 
        double fact = 1.0;
        int ijk_s[3] = {ijk[0],ijk[1],ijk[2]}; 
        // first we detect reflections
        if ( rx or ry or rz ) {
            double parities[3] = {1.,1.,1.} ; 
            if ( stag == STAG_CENTER ) {
                parities[0] = var_refl_fact(iv,0);
                parities[1] = var_refl_fact(iv,1);
                parities[2] = var_refl_fact(iv,2); 
            } else if ( stag == STAG_FACEX ) {
                // B is a pseudovector: B_d is even under d-reflection,
                // B_{j!=d} are odd
                parities[0] = 1 ;
                parities[1] = parities[2] = -1;
            } else if ( stag == STAG_FACEY ) {
                parities[1] = 1 ;
                parities[0] = parities[2] = -1;
            } else if ( stag == STAG_FACEZ ) {
                parities[2] = 1 ;
                parities[0] = parities[1] = -1;
            }

            if ( _dir[0] == -1 and rx ) {
                ijk_s[0] = ngz + (ngz-1-ijk[0]) ; 
                ijk_s[0] += stag==STAG_FACEX ? 1 : 0; 
                fact *= parities[0] ; 
                do_reflection = true ; 
            }

            if ( _dir[1] == -1 and ry ) {
                ijk_s[1] = ngz + (ngz-1-ijk[1]) ; 
                ijk_s[1] += stag==STAG_FACEY ? 1 : 0; 
                fact *= parities[1] ; 
                do_reflection = true ; 
            }

            if ( _dir[2] == -1 and rz ) {
                ijk_s[2] = ngz + (ngz-1-ijk[2]) ; 
                ijk_s[2] += stag==STAG_FACEZ ? 1 : 0; 
                fact *= parities[2] ; 
                do_reflection = true ; 
            }
        }
        if (do_reflection) {
            reflect_kernel.template apply<decltype(sv)>(sv,ijk[0],ijk[1],ijk[2],ijk_s[0],ijk_s[1],ijk_s[2],fact) ;
        } else {
            auto _bc_kind = var_bcs(iv) ;       
            switch (_bc_kind) {
                case BC_OUTFLOW:{
                    outflow_kernel.template apply<decltype(sv)>(
                        sv, VEC(ijk[0],ijk[1],ijk[2]), VEC(_dir[0], _dir[1], _dir[2]));
                    break;
                }
                case BC_LAGRANGE_EXTRAP: {
                    extrap_kernel.template apply<decltype(sv)>(
                        sv, VEC(ijk[0],ijk[1],ijk[2]), VEC(_dir[0], _dir[1], _dir[2]));
                    break;
                }
                case BC_SOMMERFELD: {
                    double vel,f0;
                    get_somm_props(iv, &vel, &f0) ; 
                    double h = dx(0,qid) ; 
                    double s[3] ; 
                    coords.get_physical_coordinates(
                        ijk[0],ijk[1],ijk[2],qid,s
                    ) ; 
                    auto sv_p = Kokkos::subview(
                        data_p, 
                        VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
                        iv, qid 
                    ) ;
                    double r = sqrt(SQR(s[0])+SQR(s[1])+SQR(s[2]));
                    s[0]/=r; s[1]/=r; s[2]/=r ; 
                    sommerfeld_kernel.template apply<decltype(sv)>(
                        sv, sv_p, r,h,vel,f0,s,dt,dtfact,VEC(ijk[0],ijk[1],ijk[2]), VEC(_dir[0], _dir[1], _dir[2]),nx,ny,nz,ngz);
                    break ; 
                }
                case BC_NONE:
                    break;
                default:
                    // fallback or assert
                    break;
            }
        }
    }
    template< typename team_handle_t >
    KOKKOS_INLINE_FUNCTION
    void operator() (
        team_handle_t const& team
    ) const 
    {
        using namespace Kokkos ; 
        auto const iq = team.league_rank() ; 

        auto _eid = eid(iq) ; 
        auto _qid = qid(iq)  ; 
          
        
        int8_t _dir[] = {dir(iq,0), dir(iq,1), dir(iq,2)} ; 
        // se lo dici forte avrà un successo strepitoso 
        int lmin[3], lmax[3], idir[3], extents[3];
        int pdim[3], npdim[3] ; 
        compute_bounds(_dir, lmin, lmax, idir, extents, pdim, npdim, _eid);

        // this is an ugly solution to the 
        // case where we are filling cbuf gzs
        // in the edge of a quadrant outside 
        // a face of the grid 
        #pragma unroll
        for( int ii=0; ii<3; ++ii) {
            extents[ii] += exloop(iq,ii) ; 
            lmin[ii] += offloop(iq,ii) ; 
        }

        // idea here is: 
        // depending on BC kind we have some number of loops 
        // which cannot be parallelized
        // anything in faces: 2 dirs parallelizable 
        // anything in edges: 1 
        // corner: all serial 
        if constexpr (bc_kind == element_kind_t::FACE) {
            TeamThreadMDRange<Rank<2>, team_handle_t> 
                range(team, extents[pdim[0]], extents[pdim[1]]) ; 
            parallel_for(
                range,
                [=,this] ( int i, int j ) {
                    int ijk[3] ; 
                    ijk[pdim[0]] = lmin[pdim[0]] + idir[pdim[0]] * i ; 
                    ijk[pdim[1]] = lmin[pdim[1]] + idir[pdim[1]] * j ; 
                    for( int k=0; k<extents[npdim[0]]; ++k) {
                        ijk[npdim[0]] = lmin[npdim[0]]+ idir[npdim[0]] * k ;
                        for( int iv=0; iv<nv; ++iv) {
                            apply_bc_impl(iv,ijk,_dir,_qid) ; 
                        }
                        #ifdef GRACE_ENABLE_Z4C_METRIC
                        if (stag==var_staggering_t::STAG_CENTER) impose_algebraic_constraintz_z4c(ijk,_qid) ; 
                        #endif 
                    }
                } 
            ) ; 

        } else if constexpr ( bc_kind == element_kind_t::EDGE ) {
            parallel_for(
                TeamThreadRange(team,extents[pdim[0]]),
                [=,this] ( int i ) {
                    int ijk[3] ; 
                    ijk[pdim[0]] = lmin[pdim[0]] + idir[pdim[0]] * i ; 
                    for( int j=0; j<extents[npdim[0]]; ++j)
                    for( int k=0; k<extents[npdim[1]]; ++k) {
                        ijk[npdim[0]] = lmin[npdim[0]]+ idir[npdim[0]] * j ;
                        ijk[npdim[1]] = lmin[npdim[1]]+ idir[npdim[1]] * k ; 
                        for( int iv=0; iv<nv; ++iv) {
                            apply_bc_impl(iv,ijk,_dir,_qid) ; 
                        }
                        #ifdef GRACE_ENABLE_Z4C_METRIC
                        if (stag==var_staggering_t::STAG_CENTER) impose_algebraic_constraintz_z4c(ijk,_qid) ; 
                        #endif 
                    }
                } 
            ) ;
        } else {
            // loop not unrollable 
            for (int kg = lmin[2]; kg != lmax[2]; kg += idir[2])
            for (int jg = lmin[1]; jg != lmax[1]; jg += idir[1])
            for (int ig = lmin[0]; ig != lmax[0]; ig += idir[0]) {
                int ijk[3] = {ig,jg,kg} ; 
                for( int iv=0; iv<nv; ++iv) {
                    apply_bc_impl(iv,ijk,_dir,_qid) ; 
                }
                #ifdef GRACE_ENABLE_Z4C_METRIC
                if (stag==var_staggering_t::STAG_CENTER) impose_algebraic_constraintz_z4c(ijk,_qid) ; 
                #endif 
            }
        }       
        
    }

    

} ; 

}} /* namespace grace::amr */
// supercalifragilistichespiralidoso! 
#endif /* GRACE_AMR_PHYS_BC_KERNELS_HH */