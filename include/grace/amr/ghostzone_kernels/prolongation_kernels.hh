/**
 * @file restrict_kernels.hh
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

#ifndef GRACE_AMR_BC_PROLONG_HH
#define GRACE_AMR_BC_PROLONG_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/utils/limiters.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {


template< typename interpolator, element_kind_t elem_kind, typename view_t > 
struct prolong_op {

    view_t view, cbuf ; 
    readonly_view_t<size_t> view_qid, cbuf_qid ; 
    readonly_view_t<uint8_t> eid ; 

    index_transformer_t transf; 

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        view = alias.get<stag>() ; 
    }

    prolong_op(
        view_t _view, view_t _cbuf, 
        Kokkos::View<size_t*> _view_qid,
        Kokkos::View<size_t*> _cbuf_qid,
        Kokkos::View<uint8_t*> _eid,
        size_t n, size_t ngz 
    ) : view(_view), cbuf(_cbuf)
      , view_qid(_view_qid)
      , cbuf_qid(_cbuf_qid)
      , eid(_eid)
      , transf(n,n,n,ngz,STAG_CENTER)
    {} 

    // this loop goes full nx 
    KOKKOS_INLINE_FUNCTION
    void operator() (size_t i, size_t j, size_t k, size_t iv, size_t iq) const 
    {
        auto qid = view_qid(iq) ; 
        auto cid = cbuf_qid(iq) ; 
        auto e_id = eid(iq) ; 

        // transform
        size_t i_c,j_c,k_c ; 
        transf.compute_indices<elem_kind,false>(
            i/2,j/2,k/2, i_c,j_c,k_c, e_id, true /* half nx */, true /* stagger gz by g/2 for lower ghosts*/
        ) ; 

        size_t i_f,j_f,k_f ; 
        transf.compute_indices<elem_kind,false>(
            i,j,k, i_f,j_f,k_f, e_id, false 
        ) ;

        int signs[] = {
            i_f%2 ? +1 : -1,
            j_f%2 ? +1 : -1,
            k_f%2 ? +1 : -1
        } ; 

        view(VEC(i_f,j_f,k_f),iv,qid) = interpolator::interpolate(
                                            VEC(i_c,j_c,k_c),
                                            cid,iv, 
                                            VEC(signs[0],signs[1],signs[2]),
                                            cbuf
                                        ) ; 
        
    }

} ; 


// specialize for faces
template< element_kind_t elem_kind, typename view_t > 
struct div_free_prolong_op {

    view_t view_x,view_y,view_z, cbuf_x,cbuf_y,cbuf_z ; 
    readonly_view_t<size_t> view_qid, cbuf_qid ; 
    readonly_view_t<uint8_t> eid ; 
    Kokkos::View<int8_t***> have_fine_data ; 
    // _have_fine_data(0,0,iq) -> lower x face has fine data ? 
    // _have_fine_data(0,1,iq) -> upper x face has fine data ? 

    index_transformer_t transf; // TODO 

    void set_data_ptr(view_alias_t alias) 
    {
        view_x = alias.get<STAG_FACEX>() ; 
        view_y = alias.get<STAG_FACEY>() ; 
        view_z = alias.get<STAG_FACEZ>() ; 
    }

    div_free_prolong_op(
        view_t _view_x, view_t _view_y, view_t _view_z, 
        view_t _cbuf_x, view_t _cbuf_y, view_t _cbuf_z, 
        Kokkos::View<size_t*> _view_qid,
        Kokkos::View<size_t*> _cbuf_qid,
        Kokkos::View<uint8_t*> _eid,
        Kokkos::View<int8_t***> _have_fine_data,
        size_t n, size_t ngz 
    ) : view_x(_view_x), view_y(_view_y), view_z(_view_z)
      , cbuf_x(_cbuf_x), cbuf_y(_cbuf_y), cbuf_z(_cbuf_z)
      , view_qid(_view_qid)
      , cbuf_qid(_cbuf_qid)
      , eid(_eid)
      , have_fine_data(_have_fine_data)
      , transf(n,n,n,ngz,STAG_CENTER) 
      /* TODO fixme ? --> not really, we don't need any staggering since i==ng is handled separately by the idx+1 check*/
    {} 

    template< typename sview_t >
    KOKKOS_INLINE_FUNCTION
    void fill_inside_face(
          size_t i_c, size_t j_c, size_t k_c
        , size_t i_f, size_t j_f, size_t k_f, size_t ivar
        , sview_t& u, sview_t& v, sview_t& w 
        , sview_t& U, sview_t& V, sview_t& W 
        , minmod const& limiter 
        , bool fillx, bool filly, bool fillz ) const
    {
        double Uy,Uz,Vx,Vz,Wx,Wy ;
        // compute first order slopes U_y, U_z, V_x, V_z, W_x, W_y
        // where e.g. Uy_{2,0,0} = slope_limiter(1/4 (U_{2,0,0}-U_{2,-4,0}), 1/4 (U_{2,4,0}-U_{2,0,0})) (TR 5)
        // NB the factor 2 difference is due to the fact that we don't do a central stencil but a slope limited 
        // derivative
        if ( fillx ) {
            Uy = 0.25 * limiter(U(i_c,j_c,k_c,ivar) - U(i_c,j_c-1,k_c,ivar), U(i_c,j_c+1,k_c,ivar) - U(i_c,j_c,k_c,ivar)) ; 
            Uz = 0.25 * limiter(U(i_c,j_c,k_c,ivar) - U(i_c,j_c,k_c-1,ivar), U(i_c,j_c,k_c+1,ivar) - U(i_c,j_c,k_c,ivar)) ; 
        }
        
        if ( filly ) {
            Vx = 0.25 * limiter(V(i_c,j_c,k_c,ivar) - V(i_c-1,j_c,k_c,ivar), V(i_c+1,j_c,k_c,ivar) - V(i_c,j_c,k_c,ivar)) ; 
            Vz = 0.25 * limiter(V(i_c,j_c,k_c,ivar) - V(i_c,j_c,k_c-1,ivar), V(i_c,j_c,k_c+1,ivar) - V(i_c,j_c,k_c,ivar)) ;
        }
        
        if ( fillz ) {
            Wx = 0.25 * limiter(W(i_c,j_c,k_c,ivar) - W(i_c-1,j_c,k_c,ivar), W(i_c+1,j_c,k_c,ivar) - W(i_c,j_c,k_c,ivar)) ; 
            Wy = 0.25 * limiter(W(i_c,j_c,k_c,ivar) - W(i_c,j_c-1,k_c,ivar), W(i_c,j_c+1,k_c,ivar) - W(i_c,j_c,k_c,ivar)) ;
        }
        //printf("Uy %f, Uz %f, Vx %f, Vz %f, Wx %f, Wy %f\n", Uy, Uz, Vx, Vz, Wx, Wy) ; 
        // here we fill 
        // u_{+-2, j, k} = 1/4 (U_{+-2,0,0} + j Uy_{+-2,0,0} + k Uz_{+-2,0,0}) (TR 4.1)
        // v_{j, +-2, k} = 1/4 (V_{0,+-2,0} + j Vx_{0,+-2,0} + k Vz_{0,+-2,0}) (TR 4.2)
        // w_{j, k, +-2} = 1/4 (W_{0,0,+-2} + j Wx_{0,0,+-2} + k Wy_{0,0,+-2}) (TR 4.2)
        // NB the factor 4 difference is due to the fact that in TR the field are fluxes but here 
        // they are averages
        for( int jj=0; jj<=+1; jj+=1) {
            for( int kk=0; kk<=+1; kk+=1) {
                int js = jj ? +1 : -1 ; 
                int ks = kk ? +1 : -1 ; 
                if ( fillx ) u(i_f     ,j_f+jj  ,k_f+kk  ,ivar) = ( U(i_c,j_c,k_c,ivar) + js * Uy + ks * Uz ) ; 
                if ( filly ) v(i_f+jj  ,j_f     ,k_f+kk  ,ivar) = ( V(i_c,j_c,k_c,ivar) + js * Vx + ks * Vz ) ; 
                if ( fillz ) w(i_f+jj  ,j_f+kk  ,k_f     ,ivar) = ( W(i_c,j_c,k_c,ivar) + js * Wx + ks * Wy ) ; 
            }
        }
    } ; 
        

    // the logic behind stag center is that we don't want 
    // the loop to be shifted in any direction. In fact 
    // we have explicit checks to avoid replacing the fine 
    // data on the face of the quad and anyway the other 
    // two fields (other two staggerings) need to start at
    // n + ng always. 

    // CHECK I think that looping in reverse direction for lower 
    // faces / edges / corners is fine here, instead of filling 
    // u_{-2,j,k} we fill u_{+2,j,k} but the rest should be equivalent. 
    
    template< typename team_handle_t >
    KOKKOS_INLINE_FUNCTION
    void operator() (team_handle_t const& team) const 
    {
        using namespace Kokkos ; 
        // block-idx maps to the element index 
        auto const iq = team.league_rank() ; 
        // extract quadrant and cbuf ids
        auto qid = view_qid(iq) ; 
        auto cid = cbuf_qid(iq) ; 
        auto e_id = eid(iq) ; 
        // get some subviews real quick
        // fine views
        auto const u = subview(
            view_x, VEC(ALL(),ALL(),ALL()), ALL(), qid 
        ) ; 
        auto const v = subview(
            view_y, VEC(ALL(),ALL(),ALL()), ALL(), qid 
        ) ; 
        auto const w = subview(
            view_z, VEC(ALL(),ALL(),ALL()), ALL(), qid 
        ) ; 
        // coarse views
        auto const U = subview(
            cbuf_x, VEC(ALL(),ALL(),ALL()), ALL(), cid 
        ) ; 
        auto const V = subview(
            cbuf_y, VEC(ALL(),ALL(),ALL()), ALL(), cid 
        ) ; 
        auto const W = subview(
            cbuf_z, VEC(ALL(),ALL(),ALL()), ALL(), cid 
        ) ; 

        size_t extents[4] ;
        if constexpr ( elem_kind == FACE ) {
            extents[0] = transf.ngz/2; 
            extents[1] = extents[2] = transf.nx/2 ;
            extents[3] = u.extent(GRACE_NSPACEDIM);
        } else if constexpr ( elem_kind == EDGE ) {
            extents[0] = extents[1] = transf.ngz/2; 
            extents[2] = transf.nx / 2;
            extents[3] = u.extent(GRACE_NSPACEDIM);
        } else {
            extents[0] = extents[1] = extents[2] = transf.ngz/2; 
            extents[3] = u.extent(GRACE_NSPACEDIM);
        }

        TeamThreadMDRange<Rank<4>, team_handle_t> range(team,extents[0],extents[1],extents[2],extents[3]) ;
        
        // create a minmod limiter 
        minmod limiter {}; 

        // In all these kernels:
        // i_c, j_c, k_c loop over coarse cells --> -2 in TR notation
        // i_f, j_f, k_f are the corresponding fine indices --> also -2 in TR notation

        // phase 1: fill data in fine faces shared
        //          with coarse faces 
        parallel_for(range, 
            [=, this](int i, int j, int k, int ivar)
            {
                size_t i_f,j_f,k_f ; 
                transf.compute_indices<elem_kind,false>(
                    2*i,2*j,2*k, i_f,j_f,k_f, e_id, false 
                ) ;
                size_t i_c,j_c,k_c ; 
                transf.compute_indices<elem_kind,false>(
                    i,j,k, i_c,j_c,k_c, e_id, true /* half nx */, true /*stag g/2*/
                ) ; 

                // we don't want to fill if we have fine data 
                bool fill_x{true}, fill_y{true}, fill_z{true} ; 
                if ( i_f == transf.first_index<elem_kind>(0,e_id,false/*half ncells*/) and have_fine_data(0,0,iq) ) {
                    fill_x = false ; 
                }
                if ( j_f == transf.first_index<elem_kind>(1,e_id,false/*half ncells*/) and have_fine_data(1,0,iq) ) {
                    fill_y = false ; 
                }
                if ( k_f == transf.first_index<elem_kind>(2,e_id,false/*half ncells*/) and have_fine_data(2,0,iq) ) {
                    fill_z = false ; 
                }

                fill_inside_face(
                    i_c,j_c,k_c,
                    i_f,j_f,k_f,ivar,
                    u,v,w,
                    U,V,W,
                    limiter,
                    fill_x,fill_y,fill_z) ; 
                
                // now we need to fill the last face at the end 
                // if there is no fine data 
                if ( (i_f == transf.last_index<elem_kind>(0,e_id,false/*half ncells*/) and (not have_fine_data(0,1,iq))) ) {
                    fill_inside_face(
                        i_c+1,j_c,k_c,
                        i_f+2,j_f,k_f,ivar,
                        u,v,w,
                        U,V,W,
                        limiter,
                        true,false,false) ; 
                }
                if ( (j_f == transf.last_index<elem_kind>(1,e_id,false/*half ncells*/) and (not have_fine_data(1,1,iq))) ) {
                    fill_inside_face(
                        i_c,j_c+1,k_c,
                        i_f,j_f+2,k_f,ivar,
                        u,v,w,
                        U,V,W,
                        limiter,
                        false,true,false) ; 
                }
                if ( (k_f == transf.last_index<elem_kind>(2,e_id,false/*half ncells*/) and (not have_fine_data(2,1,iq))) ) {
                    fill_inside_face(
                        i_c,j_c,k_c+1,
                        i_f,j_f,k_f+2,ivar,
                        u,v,w,
                        U,V,W,
                        limiter,
                        false,false,true) ; 
                }

            }
        ) ; 
        team.team_barrier() ; 
        // phase 2:
        // fill all faces not shared with coarse cell
        parallel_for(range, 
            [=, this](int i, int j, int k, int ivar)
            {
                size_t i_f,j_f,k_f ; 
                transf.compute_indices<elem_kind,false>(
                    2*i,2*j,2*k, i_f,j_f,k_f, e_id, false 
                ) ;
                size_t i_c,j_c,k_c ; 
                transf.compute_indices<elem_kind,false>(
                    i,j,k, i_c,j_c,k_c, e_id, true /* half nx */, true /*stag gz*/
                ) ; 

                // Compute 
                // Uxx = 1/8 sum_{i,j,k=+-1} ij v_{i,2j,k} + ik w_{i,j,2k} (TR 11)
                // Vyy = 1/8 sum_{i,j,k=+-1} ij u_{2i,j,k} + jk w_{i,j,2k} (TR 11)
                // Wzz = 1/8 sum_{i,j,k=+-1} ik u_{2i,j,k} + jk v_{i,2j,k} (TR 11)
                // Uxyz = 1/8 sum_{i,j,k=+-1} ijk u_{2i,j,k} / ( (dy)^2 + (dz)^2 ) (TR 12)
                // Vxyz = 1/8 sum_{i,j,k=+-1} ijk v_{i,2j,k} / ( (dx)^2 + (dz)^2 ) (TR 12)
                // Wxyz = 1/8 sum_{i,j,k=+-1} ijk v_{i,j,wk} / ( (dx)^2 + (dy)^2 ) (TR 12)
                double Uxx{0},Vyy{0},Wzz{0} ; 
                double Uxyz{0}, Vxyz{0}, Wxyz{0} ; 
                for( int ii=0; ii<=+1; ii+=1) {
                    for( int jj=0; jj<=+1; jj+=1) {
                        for( int kk=0; kk<=+1; kk+=1) {

                            int is = (ii ? +1 : -1) ;
                            int js = (jj ? +1 : -1) ; 
                            int ks = (kk ? +1 : -1) ; 

                            Uxx += is*js*v(i_f+ii  ,j_f+2*jj,k_f+kk,ivar) + is*ks*w(i_f+ii,j_f+jj  ,k_f+2*kk,ivar);
                            Vyy += is*js*u(i_f+2*ii,j_f+jj  ,k_f+kk,ivar) + js*ks*w(i_f+ii,j_f+jj  ,k_f+2*kk,ivar);
                            Wzz += is*ks*u(i_f+2*ii,j_f+jj  ,k_f+kk,ivar) + js*ks*v(i_f+ii,j_f+2*jj,k_f+kk  ,ivar);
                             
                            Uxyz += is*js*ks*u(i_f+2*ii,j_f+jj  ,k_f+kk  ,ivar) ; 
                            Vxyz += is*js*ks*v(i_f+ii  ,j_f+2*jj,k_f+kk  ,ivar) ; 
                            Wxyz += is*js*ks*w(i_f+ii  ,j_f+jj  ,k_f+2*kk,ivar) ; 
                        }
                    }
                }
                Uxx *= 0.125 ; Vyy *= 0.125; Wzz *= 0.125 ; 
                Uxyz *= 0.0625; Vxyz *= 0.0625; Wxyz *= 0.0625;

                // here we fill 
                // u_{0, j, k} = 1/2 (u_{2,j,k}+u_{-2,j,k}) + Uxx + k dz^2 Vxyz + j dy^2 Wxyz (TR 8)
                // v_{j, 0, k} = 1/2 (v_{j,2,k}+u_{j,-2,k}) + Vyy + j dx^2 Wxyz + k dz^2 Uxyz (TR 9)
                // w_{j, k, 0} = 1/2 (w_{j,k,2}+w_{j,k,-2}) + Wzz + j dy^2 Uxyz + k dx^2 Vxyz (TR 10)
                for( int jj=0; jj<=+1; jj++) {
                    for( int kk=0; kk<=+1; kk++) {
                        int js = jj ? +1 : -1 ; 
                        int ks = kk ? +1 : -1 ; 
                        u(i_f+1 ,j_f+jj,k_f+kk,ivar) = 0.5 * (u(i_f   ,j_f+jj,k_f+kk,ivar)+u(i_f+2 ,j_f+jj,k_f+kk,ivar)) + Uxx + ks * Vxyz + js * Wxyz ; 
                        v(i_f+jj,j_f+1 ,k_f+kk,ivar) = 0.5 * (v(i_f+jj,j_f   ,k_f+kk,ivar)+v(i_f+jj,j_f+2 ,k_f+kk,ivar)) + Vyy + js * Wxyz + ks * Uxyz ; 
                        w(i_f+jj,j_f+kk,k_f+1 ,ivar) = 0.5 * (w(i_f+jj,j_f+kk,k_f   ,ivar)+w(i_f+jj,j_f+kk,k_f+2 ,ivar)) + Wzz + ks * Uxyz + js * Vxyz ; 
                    }
                }
            }
        ) ;
        
        
    }

} ; 

} }

#endif 