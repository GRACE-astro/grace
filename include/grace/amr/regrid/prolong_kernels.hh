/**
 * @file copy_kernels.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief This file contains the copy kernel for regrid.
 * @version 0.1
 * @date 2025-10-29
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_AMR_REGRID_PROLONG_KERNEL_HH
#define GRACE_AMR_REGRID_PROLONG_KERNEL_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/limiters.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/amr/ghostzone_kernels/pr_helpers.hh>
#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <array>

namespace grace {


template< typename interpolator_t, typename view_t >
struct regrid_prolong_op {
    view_t data_in, data_out ; 
    readonly_view_t<std::size_t> qid_in, qid_out, varidx ;
    size_t n, g ; 
    interpolator_t op ; //!< Must be copiable to device

    regrid_prolong_op(
        view_t _data_in,
        view_t _data_out,
        Kokkos::View<size_t*> _qid_in,
        Kokkos::View<size_t*> _qid_out,
        Kokkos::View<size_t*> _varidx,
        interpolator_t _op,
        size_t _n, size_t _g
    ) : data_in(_data_in)
      , data_out(_data_out)
      , qid_in(_qid_in)
      , qid_out(_qid_out)
      , varidx(_varidx)
      , op(_op)
      , n(_n), g(_g)
    {}

    KOKKOS_INLINE_FUNCTION
    void operator() (size_t i, size_t j, size_t k, size_t vidx, size_t iq) const 
    {
        using namespace Kokkos ; 
        auto qid_fine = qid_in(iq) ; 
        auto iv = varidx(vidx) ; 

        int8_t ichild = iq % P4EST_CHILDREN ; 
        int off_x = ((ichild>>0)&1) * n/2 ;
        int off_y = ((ichild>>1)&1) * n/2 ; 
        int off_z = ((ichild>>2)&1) * n/2 ; 

        auto qid_coarse = qid_out(iq/P4EST_CHILDREN) ; 

        int signs[3] = {
            (i%2) ? interpolator_t::up_cell_flag : interpolator_t::low_cell_flag,
            (j%2) ? interpolator_t::up_cell_flag : interpolator_t::low_cell_flag,
            (k%2) ? interpolator_t::up_cell_flag : interpolator_t::low_cell_flag,
        } ; 
        auto u = subview(data_out, ALL(),ALL(),ALL(), iv, qid_coarse) ; 
        data_in(i+g,j+g,k+g,iv,qid_fine) = op(
            u,
            off_x + g + i/2,
            off_y + g + j/2,
            off_z + g + k/2,
            signs[0], signs[1], signs[2]
        ) ; 
    }
} ; 
// NB here data_type is staggered_var_arrays
template< typename view_t >
struct regrid_div_free_prolong_op {
    view_t data_in, data_out ; 
    readonly_view_t<std::size_t> qid_in, qid_out ; 
    Kokkos::View<int8_t**> have_fine_data_x, have_fine_data_y, have_fine_data_z; 
    size_t n, g, nv ; 

    regrid_div_free_prolong_op(
        view_t _data_in,
        view_t _data_out,
        Kokkos::View<size_t*> _qid_in,
        Kokkos::View<size_t*> _qid_out,
        Kokkos::View<int8_t**> _have_fine_data_x,
        Kokkos::View<int8_t**> _have_fine_data_y,
        Kokkos::View<int8_t**> _have_fine_data_z,
        size_t _n, size_t _g, size_t _nv
    ) : data_in(_data_in)
      , data_out(_data_out)
      , qid_in(_qid_in)
      , qid_out(_qid_out)
      , have_fine_data_x(_have_fine_data_x)
      , have_fine_data_y(_have_fine_data_y)
      , have_fine_data_z(_have_fine_data_z)
      , n(_n), g(_g), nv(_nv)
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

    template< typename team_handle_t >
    KOKKOS_INLINE_FUNCTION
    void operator() (team_handle_t const& team) const 
    {
        using namespace Kokkos ; 
        // block-idx maps to the element index 
        auto const iq = team.league_rank() ; 
        // extract quadrant ids
        auto qid_fine = qid_in(iq) ; 
        int8_t ichild = iq % P4EST_CHILDREN ; 
        int off_x = ((ichild>>0)&1) * n/2 ;
        int off_y = ((ichild>>1)&1) * n/2 ; 
        int off_z = ((ichild>>2)&1) * n/2 ; 
        auto qid_coarse = qid_out(iq/P4EST_CHILDREN) ; 
        // get some subviews real quick
        // fine views
        auto const u = subview(
            data_in.face_staggered_fields_x, VEC(ALL(),ALL(),ALL()), ALL(), qid_fine
        ) ; 
        auto const v = subview(
            data_in.face_staggered_fields_y, VEC(ALL(),ALL(),ALL()), ALL(), qid_fine
        ) ; 
        auto const w = subview(
            data_in.face_staggered_fields_z, VEC(ALL(),ALL(),ALL()), ALL(), qid_fine
        ) ; 
        // coarse views
        auto const U = subview(
            data_out.face_staggered_fields_x, VEC(ALL(),ALL(),ALL()), ALL(), qid_coarse 
        ) ; 
        auto const V = subview(
            data_out.face_staggered_fields_y, VEC(ALL(),ALL(),ALL()), ALL(), qid_coarse 
        ) ; 
        auto const W = subview(
            data_out.face_staggered_fields_z, VEC(ALL(),ALL(),ALL()), ALL(), qid_coarse 
        ) ; 

        TeamThreadMDRange<Rank<4>, team_handle_t> range(team,n/2,n/2,n/2,nv) ;

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
                size_t i_f{g+2*i},j_f{g+2*j},k_f{g+2*k} ; 

                size_t i_c{g+i+off_x},j_c{g+j+off_y},k_c{g+k+off_z} ; 

                // we don't want to fill if we have fine data 
                bool fill_x{true}, fill_y{true}, fill_z{true} ; 

                if ( i_c == g and have_fine_data_x(0,iq) ) {
                    fill_x = false ; 
                }
                if ( j_c == g and have_fine_data_y(0,iq) ) {
                    fill_y = false ; 
                }
                if ( k_c == g and have_fine_data_z(0,iq) ) {
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
                if ( (i_f == n+g-2) and (not have_fine_data_x(1,iq)) ) {
                    fill_inside_face(
                        i_c+1,j_c,k_c,
                        i_f+2,j_f,k_f,ivar,
                        u,v,w,
                        U,V,W,
                        limiter,
                        true,false,false) ; 
                }
                if ( (j_f == n+g-2) and (not have_fine_data_y(1,iq)) ) {
                    fill_inside_face(
                        i_c,j_c+1,k_c,
                        i_f,j_f+2,k_f,ivar,
                        u,v,w,
                        U,V,W,
                        limiter,
                        false,true,false) ; 
                }
                if ( (k_f == n+g-2) and (not have_fine_data_z(1,iq)) ) {
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
                size_t i_f{g+2*i},j_f{g+2*j},k_f{g+2*k} ; 

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

                            int is = ii ? +1 : -1 ;
                            int js = jj ? +1 : -1 ; 
                            int ks = kk ? +1 : -1 ; 

                            Uxx += 0.125 * (is*js*v(i_f+ii  ,j_f+2*jj,k_f+kk,ivar) + is*ks*w(i_f+ii,j_f+jj  ,k_f+2*kk,ivar));
                            Vyy += 0.125 * (is*js*u(i_f+2*ii,j_f+jj  ,k_f+kk,ivar) + js*ks*w(i_f+ii,j_f+jj  ,k_f+2*kk,ivar));
                            Wzz += 0.125 * (is*ks*u(i_f+2*ii,j_f+jj  ,k_f+kk,ivar) + js*ks*v(i_f+ii,j_f+2*jj,k_f+kk  ,ivar));
                             
                            Uxyz += 0.125 * 0.5 *  (is*js*ks*u(i_f+2*ii,j_f+jj  ,k_f+kk  ,ivar)) ; 
                            Vxyz += 0.125 * 0.5 *  (is*js*ks*v(i_f+ii  ,j_f+2*jj,k_f+kk  ,ivar)) ; 
                            Wxyz += 0.125 * 0.5 *  (is*js*ks*w(i_f+ii  ,j_f+jj  ,k_f+2*kk,ivar)) ; 
                        }
                    }
                }
                 
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


template<typename view_t> 
gpu_task_t make_prolong(
    view_t& data_in, 
    view_t& data_out,
    std::vector<size_t> const& qin, 
    std::vector<size_t> const& qout,
    std::vector<size_t> const& varlist_lo,
    std::vector<size_t> const& varlist_ho,
    std::vector<double> const& ho_prolong_coeffs,
    grace::device_stream_t& stream,
    task_id_t& task_counter)
{
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS;

    using prolong_op_lo = slope_limited_prolong_op<grace::minmod> ; 
    using prolong_op_ho = lagrange_prolong_op<4> ; 

    GRACE_TRACE("Registering prolong task, tid {}, n elements {}", task_counter, qin.size());
    Kokkos::View<size_t*> qid_src("regrid_prolong_src_qid", qout.size()) 
                        , qid_dst("regrid_prolong_dst_qid", qin.size()) 
                        , varidx_lo("regrid_prolong_lo_vidx", varlist_lo.size())
                        , varidx_ho("regrid_prolong_ho_vidx", varlist_ho.size());
    deep_copy_vec_to_view(qid_src, qout) ; 
    deep_copy_vec_to_view(qid_dst, qin) ;
    deep_copy_vec_to_view(varidx_lo, varlist_lo) ; 
    deep_copy_vec_to_view(varidx_ho, varlist_ho) ; 

    Kokkos::View<double*> ho_prolong_coeffs_d("prolong_coefficients", ho_prolong_coeffs.size()) ; 
    deep_copy_vec_to_view(ho_prolong_coeffs_d,ho_prolong_coeffs) ; 

    gpu_task_t task {} ; 

    regrid_prolong_op<prolong_op_lo,view_t>
        functor_lo(data_in,data_out,qid_dst,qid_src,varidx_lo,prolong_op_lo{}, nx, ngz) ; 
    regrid_prolong_op<prolong_op_ho,view_t>
        functor_ho(data_in,data_out,qid_dst,qid_src,varidx_ho,prolong_op_ho{ho_prolong_coeffs_d}, nx, ngz) ; 

    auto exec_space = grace::make_exec_space(stream) ;
    // loop over fine quad_ids --> qid_in 
    MDRangePolicy<Rank<5,Iterate::Left>>
    policy_lo{
        exec_space, {0,0,0,0,0}, {nx,ny,nz,varlist_lo.size(),qin.size()}
    } ; 

    MDRangePolicy<Rank<5,Iterate::Left>>
    policy_ho{
        exec_space, {0,0,0,0,0}, {nx,ny,nz,varlist_ho.size(),qin.size()}
    } ;
    
    task._run = [functor_lo, functor_ho, policy_lo, policy_ho] (view_alias_t dummy) {
        parallel_for("regrid_prolong", policy_lo, functor_lo) ;
        parallel_for("regrid_prolong_high_order", policy_ho, functor_ho) ;
        #ifdef GRACE_DEBUG 
        Kokkos::fence() ; 
        #endif
    } ; 

    task.stream = &stream ; 
    task.task_id = task_counter++ ; 

    return task ; 
}
// NB data here is stag_var_arrays
template<typename view_t> 
gpu_task_t make_div_free_prolong(
    view_t& data_in, 
    view_t& data_out,
    std::vector<size_t> const& qin, 
    std::vector<size_t> const& qout,
    std::vector<std::array<int8_t,2>> const& have_fine_x,
    std::vector<std::array<int8_t,2>> const& have_fine_y,
    std::vector<std::array<int8_t,2>> const& have_fine_z,
    grace::device_stream_t& stream,
    size_t nvars,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list,
    std::unordered_set<task_id_t> const& dependencies
)
{
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS;
    GRACE_TRACE("Registering div-preserving prolong task, tid {}, n elements {}", task_counter, qin.size());
    Kokkos::View<size_t*> qid_src("regrid_prolong_src_qid", qout.size()) 
                        , qid_dst("regrid_prolong_dst_qid", qin.size()) ;
    deep_copy_vec_to_view(qid_src, qout) ; 
    deep_copy_vec_to_view(qid_dst, qin) ;
    // loop runs through incoming quads 
    Kokkos::View<int8_t**> have_fine_data_x("regrid_prolong_have_fine", 2, qin.size()) ; 
    auto have_fine_h_x = create_mirror_view(have_fine_data_x);
    Kokkos::View<int8_t**> have_fine_data_y("regrid_prolong_have_fine", 2, qin.size()) ; 
    auto have_fine_h_y = create_mirror_view(have_fine_data_y);
    Kokkos::View<int8_t**> have_fine_data_z("regrid_prolong_have_fine", 2, qin.size()) ; 
    auto have_fine_h_z = create_mirror_view(have_fine_data_z);
    for( int i=0; i<qin.size(); ++i) {
        for( int j=0; j<2; ++j) {
            have_fine_h_x(j,i) = have_fine_x[i][j];
            have_fine_h_y(j,i) = have_fine_y[i][j];
            have_fine_h_z(j,i) = have_fine_z[i][j];
        }
    }
    deep_copy(have_fine_data_x,have_fine_h_x) ;
    deep_copy(have_fine_data_y,have_fine_h_y) ;
    deep_copy(have_fine_data_z,have_fine_h_z) ;

    gpu_task_t task {} ; 

    regrid_div_free_prolong_op
        functor(data_in,data_out,qid_dst,qid_src,have_fine_data_x,have_fine_data_y,have_fine_data_z, nx, ngz,nvars) ; 

    auto exec_space = grace::make_exec_space(stream) ;
    // loop over fine quad_ids --> qid_in 
    TeamPolicy
    policy{
        exec_space, static_cast<int>(qin.size()), AUTO
    } ; 
    
    task._run = [functor, policy] (view_alias_t dummy) {
        parallel_for("regrid_div_free_prolong", policy, functor) ;
        #ifdef GRACE_DEBUG 
        Kokkos::fence() ; 
        #endif
    } ; 

    task.stream = &stream ; 
    task.task_id = task_counter++ ; 
    for( auto tid: dependencies ) {
        task._dependencies.push_back(tid) ; 
        task_list[tid]->_dependents.push_back(task.task_id) ; 
    }
    
    return task ;

} 
}
#endif 