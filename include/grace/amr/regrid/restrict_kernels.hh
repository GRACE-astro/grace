/**
 * @file restrict_kernels.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief This file contains the restriction kernel for regrid.
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

#ifndef GRACE_AMR_REGRID_RESTRICT_KERNEL_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace {

template< typename view_t >
struct regrid_restrict_op {
    view_t data_in, data_out ; 
    readonly_view_t<std::size_t> qid_in, qid_out ; 
    size_t n, g ; 

    regrid_restrict_op(
        view_t _data_in,
        view_t _data_out,
        Kokkos::View<size_t*> _qid_in,
        Kokkos::View<size_t*> _qid_out,
        size_t _n, size_t _g
    ) : data_in(_data_in)
      , data_out(_data_out)
      , qid_in(_qid_in)
      , qid_out(_qid_out)
      , n(_n), g(_g)
    {}

    // loop runs over coarse indices 
    KOKKOS_INLINE_FUNCTION 
    void operator() (size_t i, size_t j, size_t k, size_t ivar, size_t iq) const
    {
        // restrict: fine = out, coarse = in 

        // coarse quad_id
        auto qc = qid_in(iq) ;

        // child id 
        int8_t ic = (2*i>=n) + ((2*j>=n)<<1) + ((2*k>=n)<<2) ; 
        // fine quad_id 
        auto qf = qid_out(P4EST_CHILDREN * iq + ic ) ; 

        size_t i_f = 2*i%n ; 
        size_t j_f = 2*j%n ; 
        size_t k_f = 2*k%n ; 
        // sum the 8 fine cells inside the coarse cell
        double val{0} ; 
        for( int ii=0; ii<2; ++ii) for( int jj=0; jj<2; ++jj) for(int kk=0; kk<2;++kk) 
            val += data_out(g + i_f + ii,g + j_f + jj,g + k_f + kk, ivar, qf) ; 
        data_in(i+g,j+g,k+g,ivar,qc) = val * 0.125 ; 
    }
} ; 

template< int stag_dir, typename view_t >
struct regrid_div_preserving_restrict_op {
    view_t data_in, data_out ; 
    readonly_view_t<std::size_t> qid_in, qid_out ; 
    size_t n, g ; 

    regrid_div_preserving_restrict_op(
        view_t _data_in,
        view_t _data_out,
        Kokkos::View<size_t*> _qid_in,
        Kokkos::View<size_t*> _qid_out,
        size_t _n, size_t _g
    ) : data_in(_data_in)
      , data_out(_data_out)
      , qid_in(_qid_in)
      , qid_out(_qid_out)
      , n(_n), g(_g)
    {}

    // loop runs over coarse indices 
    KOKKOS_INLINE_FUNCTION 
    void operator() (size_t i, size_t j, size_t k, size_t ivar, size_t iq) const
    {
        // restrict: fine = out, coarse = in 
        // coarse quad_id
        auto qc = qid_in(iq) ;
        // child id 
        int8_t icx = (2*i>=n) ; 
        int8_t icy = (2*j>=n) ;
        int8_t icz = (2*k>=n) ;
        int8_t ic = icx + (icy<<1) + (icz<<2) ; 
        // fine quad_id 
        auto qf = qid_out(P4EST_CHILDREN * iq + ic ) ; 
        size_t i_f, j_f, k_f ; 
        if ( stag_dir == 0 ) {
            i_f = icx ? 2*i-n : 2*i ; 
            j_f = 2*j%n ; 
            k_f = 2*k%n ; 
        } else if ( stag_dir == 1 ) {
            j_f = icy ? 2*j-n : 2*j ; 
            i_f = 2*i%n ; 
            k_f = 2*k%n ; 
        } else {
            k_f = icz ? 2*k-n : 2*k ; 
            i_f = 2*i%n ; 
            j_f = 2*j%n ; 
        }
        //printf("stag %d qf %zu if %zu jf %zu kf %zu qc %zu ic %zu jc %zu kc %zu\n", stag_dir,qf,i_f+g,j_f+g,k_f+g,qc,i+g,j+g,k+g) ; 
        // sum the 4 fine faces inside the coarse face
        double val{0} ; 
        for( int ii=0; ii<=(stag_dir!=0); ++ii) {
            for( int jj=0; jj<=(stag_dir!=1); ++jj){
                for(int kk=0; kk<=(stag_dir!=2); ++kk){
                    val += data_out(g + i_f + ii,g + j_f + jj,g + k_f + kk, ivar, qf) ; 
                }
            }
        }

        //if ( i+g < data_in.extent(0) and j+g <  data_in.extent(1) and k+g <  data_in.extent(2) and i>=0 and j>=0 and k>=0 and qc < data_in.extent(4))
        data_in(i+g,j+g,k+g,ivar,qc) = val * 0.25 ; 
    }
} ; 

template< var_staggering_t stag, typename view_t> 
gpu_task_t make_div_free_restrict(
    view_t data_in, 
    view_t data_out,
    std::vector<size_t> const& qin, 
    std::vector<size_t> const& qout,
    grace::device_stream_t& stream,
    size_t nvars,
    task_id_t& task_counter)
{
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    constexpr int stag_dir = (stag==STAG_FACEX ? 0 : (stag == STAG_FACEY ? 1 : 2 )) ; 

    GRACE_TRACE("Registering restrict task, stag dir {} tid {}, n elements {}", stag_dir, task_counter, qin.size());

    Kokkos::View<size_t*> qid_src("regrid_copy_src_qid", qout.size()) 
                        , qid_dst("regrid_copy_dst_qid", qin.size()) ;
    deep_copy_vec_to_view(qid_src, qout) ; 
    deep_copy_vec_to_view(qid_dst, qin) ;

    gpu_task_t task {} ; 
    

    regrid_div_preserving_restrict_op<stag_dir,decltype(data_in)> 
        functor(data_in,data_out,qid_dst,qid_src,nx,ngz) ; 

    DefaultExecutionSpace exec_space{stream} ; 

    static_assert(std::is_same_v<typename decltype(functor.data_in)::memory_space,
                             typename DefaultExecutionSpace::memory_space>,
                "data_in memory_space must match execution_space");
    static_assert(std::is_same_v<typename decltype(functor.data_out)::memory_space,
                                typename DefaultExecutionSpace::memory_space>,
                "data_out memory_space must match execution_space");

    auto s = get_index_staggerings(stag) ;
    // loop over coarse quads 
    MDRangePolicy<Rank<5,Iterate::Left>>
    policy{
        exec_space, {0,0,0,0,0}, {nx+s[0],ny+s[1],nz+s[2],nvars,qin.size()}
        
    } ;
    
    GRACE_TRACE_DBG("Registering restrict, stag {}, s {} {} {}, range {} {} {}", static_cast<int>(stag), s[0],s[1],s[2], nx+s[0],ny+s[1],nz+s[2] ) ; 
    task._run = [functor, policy] (view_alias_t dummy) {
        GRACE_TRACE_DBG("Data in extents ({},{},{},{},{})",functor.data_in.extent(0),functor.data_in.extent(1),functor.data_in.extent(2),functor.data_in.extent(3),functor.data_in.extent(4));
        GRACE_TRACE_DBG("Data out extents ({},{},{},{},{})",functor.data_out.extent(0),functor.data_out.extent(1),functor.data_out.extent(2),functor.data_out.extent(3),functor.data_out.extent(4));

        parallel_for("regrid_div_free_restrict", policy, functor) ;
        #ifdef GRACE_DEBUG 
        Kokkos::fence() ; 
        #endif
    } ; 

    task.stream = &stream ; 
    task.task_id = task_counter++ ; 

    return task;
}

template<typename view_t> 
gpu_task_t make_restrict(
    view_t& data_in, 
    view_t& data_out,
    std::vector<size_t> const& qin, 
    std::vector<size_t> const& qout,
    grace::device_stream_t& stream,
    size_t nvars,
    task_id_t& task_counter)
{
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    GRACE_TRACE("Registering restrict task, tid {}, n elements {}", task_counter, qin.size());
    Kokkos::View<size_t*> qid_src("regrid_copy_src_qid", qout.size()) 
                        , qid_dst("regrid_copy_dst_qid", qin.size()) ;
    deep_copy_vec_to_view(qid_src, qout) ; 
    deep_copy_vec_to_view(qid_dst, qin) ;
    
    gpu_task_t task {} ; 
    
    regrid_restrict_op functor(data_in,data_out,qid_dst,qid_src,nx,ngz) ; 

    DefaultExecutionSpace exec_space{stream} ; 

    // loop over coarse quads 
    MDRangePolicy<Rank<5,Iterate::Left>>
    policy{
        exec_space, {0,0,0,0,0}, {nx,ny,nz,nvars,qin.size()}
    } ;

    task._run = [functor, policy] (view_alias_t dummy) {
        parallel_for("regrid_restrict", policy, functor) ;
        #ifdef GRACE_DEBUG 
        Kokkos::fence() ; 
        #endif
    } ; 

    task.stream = &stream ; 
    task.task_id = task_counter++ ; 

    return task ; 
}

} /* namespace grace */

#endif /*  #ifndef GRACE_AMR_REGRID_RESTRICT_KERNEL_HH */