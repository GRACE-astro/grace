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

#ifndef GRACE_AMR_REGRID_CPY_KERNEL_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <Kokkos_Core.hpp>

namespace grace {

template< typename view_t >
struct regrid_copy_op {
    view_t data_in, data_out ; 
    readonly_view_t<std::size_t> qid_in, qid_out ; 

    regrid_copy_op(
        view_t _data_in,
        view_t _data_out,
        Kokkos::View<size_t*> _qid_in,
        Kokkos::View<size_t*> _qid_out
    ) : data_in(_data_in)
      , data_out(_data_out)
      , qid_in(_qid_in)
      , qid_out(_qid_out)
    {}


    KOKKOS_INLINE_FUNCTION 
    void operator() (size_t i, size_t j, size_t k, size_t ivar, size_t iq) const
    {
        auto qi = qid_in(iq) ; 
        auto qo = qid_out(iq) ; 
        data_in(i,j,k,ivar,qi) = data_out(i,j,k,ivar,qo) ; 
    }
} ; 
template< typename view_t >
struct regrid_copy_fine_data_op {
    view_t data_in, data_out ; 
    readonly_view_t<grace::amr::fine_interface_desc_t> info ; 

    size_t n, g; 

    regrid_copy_fine_data_op(
        view_t _data_in,
        view_t _data_out,
        Kokkos::View<grace::amr::fine_interface_desc_t*> _info,
        size_t _n, size_t _g
    ) : data_in(_data_in)
      , data_out(_data_out)
      , info(_info)
      , n(_n), g(_g)
    {}

    KOKKOS_INLINE_FUNCTION 
    void operator() (size_t j, size_t k, size_t ivar, size_t iq) const
    {
        auto& desc = info(iq) ; 
        auto q_dst = desc.qid_dst;
        auto q_src = desc.qid_src ;

        auto iface_dst = desc.fid_dst ; 
        auto iface_src = desc.fid_src ; 

        int8_t axis = iface_dst / 2; 
        
        size_t i_a,j_a,k_a ; 
        size_t i_b,j_b,k_b ;
        int8_t sidei = iface_dst % 2;
        int8_t sideo = iface_src % 2;
        if ( axis == 0 ) {
            j_a = j_b = g + j ; 
            k_a = k_b = g + k ; 
            i_a = sidei ? n + g : g ; 
            i_b = sideo ? n + g : g ; 
            data_in(i_a,j_a,k_a,ivar,q_dst) 
                = data_out(i_b,j_b,k_b,ivar,q_src) ;
        } else if ( axis == 1 ) {
            i_a = i_b = g + j ; 
            k_a = k_b = g + k ; 
            j_a = sidei ? n + g : g ; 
            j_b = sideo ? n + g : g ; 
            data_in(i_a,j_a,k_a,ivar,q_dst) 
                = data_out(i_b,j_b,k_b,ivar,q_src) ;
        } else if ( axis == 2 ) {
            i_a = i_b = g + j ; 
            j_a = j_b = g + k ; 
            k_a = sidei ? n + g : g ; 
            k_b = sideo ? n + g : g ; 
            data_in(i_a,j_a,k_a,ivar,q_dst) 
                = data_out(i_b,j_b,k_b,ivar,q_src) ;
        }
    }
} ; 

template< typename view_t >
struct regrid_pack_fine_data_op {
    view_t data ; 
    amr::face_buffer_t buffer ; 
    readonly_view_t<grace::amr::fine_interface_desc_t> info ; 

    size_t n, g, rank ; 

    regrid_pack_fine_data_op(
        view_t _data,
        amr::face_buffer_t _buffer,
        Kokkos::View<grace::amr::fine_interface_desc_t*> _info,
        size_t _n, size_t _g, size_t _rank
    ) : data(_data)
      , buffer(_buffer)
      , info(_info)
      , n(_n), g(_g), rank(_rank)
    {}

    KOKKOS_INLINE_FUNCTION 
    void operator() (size_t j, size_t k, size_t ivar, size_t iq) const
    {
        auto qid_src = info(iq).qid_src ;
        auto qid_dst = info(iq).qid_dst ; 
        auto _fid = info(iq).fid_src ; 
        int8_t axis = _fid / 2 ; 
        int8_t side = _fid % 2 ; 
        size_t i_a,j_a,k_a ; 
        if ( axis == 0 ) {
            j_a = g + j ; 
            k_a = g + k ; 
            i_a = side ? n + g : g ; 
        } else if ( axis == 1 ) {
            i_a = g + j ; 
            k_a = g + k ; 
            j_a = side ? n + g : g ; 
        } else if ( axis == 2 ) {
            i_a = g + j ; 
            j_a = g + k ; 
            k_a = side ? n + g : g ; 
        }

        buffer(j,k,ivar,qid_dst,rank) = 
            data(i_a,j_a,k_a,ivar,qid_src) ; 
    }
} ; 


template< typename view_t >
struct regrid_unpack_fine_data_op {
    view_t data ; 
    amr::face_buffer_t buffer ; 
    readonly_view_t<grace::amr::fine_interface_desc_t> info ; 

    size_t n, g, rank ; 

    regrid_unpack_fine_data_op(
        view_t _data,
        amr::face_buffer_t _buffer,
        Kokkos::View<grace::amr::fine_interface_desc_t*> _info,
        size_t _n, size_t _g, size_t _rank
    ) : data(_data)
      , buffer(_buffer)
      , info(_info)
      , n(_n), g(_g), rank(_rank)
    {}

    KOKKOS_INLINE_FUNCTION 
    void operator() (size_t j, size_t k, size_t ivar, size_t iq) const
    {
        auto qid_dst = info(iq).qid_dst ; 
        auto _fid = info(iq).fid_dst ;
        auto qid_src = info(iq).qid_src  ; 
        int8_t axis = _fid / 2 ; 
        int8_t side = _fid % 2 ; 
        size_t i_a,j_a,k_a ; 
        if ( axis == 0 ) {
            j_a = g + j ; 
            k_a = g + k ; 
            i_a = side ? n + g : g ; 
        } else if ( axis == 1 ) {
            i_a = g + j ; 
            k_a = g + k ; 
            j_a = side ? n + g : g ; 
        } else if ( axis == 2 ) {
            i_a = g + j ; 
            j_a = g + k ; 
            k_a = side ? n + g : g ; 
        }

        data(i_a,j_a,k_a,ivar,qid_dst) = buffer(j,k,ivar,qid_src,rank); 
    }
} ; 

template< var_staggering_t stag, typename view_t> 
gpu_task_t make_copy(
    view_t& data_in, 
    view_t& data_out,
    std::vector<size_t> const& qin, 
    std::vector<size_t> const& qout,
    grace::device_stream_t& stream,
    size_t nvars,
    task_id_t& task_counter)
{
    DECLARE_GRID_EXTENTS ;
    using namespace Kokkos ; 
    GRACE_TRACE("Registering regrid copy task, stag {} tid {}, n elements {}", static_cast<int>(stag), task_counter, qin.size());
    Kokkos::View<size_t*> qid_src("regrid_copy_src_qid", qout.size()) 
                        , qid_dst("regrid_copy_dst_qid", qin.size()) ;
    deep_copy_vec_to_view(qid_src, qout) ; 
    deep_copy_vec_to_view(qid_dst, qin) ;
    
    gpu_task_t task {} ; 

    regrid_copy_op functor(data_in,data_out,qid_dst,qid_src) ; 

    auto exec_space = grace::make_exec_space(stream) ;

    auto s = get_index_staggerings(stag) ;

    MDRangePolicy<Rank<5,Iterate::Left>>
    policy{
        exec_space, 
        { static_cast<long>(0),
          static_cast<long>(0),
          static_cast<long>(0),
          0,
          0
        }, 
        { static_cast<long>(nx+s[0]+2*ngz),
          static_cast<long>(ny+s[1]+2*ngz),
          static_cast<long>(nz+s[2]+2*ngz),
          static_cast<long>(nvars),
          static_cast<long>(qin.size())
        }
    } ; 
    
    task._run = [functor, policy] (view_alias_t dummy) {
        parallel_for("regrid_copy", policy, functor) ;
        #ifdef GRACE_DEBUG 
        Kokkos::fence() ; 
        GRACE_TRACE("Copy done") ; 
        #endif
    } ; 

    task.stream = &stream ; 
    task.task_id = task_counter++ ; 

    return task ; 

}


template< typename view_t> 
gpu_task_t make_copy_face(
    view_t& data_in, 
    view_t& data_out,
    std::vector<grace::amr::fine_interface_desc_t> const& info, 
    grace::device_stream_t& stream,
    size_t nvars,
    task_id_t& task_counter)
{
    using namespace Kokkos; 
    DECLARE_GRID_EXTENTS;
    GRACE_TRACE("Registering regrid face copy task, tid {}, n elements {}", task_counter, info.size());
    Kokkos::View<grace::amr::fine_interface_desc_t*> info_d("regrid_copy_fine_info", info.size()) ;
    deep_copy_vec_to_view(info_d, info) ; 
    
    gpu_task_t task {} ; 

    regrid_copy_fine_data_op functor(data_in,data_out,info_d,nx,ngz) ; 

    auto exec_space = grace::make_exec_space(stream) ;

    MDRangePolicy<Rank<4,Iterate::Left>>
    policy{
        exec_space, {0,0,0,0}, {nx,nx,nvars,info.size()}
    } ; 
    
    task._run = [functor, policy] (view_alias_t dummy) {
        parallel_for("regrid_copy", policy, functor) ;
        #ifdef GRACE_DEBUG
        Kokkos::fence() ;
        GRACE_TRACE("Face copy done") ;
        #endif
    } ;

    task.stream = &stream ;
    task.task_id = task_counter++ ;

    return task ;

}


template<typename view_t>
gpu_task_t make_pack_face(
    view_t& data, 
    amr::face_buffer_t& buffer,
    std::vector<grace::amr::fine_interface_desc_t> const& info, 
    grace::device_stream_t& stream,
    size_t nvars,
    size_t rank,
    task_id_t send_tid, 
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list
)
{
    using namespace Kokkos; 
    DECLARE_GRID_EXTENTS;

    GRACE_TRACE("Registering regrid face pack task, tid {}, n elements {}", task_counter, info.size());

    Kokkos::View<grace::amr::fine_interface_desc_t*> info_d("regrid_pack_fine_info", info.size()) ;
    deep_copy_vec_to_view(info_d, info) ; 
    
    gpu_task_t task {} ; 

    regrid_pack_fine_data_op functor(data,buffer,info_d,nx,ngz,rank) ; 

    auto exec_space = grace::make_exec_space(stream) ;

    MDRangePolicy<Rank<4,Iterate::Left>>
    policy{
        exec_space, {0,0,0,0}, {nx,nx,nvars,info.size()}
    } ; 
    
    task._run = [functor, policy] (view_alias_t dummy) {
        parallel_for("regrid_copy", policy, functor) ;
        #ifdef GRACE_DEBUG
        Kokkos::fence() ;
        #endif
    } ;

    task.stream = &stream ;
    task.task_id = task_counter++ ;

    task_list[send_tid]->_dependencies.push_back(task.task_id) ;
    task._dependents.push_back(send_tid) ;

    return task ;

}

template<typename view_t>
gpu_task_t make_unpack_face(
    view_t& data, 
    amr::face_buffer_t& buffer,
    std::vector<grace::amr::fine_interface_desc_t> const& info, 
    grace::device_stream_t& stream,
    size_t nvars,
    size_t rank,
    task_id_t recv_tid,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list)
{
    using namespace Kokkos; 
    DECLARE_GRID_EXTENTS;

    GRACE_TRACE("Registering regrid face unpack task, tid {}, n elements {}", task_counter, info.size());

    Kokkos::View<grace::amr::fine_interface_desc_t*> info_d("regrid_pack_fine_info", info.size()) ;
    deep_copy_vec_to_view(info_d, info) ; 
    
    gpu_task_t task {} ; 

    regrid_unpack_fine_data_op functor(data,buffer,info_d,nx,ngz,rank) ; 

    auto exec_space = grace::make_exec_space(stream) ;

    MDRangePolicy<Rank<4,Iterate::Left>>
    policy{
        exec_space, {0,0,0,0}, {nx,nx,nvars,info.size()}
    } ; 
    
    task._run = [functor, policy] (view_alias_t dummy) {
        parallel_for("regrid_copy", policy, functor) ;
        #ifdef GRACE_DEBUG
        Kokkos::fence() ;
        #endif
    } ;

    task.stream = &stream ;
    task.task_id = task_counter++ ;

    task_list[recv_tid]->_dependents.push_back(task.task_id) ;
    task._dependencies.push_back(recv_tid) ;

    return task ;

}
} /* namespace grace */
#endif /*  #ifndef GRACE_AMR_REGRID_CPY_KERNEL_HH */