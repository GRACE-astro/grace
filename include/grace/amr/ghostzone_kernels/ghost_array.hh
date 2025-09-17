/**
 * @file ghost_array.hh
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
#ifndef GRACE_AMR_GHOST_ARRAY_HH
#define GRACE_AMR_GHOST_ARRAY_HH

#include <grace_config.h>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <vector>

namespace grace { namespace amr {


struct ghost_array_t 
{

    ghost_array_t(std::string const& name)
        :  _data(name,0), _size(0)
    {}

    void set_strides(size_t const& nx, size_t const& ny, size_t const& nz, 
                     size_t const& nv, size_t const& ngz)
    {
        fstrides = std::array<size_t,4> {ngz,ngz*nx,ngz*nx*nx,ngz*nx*nx*nv};
        estrides = std::array<size_t,4> {ngz,ngz*ngz,ngz*ngz*nx,ngz*ngz*nx*nv};
        cstrides = std::array<size_t,4> {ngz,ngz*ngz,ngz*ngz*ngz,ngz*ngz*ngz*nv};
    }

    void set_offsets(
        std::vector<size_t> const& _roffsets, 
        std::array<std::vector<size_t>,6> const & _offsets 
    ) 
    {
        grace::deep_copy_vec_to_const_view(rank_offsets, _roffsets) ; 
        grace::deep_copy_vec_to_const_view(edge_offsets, _offsets[1]) ; 
        grace::deep_copy_vec_to_const_view(corner_offsets, _offsets[2]) ; 
        grace::deep_copy_vec_to_const_view(cbuf_face_offsets, _offsets[3]) ; 
        grace::deep_copy_vec_to_const_view(cbuf_edge_offsets, _offsets[4]) ; 
        grace::deep_copy_vec_to_const_view(cbuf_corner_offsets, _offsets[5]) ; 
        #if 0
        // TODO remove 
        auto h_ro = Kokkos::create_mirror_view(rank_offsets) ; 
        Kokkos::deep_copy(h_ro,rank_offsets) ; 
        for( int i=0; i<_roffsets.size(); ++i) {
            GRACE_TRACE("Offset {}", h_ro(i)) ; 
        }
        #endif 
    }

    void realloc(size_t const& _new_size) {
        _size = _new_size ; 
        Kokkos::realloc(_data, _new_size) ; 
    }

    GRACE_HOST GRACE_ALWAYS_INLINE 
    size_t size() const { return _size ; }

    GRACE_HOST GRACE_ALWAYS_INLINE 
    double* data() const { return _data.data() ; }

    template< element_kind_t elem_kind > 
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double& at_interface(size_t const& i, size_t const& j, size_t const& k, 
                         size_t const& iv, size_t const& ie, size_t const& rank) const 
    {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            return at_faces(i,j,k,iv,ie,rank) ; 
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            return at_edges(i,j,k,iv,ie,rank) ; 
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            return at_corners(i,j,k,iv,ie,rank) ; 
        }
    }

    template< element_kind_t elem_kind > 
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double& at_cbuf(size_t const& i, size_t const& j, size_t const& k, 
                    size_t const& iv, size_t const& ie, size_t const& rank) const 
    {
        auto offset = rank_offsets(rank) ; 
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            offset += cbuf_face_offsets(rank) ; 
            return get(i,j,k,iv,ie,fstrides[0],fstrides[1]/2,fstrides[2]/4,fstrides[3]/4,offset) ; 
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            offset += cbuf_edge_offsets(rank) ; 
            return get(i,j,k,iv,ie,estrides[0],estrides[1],estrides[2]/2,estrides[3]/2,offset) ;
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            offset += cbuf_corner_offsets(rank) ; 
            return get(i,j,k,iv,ie,cstrides[0],cstrides[1],cstrides[2],cstrides[3],offset) ;
        }
    }


private: 
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    double& get(
        size_t const& i, size_t const& j, size_t const& k, size_t const& iv, size_t const& ie, 
        size_t const& s0, size_t const& s1, size_t const& s2, size_t const& s3, 
        size_t const& offset) const 
    {
        auto c_index = i 
                     + s0 * j 
                     + s1 * k 
                     + s2 * iv 
                     + s3 * ie ;
        return _data(offset+c_index); 
    }
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double& at_faces(size_t const& ii, size_t const& jj, size_t const& kk, 
                     size_t const& iv, size_t const& ie, size_t const& rank) const 
    {
        auto const offset = rank_offsets(rank) ; 
        return get(ii,jj,kk,iv,ie,fstrides[0],fstrides[1],fstrides[2],fstrides[3],offset) ; 
    }

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double& at_edges(size_t const& ii, size_t const& jj, size_t const& kk, 
                     size_t const& iv, size_t const& ie, size_t const& rank) const 
    {
        auto offset = rank_offsets(rank) + edge_offsets(rank); 
        return get(ii,jj,kk,iv,ie,estrides[0],estrides[1],estrides[2],estrides[3],offset) ; 
    }

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double& at_corners(size_t const& ii, size_t const& jj, size_t const& kk, 
                      size_t const& iv, size_t const& ie, size_t const& rank) const 
    {
        auto offset = rank_offsets(rank) + corner_offsets(rank) ; 
        return get(ii,jj,kk,iv,ie,cstrides[0],cstrides[1],cstrides[2],cstrides[3],offset) ; 
    }
    readonly_view_t<std::size_t> rank_offsets, edge_offsets, corner_offsets, cbuf_face_offsets, cbuf_edge_offsets, cbuf_corner_offsets ; 
    std::array<size_t, 4> fstrides, estrides, cstrides ;
    Kokkos::View<double *, grace::default_space> _data ; 
    std::size_t _size ; 
} ; 

}} /* namespace grace::amr */

#endif /* GRACE_AMR_GHOST_ARRAY_HH */