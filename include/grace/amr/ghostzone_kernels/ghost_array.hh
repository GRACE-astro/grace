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

    ghost_array_t() = default ; 

    ghost_array_t(std::string const& name)
        :  _data(name,0), _size(0)
    {}

    void set_strides(std::array<size_t,4> const& strides )
    {
        auto n = strides[0] ; auto nv = strides[1]; auto ngz = strides[2]; auto n2 = strides[3] ;
        fstrides = std::array<size_t,4> {ngz,ngz*n,ngz*n*n,ngz*n*n*nv};
        estrides = std::array<size_t,4> {ngz,ngz*ngz,ngz*ngz*n,ngz*ngz*n*nv};
        cstrides = std::array<size_t,4> {ngz,ngz*ngz,ngz*ngz*ngz,ngz*ngz*ngz*nv};
        // cbufs 
        cfstrides = std::array<size_t,4> {ngz,ngz*n2,ngz*n2*n2,ngz*n2*n2*nv};
        cestrides = std::array<size_t,4> {ngz,ngz*ngz,ngz*ngz*n2,ngz*ngz*n2*nv};
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
            return get(i,j,k,iv,ie,cfstrides[0],cfstrides[1],cfstrides[2],cfstrides[3],offset) ; 
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            offset += cbuf_edge_offsets(rank) ; 
            return get(i,j,k,iv,ie,cestrides[0],cestrides[1],cestrides[2],cestrides[3],offset) ;
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
    std::array<size_t, 4> fstrides, estrides, cstrides, cfstrides, cestrides ;
    Kokkos::View<double *, grace::default_space> _data ; 
    std::size_t _size ; 
} ; 

//! For refluxing or emf exchanges
struct reflux_array_t {
    //! Ctor
    reflux_array_t() = default ; 
    //! Dtor
    reflux_array_t(std::string const& name)
        :  _data(name,0), _size(0)
    {}

    //! Set strides
    void set_strides(size_t ncells, size_t nvars )
    {
        fstrides = std::array<size_t,3> {ncells,ncells*ncells,ncells*ncells*nvars};
    }

    //! Set offsets
    void set_offsets(
        std::vector<size_t> const& _roffsets, 
    ) 
    {
        grace::deep_copy_vec_to_const_view(rank_offsets, _roffsets) ; 
    }

    //! Reallocate data
    void realloc(size_t const& _new_size) {
        _size = _new_size ; 
        Kokkos::realloc(_data, _new_size) ; 
    }

    //! Get size
    GRACE_HOST GRACE_ALWAYS_INLINE 
    size_t size() const { return _size ; }

    //! Get data pointer
    GRACE_HOST GRACE_ALWAYS_INLINE 
    double* data() const { return _data.data() ; }


    template< element_kind_t elem_kind > 
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double& operator() (size_t const& i, size_t const& j, size_t const& iv, size_t const& ie, size_t const& rank) const 
    {
        auto offset = rank_offsets(rank) ;
        return get(i,j,iv,ie,fstrides[0],fstrides[1],fstrides[2],offset) ;  
    }

    private:
    //! Accessor 
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    double& get(
        size_t const& i, size_t const& j, size_t const& iv, size_t const& ie, 
        size_t const& s0, size_t const& s1, size_t const& s2,
        size_t const& offset) const 
    {
        auto c_index = i 
                     + s0 * j 
                     + s1 * iv 
                     + s2 * ie ;
        return _data(offset+c_index); 
    }

    readonly_view_t<std::size_t> rank_offsets ; 
    std::array<size_t, 3> fstrides ; 
    Kokkos::View<double *, grace::default_space> _data ; //! Data
    std::size_t _size ;

} ; 

struct face_buffer_t {

    face_buffer_t() = default ; 

    face_buffer_t(std::string const& name)
        :  _data(name,0), _size(0)
    {}

    void set_strides(std::array<size_t,2> const& _strides )
    {
        auto n = _strides[0] ; auto nv = _strides[1]; 
        strides = std::array<size_t,3> {n,n*n,nv*n*n};
    }

    void set_offsets(
        std::vector<int> const& _roffsets
    ) 
    {
        grace::deep_copy_vec_to_const_view(rank_offsets, _roffsets) ; 
    }

    void realloc(size_t const& _new_size) {
        _size = _new_size ; 
        Kokkos::realloc(_data, _new_size) ; 
    }

    GRACE_HOST GRACE_ALWAYS_INLINE 
    size_t size() const { return _size ; }

    GRACE_HOST GRACE_ALWAYS_INLINE 
    double* data() const { return _data.data() ; }

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    double& operator() (size_t const& i, size_t const& j, size_t const& iv, size_t const& ie, size_t const& rank) const
    {
        return get(i,j,iv,ie,strides[0],strides[1],strides[2],rank_offsets(rank)) ; 
    }

    private:

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    double& get(
        size_t const& i, size_t const& j, size_t const& iv, size_t const& ie, 
        size_t const& s0, size_t const& s1, size_t const& s2, 
        size_t const& offset) const 
    {
        auto c_index = i 
                     + s0 * j
                     + s1 * iv 
                     + s2 * ie ;
        return _data(offset+c_index); 
    }

    
    readonly_view_t<int> rank_offsets;
    std::array<size_t, 3> strides;
    Kokkos::View<double *, grace::default_space> _data ;
    std::size_t _size ;     
} ; 

}} /* namespace grace::amr */

#endif /* GRACE_AMR_GHOST_ARRAY_HH */