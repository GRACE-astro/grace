/**
 * @file device_vector.hh
 * @author  Carlo Musolino
 * @brief Definition of the device_vector container.
 * @date 2024-09-07
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

#ifndef GRACE_UTILS_DEVICE_VECTOR_HH
#define GRACE_UTILS_DEVICE_VECTOR_HH

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/iterator.hh>
#include <grace/utils/type_name.hh> 
#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <vector> 
#include <algorithm>

namespace grace {

/**
 * @brief Device vector class.
 * This container is designed to aid performing simple tasks where data 
 * needs to be exchanged frequently between host and device. This container 
 * has a host-side allocation in the form of a <code>std::vector<T></code> and 
 * a device side allocation in the form of a <code>Kokkos::View<T*></code>. Note
 * that the device side container is always one-dimensional. This class provides 
 * an easy way to mirror a 1-d vector on device, and provides all the common 
 * interface of <code>std::vector</code> on host side. 
 * The main purpose of this class is to have a variable-size allocation on host 
 * where elements can be appended and a simple way to mirror it on device. Since 
 * frequent allocations on device can lead to segmentation and severe performance 
 * degradation, the sizes of the two allocations are never guaranteed to match except 
 * right after the constructor is called or after the two containers are manually 
 * sync'ed via a call to <code>to_host()</code> or <code>to_device()</code>
 * 
 * \ingroup utils
 * 
 * @tparam T Data type held by the containers.
 */
template< typename T >
class device_vector {
 
 private:

    host_container_type _host_array ;
    device_container_type _device_array ;

    using size_type       = typename host_container_type::size_type ; //!< Type of index 
    using value_type      = typename host_container_type::value_type  ; //!< Type of values 
    using pointer         = typename host_container_type::pointer  ; //!< Type of pointer to values 
    using reference       = typename host_container_type::reference ; //!< Type of reference to values 
    using iterator        = typename host_container_type::iterator       ; //!< Iterator type 
    using const_iterator  = typename host_container_type::const_iterator ; //!< Const iterator type 
    using host_container_type = std::vector<T> ;
    using device_container_type = Kokkos::View<T*> ; 

     

 public: 
    /**
     * @brief Construct a new empty device_vector object.
     * 
     */
    device_vector() = default ;
    /**
     * @brief Construct a new empty device_vector 
     * object of a specified size.
     * 
     * @param N Size of the initial allocations.
     */
    device_vector( size_t N ) 
     : _host_array(N), _device_array()
    {
        Kokkos::realloc( _device_array, N ) ; 
    } 

    device_vector( size_t N, T const& val ) 
     : _host_array(N,val), _device_array()
    {
        Kokkos::realloc( _device_array, N ) ;
        grace::deep_copy_vec_to_view(_device_array, _host_array) ;  
    }   
    /**
     * @brief Construct a new device_vector object from a vector
     * 
     * @param rhs The vector used to initialize.
     */
    device_vector( host_container_type const& rhs )
     : _host_container(rhs)
    {
        to_device() ;
    }
    /**
     * @brief Move constructor from a <code>std::vector<T></code>
     * 
     * @param rhs The vector used to initialize.
     */
    device_vector( host_container_type && rhs )
     : _host_container(rhs)
    {
        to_device() ; 
    }
    /**
     * @brief Access to an element of the host-side vector.
     * Checks for bounds in debug mode.
     * @param idx Index where the element needs to be accessed.
     * @return value_type Element at the requested index.
     */
    reference GRACE_ALWAYS_INLINE GRACE_HOST
    operator[] (size_type const& idx) {
        ASSERT_DBG( idx < size()
                  , "Out-of-bounds access on host side of "
                    "device_array.") ; 
        return _host_array[idx] ; 
    }
    /**
     * @brief Const-access to an element of the host-side vector.
     * Checks for bounds in debug mode.
     * @param idx Index where the element needs to be accessed.
     * @return value_type Element at the requested index.
     */
    value_type GRACE_ALWAYS_INLINE GRACE_HOST
    operator[] (size_type const& idx) const {
        ASSERT_DBG( idx < size()
                  , "Out-of-bounds access on host side of "
                    "device_array.") ; 
        return _host_array[idx] ; 
    }

    iterator begin() {
        return _host_array.begin() ; 
    }

    iterator end() {
        return _host_array.end() ; 
    }

    const_iterator cbegin() const {
        return _host_array.cbegin() ; 
    }

    const_iterator cend() const {
        return _host_array.cend() ; 
    }
    /**
     * @brief Append an element at the end of the host vector.
     * NB: This does \b not sync the data to device, and this is by design.
     * When the vector needs to be used on device side the sync has to be done 
     * manually by calling <code>to_device()</code>
     * @param val Value to push_back.
     */
    void GRACE_HOST
    push_back( value_type const& val) {
        _host_array.push_back(val) ; 
    }
    /**
     * @brief Get raw pointer to host vector.
     * 
     * @return pointer Pointer to host vector memory.
     */
    pointer GRACE_ALWAYS_INLINE GRACE_HOST
    data() { return _host_array.data(); }
    /**
     * @brief Get the device-side <code>Kokkos::View<T*></code>.
     * 
     * @return device_container_type The device-side View.
     */
    device_container_type GRACE_ALWAYS_INLINE GRACE_HOST
    d_view() {
        return _device_array ; 
    }
    /**
     * @brief Get the size of the host vector.
     * NB: This is \b not guaranteed to coincide with 
     * the size of the device view. To find that, use 
     * <code>device_size()</code>.
     * @return index_type The size of the host-side vector.
     */
    size_type GRACE_ALWAYS_INLINE GRACE_HOST
    size()  
    {
        return _host_array.size()  ;
    }
    /**
     * @brief Size of device-side allocation.
     * NB: This is \b not guaranteed to coincide with
     * host-side size.
     * @return index_type Size of device-side View.
     */
    size_type GRACE_ALWAYS_INLINE GRACE_HOST
    device_size()  
    {
        return _device_array.extent(0)  ;
    }
    /**
     * @brief Sync data to device.
     */
    void GRACE_HOST
    to_device()
    {
        if ( ! (size() == device_size()) ) {
            Kokkos::realloc(_device_array, size()) ; 
        }
        grace::deep_copy_vec_to_view(_device_array,_host_array) ; 
    }
    /**
     * @brief Sync data to host.
     */
    void GRACE_HOST
    to_host() 
    {
        if ( ! (size() == device_size()) ) {
            _host_array.resize(device_size()) ; 
        }
        grace::deep_copy_view_to_vec(_host_array,_device_array) ; 
    }

} ; 


}

#endif /* GRACE_UTILS_DEVICE_VECTOR_HH */