/**
 * @file memory_defaults.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Default space for memory allocation / parallel dispatch in GRACE.
 * @version 0.1
 * @date 2023-06-16
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference 
 * methods to simulate relativistic astrophysical systems and plasma
 * dynamics.
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
#ifndef B8F66784_D4E4_4051_84B1_B8AFB588A053
#define B8F66784_D4E4_4051_84B1_B8AFB588A053

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <Kokkos_Core.hpp>

#include <vector> 
#include <array>

namespace grace { 
//*****************************************************************************************************
/**
 * @brief Typedef that defines the default Memory and Execution space(s) in GRACE.
 *        The hierarchy of precedence goes as follows:
 *        1) If a device is enabled, we use it
 *        2) If a host parallel mode is enabled, we use Host memory and 
 *           host-parallel execution
 *        3) If only host serial is available, that's what we use. (for debug only hopefully!)
 * See the general README on how to enable backends during GRACE's build process.
 * \ingroup variables
 */
//*****************************************************************************************************
#if defined(GRACE_ENABLE_CUDA)
using default_space = Kokkos::CudaSpace   ;  
#elif defined(GRACE_ENABLE_HIP)
using default_space = Kokkos::HIPSpace    ;
#elif defined(GRACE_ENABLE_SYCL)
#if KOKKOS_VERSION >= 40000
using default_space = Kokkos::SYCLDeviceUSMSpace;
#else
using default_space = Kokkos::Experimental::SYCLDeviceUSMSpace    ;
#endif
#elif defined(GRACE_ENABLE_OMP) or defined(GRACE_ENABLE_SERIAL)
using default_space = Kokkos::HostSpace   ;
#endif   
using default_execution_space = default_space::execution_space ;
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Deep copy a <code>std::vector<T></code> to a <code>Kokkos::View<T*></code> on device
 * \ingroup utils
 * @tparam ViewT Type of the View
 * @tparam T     Data type
 * @param view   View where the data will be copied
 * @param vec    Vector from which the data will be copied
 */
template< typename ViewT
        , typename T >
static void GRACE_ALWAYS_INLINE
deep_copy_vec_to_view(ViewT view, std::vector<T> const& vec)
{
    static_assert(std::is_same_v<T,typename ViewT::value_type>
                 , "In deep_copy_vec_to_view: data types mismatch.") ;
    static_assert( ViewT::rank() == 1
                 , "In deep_copy_vec_to_view: view must have rank 1.") ; 
    Kokkos::realloc(view, vec.size()) ; 
    auto h_view = Kokkos::create_mirror_view(view) ; 
    for(int i=0; i<vec.size(); ++i) h_view(i) = vec[i] ; 
    Kokkos::deep_copy(view,h_view) ; 
}
//*****************************************************************************************************
/**
 * @brief Deep copy a <code>std::vector<std::array<T,N>></code> to a <code>Kokkos::View<T*[N]></code> on device
 * \ingroup utils
 * @tparam N     Array size 
 * @tparam ViewT Type of the View
 * @tparam T     Data type
 * @param view   View where the data will be copied
 * @param vec    Vector from which the data will be copied
 */
template< size_t N
        , typename ViewT
        , typename T >
static void GRACE_ALWAYS_INLINE
deep_copy_vec_to_2D_view(ViewT view, std::vector<std::array<T,N>> const& vec)
{
    Kokkos::realloc(view, vec.size()) ; 
    auto h_view = Kokkos::create_mirror_view(view) ; 
    for(int i=0; i<vec.size(); ++i) {
        for( int j=0; j<N; ++j) h_view(i,j) = vec[i][j] ; 
    }
    Kokkos::deep_copy(view,h_view) ; 
}
//*****************************************************************************************************
/**
 * @brief Deep copy a <code>std::vector<T></code> to a <code>Kokkos::View<const T*></code> on device
 * \ingroup utils
 * @tparam ViewT Type of the const View
 * @tparam T     Data type
 * @param view   Const View where the data will be copied
 * @param vec    Vector from which the data will be copied
 */
template< typename ViewT
        , typename T >
static void GRACE_ALWAYS_INLINE
deep_copy_vec_to_const_view(ViewT& view, std::vector<T> const& vec)
{
    static_assert(std::is_same_v<T,std::remove_cv_t< typename ViewT::value_type> >
                 , "In deep_copy_vec_to_view: data types mismatch.") ;
    static_assert( ViewT::rank() == 1
                 , "In deep_copy_vec_to_view: view must have rank 1.") ; 
    Kokkos::View<T*> d_view("tmp", vec.size()) ; 
    auto h_view = Kokkos::create_mirror_view(d_view) ; 
    for(int i=0; i<vec.size(); ++i) h_view(i) = vec[i] ; 
    Kokkos::deep_copy(d_view,h_view) ; 
    view = ViewT(d_view) ;
}
//*****************************************************************************************************
/**
 * @brief Deep copy a <code>std::vector<std::array<T,N>></code> to a <code>Kokkos::View<const T*[N]></code> on device
 * \ingroup utils
 * @tparam ViewT Type of the const View
 * @tparam T     Data type
 * @param view   Const View where the data will be copied
 * @param vec    Vector from which the data will be copied
 */
template< size_t N
        , typename ViewT
        , typename T >
static void GRACE_ALWAYS_INLINE
deep_copy_vec_to_const_2D_view(ViewT& view, std::vector<std::array<T,N>> const& vec)
{
    static_assert(std::is_same_v<T,std::remove_cv_t< typename ViewT::value_type> >
                 , "In deep_copy_vec_to_view: data types mismatch.") ;
    Kokkos::View<T*[N]> d_view("tmp", vec.size()) ; 
    auto h_view = Kokkos::create_mirror_view(d_view) ; 
    for(int i=0; i<vec.size(); ++i) {
        for( int j=0; j<N; ++j) h_view(i,j) = vec[i][j] ;
    } 
    Kokkos::deep_copy(d_view,h_view) ; 
    view = ViewT(d_view) ;
}
//*****************************************************************************************************
  /**
 * @brief Deep copy a <code>std::vector<T></code> to a <code>Kokkos::View<T*></code> on device
 * \ingroup utils
 * @tparam ViewT Type of the View
 * @tparam T     Data type
 * @param vec   Vector where the data will be copied
 * @param view  View from which the data will be copied 
 */
template< typename ViewT
        , typename T >
static void GRACE_ALWAYS_INLINE
deep_copy_view_to_vec(std::vector<T> const& vec, ViewT view)
{
    static_assert(std::is_same_v<T,typename ViewT::value_type>
                 , "In deep_copy_vec_to_view: data types mismatch.") ;
    static_assert( ViewT::rank() == 1
                 , "In deep_copy_vec_to_view: view must have rank 1.") ;
    vec.resize(view.extent(0)) ; 
    auto h_view = Kokkos::create_mirror_view(view) ; 
    Kokkos::deep_copy(h_view,view) ; 
    for(int i=0; i<vec.size(); ++i) vec[i] = h_view(i) ; 
}
} /* namespace grace */

#endif /* B8F66784_D4E4_4051_84B1_B8AFB588A053 */
