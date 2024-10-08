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
#include <grace/utils/device/device.h>

#include <Kokkos_Core.hpp>

#include <vector> 

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
static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
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

} /* namespace grace */

#endif /* B8F66784_D4E4_4051_84B1_B8AFB588A053 */
