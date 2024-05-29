/**
 * @file memory_defaults.hh
 * @author your name (you@domain.com)
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

#include <Kokkos_Core.hpp>

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
} /* namespace grace */

#endif /* B8F66784_D4E4_4051_84B1_B8AFB588A053 */
