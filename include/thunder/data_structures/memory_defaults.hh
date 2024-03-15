/**
 * @file memory_defaults.hh
 * @author your name (you@domain.com)
 * @brief Default space for memory allocation / parallel dispatch in Thunder.
 * @version 0.1
 * @date 2023-06-16
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference 
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

#include <thunder_config.h>

namespace thunder { 

#if defined(THUNDER_ENABLE_CUDA)
using DefaultSpace = Kokkos::CudaSpace   ; 
#elif defined(THUNDER_ENABLE_HIP)
using DefaultSpace = Kokkos::HIPSpace    ;
#elif defined(THUNDER_ENABLE_OMP) or defined(THUNDER_ENABLE_SERIAL)
using DefaultSpace = Kokkos::HostSpace   ;
#endif   

}

#endif /* B8F66784_D4E4_4051_84B1_B8AFB588A053 */
