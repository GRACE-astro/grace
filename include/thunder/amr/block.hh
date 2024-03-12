/**
 * @file block.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief This file contains the definition of the block class 
 * @version 0.1
 * @date 2023-06-14
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
#ifndef CC156A94_9EBD_4F68_9A1B_6D3009738AF7
#define CC156A94_9EBD_4F68_9A1B_6D3009738AF7

#include <thunder/thunder_config.h> 

#include <Kokkos_Core.hpp> 

namespace thunder { namespace amr {
/**
 * @brief A mesh block is an ensemble of elements sitting on
 *        a shared-memory computing unit. This typically means 
 *        an MPI rank which may comprise multiple CPUs/GPUs.
 * 
 * @tparam THUNDER_NSPACEDIM 
 * @tparam coord_policy 
 * 
 * A mesh block holds coordinates of all the collocation points within 
 * the elements belonging to the block. It also holds the determinant of 
 * the coordinate mapping jacobian and its inverse at the same points. 
 */
template< size_t ndim = THUNDER_NSPACEDIM
        , coordinate_system coord_policy = THUNDER_COORDINATE_SYSTEM > 
struct block 
{
 public: 
    THUNDER_ALWAYS_INLINE  block() ; 
    THUNDER_ALWAYS_INLINE ~block() ; 

    Kokkos::View< double** > x
                           , y 
                           #ifdef THUNDER_3D 
                           , z 
                           #endif 
                           ; //!< Physical coordinates 

    Kokkos::View< double**** > dJ, dJi ; //!< Jacobian and inverse determinant 

} ; 

}} // namespace thunder/amr

#endif /* CC156A94_9EBD_4F68_9A1B_6D3009738AF7 */
