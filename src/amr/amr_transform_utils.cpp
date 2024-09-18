/**
 * @file amr_transform_utilities.cpp
 * @author  ()
 * @brief 
 * @date 2024-09-04
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

#include <grace_config.h>

#include <grace/amr/amr_transform_utils.hh>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/amr/p4est_headers.hh>

#include <array> 

namespace grace {

std::array<std::array<int8_t,P4EST_HALF>, P4EST_FACES> 
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
face_corners() {
    #ifdef GRACE_3D 
    return {
        { 0, 2, 4, 6 },
        { 1, 3, 5, 7 },
        { 0, 1, 4, 5 },
        { 2, 3, 6, 7 },
        { 0, 1, 2, 3 },
        { 4, 5, 6, 7 }
    } ; 
    #else 
    return {
        { 0, 2 },
        { 1, 3 },
        { 0, 1 },
        { 2, 3 } 
    };
    #endif 
}

int8_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
face_corner(int8_t iface, int8_t icorner) {
    auto f2c = face_corners() ; 
    return f2c[iface][icorner] ; 
}

std::array<int8_t, P4EST_FACES>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
face_duals() {
    #ifdef GRACE_3D 
    return { 1, 0, 3, 2, 5, 4 }; 
    #else 
    return {} ; 
    #endif 
}

int8_t
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
face_dual(int8_t const iface) {
    auto const fd = face_duals() ; 
    return fd[iface] ; 
}

#ifdef GRACE_3D 
std::array<std::array<int8_t, P8EST_EDGES>, P8EST_FACES>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
face_edges() {
    return {
        { 4, 6,  8, 10 }, // 0
        { 5, 7,  9, 11 }, // 1
        { 0, 2,  8,  9 }, // 2
        { 1, 3, 10, 11 }, // 3
        { 0, 1,  4,  5 }, // 4
        { 2, 3,  6,  7 }
    } ; 
}

int8_t
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
face_edge(int8_t const iface, int8_t const iedge) {
    auto f2e = face_edges() ; 
    return f2e[iface][iedge] ; 
}
#endif 


}