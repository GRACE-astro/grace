/**
 * @file c2p_types
 * @author Khalil Pierre 
 * @brief 
 * @date 2026-03-23
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


#ifndef GRACE_PHYSICS_EOS_C2P_TYPES_HH
#define GRACE_PHYSICS_EOS_C2P_TYPES_HH

#include <cstdint>

namespace grace {

enum c2p_sig_t : uint8_t {
    C2P_SUCCESS=0,
    C2P_EPS_TOO_HIGH,
    C2P_EPS_TOO_LOW,
    C2P_RHO_TOO_HIGH,
    C2P_RHO_TOO_LOW,
    C2P_VEL_TOO_HIGH,
    C2P_NSIG
} ;

struct c2p_err_t {
    bool adjust_tau{false};
    bool adjust_s{false};
    bool adjust_d{false};
    bool adjust_ent{true};
} ;

} // namespace grace

#endif /* GRACE_PHYSICS_EOS_C2P_TYPES_HH */
