/**
 * @file longevity_policies.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2023-06-14
 * 
 * @copyright This file is part of MagMA.
 * MagMA is an evolution framework that uses Finite Difference
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
#ifndef GRACE_UTILS_LONGEVITY_POLICIES_HH
#define GRACE_UTILS_LONGEVITY_POLICIES_HH

namespace utils { 

template< typename T> 
struct default_longevity 
{
    static void schedule_destruction(T)
}

}

#endif /* GRACE_UTILS_LONGEVITY_POLICIES_HH */
