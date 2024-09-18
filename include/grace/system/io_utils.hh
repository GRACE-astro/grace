/**
 * @file io_utils.hh
 * @author  Carlo Musolino
 * @brief 
 * @date 2024-09-16
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

#ifndef GRACE_SYSTEM_IO_UTILS_HH
#define GRACE_SYSTEM_IO_UTILS_HH

#include <grace_config.h>

#include <grace/amr/p4est_headers.hh>

#include <vector>
#include <string> 
#include <cmath> 

namespace grace {

class io_utils_impl_t {

 static constexpr size_t nvertex = P4EST_CHILDREN ; 

 public: 
    
    io_utils_impl_t() = default ;

 private: 
    
    std::vector<double> _corner_coords ; 
    std::vector<unsigned int> _cell_corner_ids ; 

} ; 


}
#endif /* GRACE_SYSTEM_IO_UTILS_HH */