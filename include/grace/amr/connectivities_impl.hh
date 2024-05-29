/**
 * @file connectivities_impl.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-06-09
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
#ifndef E6509461_33AC_416F_9732_3225C70D796D
#define E6509461_33AC_416F_9732_3225C70D796D
#include <grace_config.h>
#include <grace/amr/p4est_headers.hh>

namespace grace { namespace amr { namespace detail { 
#ifdef GRACE_3D
p4est_connectivity_t*
new_cartesian_connectivity( double xmin, double xmax, bool periodic_x
                            , double ymin, double ymax, bool periodic_y 
                            , double zmin, double zmax, bool periodic_z ) ;

p4est_connectivity_t*
new_spherical_connectivity( double rmin, double rmax, double rmax_log, bool extend_with_logr ) ; 
#else 
p4est_connectivity_t*
new_spherical_connectivity( double rmin, double rmax, double rmax_log, bool extend_with_logr ) ; 
p4est_connectivity_t*
new_cartesian_connectivity( double xmin, double xmax, bool periodic_x
                             , double ymin, double ymax, bool periodic_y );
#endif 
} } } /* namespace grace::amr::detail */ 

#endif /* E6509461_33AC_416F_9732_3225C70D796D */
