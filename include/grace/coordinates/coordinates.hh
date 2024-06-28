/**
 * @file coordinates.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-06-14
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
#ifndef F25FCED7_32FD_48EF_A294_4D29ABC78524
#define F25FCED7_32FD_48EF_A294_4D29ABC78524

#include <grace_config.h>

#include <grace/data_structures/variable_properties.hh>

namespace grace { 
/*****************************************************************************************/
/**
 * @brief Fill cell coordinates array.
 * \ingroup coordinates 
 */
void fill_cell_coordinates( scalar_array_t<GRACE_NSPACEDIM>&
                          , scalar_array_t<GRACE_NSPACEDIM>&
                          , scalar_array_t<GRACE_NSPACEDIM>&
                          , cell_vol_array_t<GRACE_NSPACEDIM>&
                          , staggered_coordinate_arrays_t& ) ; 
/*****************************************************************************************/
/**
 * @brief Fill a device view with physical coordinates on the grid. 
 * 
 * @param pcoords The view to be filled with coordinates.
 */
void fill_physical_coordinates( coord_array_t<GRACE_NSPACEDIM>& pcoords ) ; 
} /* namespace grace */ 

#endif /* F25FCED7_32FD_48EF_A294_4D29ABC78524 */
