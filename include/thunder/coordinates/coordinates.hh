/**
 * @file coordinates.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
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
#ifndef F25FCED7_32FD_48EF_A294_4D29ABC78524
#define F25FCED7_32FD_48EF_A294_4D29ABC78524

#include <thunder_config.h>

#include <thunder/data_structures/variable_properties.hh>

namespace thunder { 

/**
 * @brief Fill cell coordinates array.
 * \ingroup amr 
 */
void fill_cell_coordinates(coord_array_t<THUNDER_NSPACEDIM>&, scalar_array_t<THUNDER_NSPACEDIM>&) ; 
/*
struct coordinate_converter_impl_t 
{
    double _F0, _Fr, _F1, _Fr1 ; 
    double _S0, _Sr, _S1, _Sr1 ; 

    coordinate_converter_impl_t() ; 

    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    transform_coordinates(
        int in_tree, int face,
        VEC(double const& x, double const& y, double const& z),
        VEC(double& zeta, double& xi, double& eta)
    ) 
    {
        transform_coordinates_impl[itree*P4EST_FACES + face](
            VEC(x,y,z),VEC(zeta,xi,eta)
        );
    }; 

 private:
    
    typedef void (*coord_func_t) THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE (
        VEC(double const&, double const&, double const&),
        VEC(double&, double&, double&)
    )  ;

    coord_func_t transform_coordinates_impl[] ; 

};
*/
} /* namespace thunder */ 

#endif /* F25FCED7_32FD_48EF_A294_4D29ABC78524 */
