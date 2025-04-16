/**
 * @file cell_output.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-24
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#include <grace/IO/hdf5_output.hh>
#include <grace/IO/vtk_output.hh>
#include <grace/config/config_parser.hh>
#include <iostream>

namespace grace { namespace IO {

void write_cell_output(bool volume_output, bool surface_output_plane, bool surface_output_sphere )
{
    if(     (not volume_output)
        and (not surface_output_plane)
        and (not surface_output_sphere) )
    {
        return ; 
    }

    auto& params = grace::config_parser::get() ; 
    bool output_hdf5 = params["IO"]["output_use_hdf5"].as<bool>() ; 

    if( output_hdf5 ) {                                                                // sliced
        write_cell_data_hdf5(volume_output,surface_output_plane,surface_output_sphere) ; 
    } else {
        write_cell_data_vtk(volume_output,surface_output_plane,surface_output_sphere) ;
    }
}

}}