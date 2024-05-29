/**
 * @file vtk_output_auxiliaries.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-17
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

#include <grace/IO/vtk_output_auxiliaries.hh>
#include <grace/system/grace_runtime.hh>

namespace grace { namespace IO {

namespace detail {

std::vector<std::string> _volume_filenames ; 
std::vector<int> _volume_iterations ; 
std::vector<double> _volume_times   ; 

std::vector<std::vector<std::string>> _surface_plane_filenames ; 
std::vector<int> _surface_plane_iterations ; 
std::vector<double> _surface_plane_times   ; 

std::vector<std::vector<std::string>> _surface_sphere_filenames ; 
std::vector<int> _surface_sphere_iterations ; 
std::vector<double> _surface_sphere_times   ; 

void init_auxiliaries() {
    auto& runtime = grace::runtime::get() ;
    int n_planes = runtime.n_surface_output_planes() ; 
    int n_spheres = runtime.n_surface_output_spheres() ; 
    _surface_plane_filenames.resize(n_planes) ; 
    _surface_sphere_filenames.resize(n_spheres) ; 
}

}

}}