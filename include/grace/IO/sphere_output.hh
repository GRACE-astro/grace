

/**
 * @file sphere_output.hh
 * @authors Konrad Topolski, Kenneth Miller 
 * @brief 
 * @date 2025-03-21
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

#ifndef GRACE_SPHERE_OUTPUT
#define GRACE_SPHERE_OUTPUT

#include <grace_config.h> 
#include <hdf5.h>

#include <array>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <cmath>

#include <grace/healpix/detectors.hh>


namespace grace { namespace IO {


    using namespace healpix;
    extern std::map<std::string, healpix_detector> detectors;



    void initialize_spherical_detectors(const int n_detectors, 
                                        const int nside,
                                        const std::vector<std::array<double,3>> output_spheres_centres,
                                        const std::vector<double> output_spheres_radii,
                                        const std::vector<std::string> output_spheres_names);

    void update_spherical_detectors();

    void compute_multipoles();
    
    void compute_spherical_surface_variable_data();

    void write_sphere_cell_data_hdf5() ; 

    void write_multipole_and_integral_timeseries() ; 

    }

}



#endif /** GRACE_SPHERE_OUTPUT  */