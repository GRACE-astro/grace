/**
 * @file detectors.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-03-12
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

#include <grace/utils/grace_utils.hh>
#include <grace/utils/format_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>

#include <grace/healpix/detectors.hh> 



namespace grace {
namespace healpix {


void GRACE_HOST 
get_coord_from_healpix_index(
        const size_t nside, const double radius, const int idx,
        double& x1, double& x2, double& x3){
        // get the coordinate from the healpix ring index form, see arXiv:astro-ph/0409513
        // idx should run from 0 to 6N**2+2N-1 (inclusive) for the northern hemisphere+equitor if zsymm
        // idx should run from 0 to 12N**2-1 (inclusive) when z symmetry is not present 
        // local variables
        size_t  i, j, ss;
        double ph, z, phi, theta, aux, dummy;

        // calculate the northern polar cap
        if (idx<2*(nside-1)*nside){
        ph = (idx+1.)/2;
        aux = int(ph);
        aux = sqrt(ph-sqrt(aux));
        i = int(aux)+1;
        j = idx+1-2*i*(i-1);
        z = 1-i*i/(3.0*nside*nside);
        ss = 1;
        phi = M_PI*(j-ss/2.0)/(2*i);
        } // calculate the northern and southern belt
        else if (idx<10*nside*nside+2*nside){
        ph = idx-2*nside*(nside-1);
        i = int(ph/(4.*nside))+nside;
        j =  std::fmod(ph, 4.*nside)+1;
        z = 4.0/3.0-2*i/(3.0*nside);
        ss = (i-nside+1) % 2;
        phi = M_PI*(j+ss/2.0-1.0)/(2*nside);
        } // calculate the southern polar cap
        else if (idx<12*nside*nside){
        ph = ((12*nside*nside-idx-1)+1.)/2;
        aux = int(ph);
        aux = sqrt(ph-sqrt(aux));
        i = int(aux)+1;
        j = idx+1-2*i*(i-1);
        z = 1-i*i/(3.0*nside*nside);
        ss = 1;
        phi = 2*M_PI-M_PI*(j-ss/2.0)/(2*i);
        }
        else {
        ERROR("An index larger than 12*nside**2 is not allowed in healpix outflow!");
        }        
        // output coordinate according to coordinate type
        
        theta = std::acos(z);
        x1 = std::sin(theta) * std::cos(phi) * radius;
        x2 = std::sin(theta) * std::sin(phi) * radius;
        x3 = radius*z;
}


// definition of the compute fluxes function...  
void GRACE_HOST_DEVICE
compute_surface_fluxes(){

// auto& params = grace::config_parser::get() 

}

}
}