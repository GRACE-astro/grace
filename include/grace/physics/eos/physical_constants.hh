/**
 * @file physical_constants.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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
#ifndef GRACE_PHYS_CONSTANTS_HH
#define GRACE_PHYS_CONSTANTS_HH
namespace grace { namespace physical_constants {

#define CONSTDEF(x,y) static constexpr double x = y 

// constants SI 
CONSTDEF(G_si, 6.6738e-11); // m^3 / (kg s^2)
CONSTDEF(c_si, 299792458.0); // m/s
CONSTDEF(Msun_si, 1.9885e30); // kg 
CONSTDEF(mu0_si, 1.256637061435917e-6); // Newton/Ampere^2
CONSTDEF(Kb_si, 1.3806488e-23); // J / K 
CONSTDEF(fm_si, 1e-15); // m 
CONSTDEF(e_si, 1.602176634e-19); // kg 
CONSTDEF(h_si, 6.62607015e-34); // m^2 kg / s 

// particle masses 
CONSTDEF(me_MeV,0.51099895069); // electron mass 
CONSTDEF(mp_MeV,938.27208943);  // electron mass 
CONSTDEF(mn_MeV,939.56542194);  // electron mass 

// fine structure constant
CONSTDEF(alpha_fine,1./137.);  // electron mass 

// Scattering 
CONSTDEF(sigmaT_cgs, 6.6524587051e-25) ; // Thompson scattering 

// Magnetic field 
CONSTDEF(CU_to_Tesla, 8352228495337975.0) ; // Geometric units to Tesla 


// Conversion 
CONSTDEF(MeV_to_kg, 1.782661906940917e-30) ; 
CONSTDEF(MeV_to_g, 1.782661906940917e-27) ;

} } /* namespace grace::physical_constants */

#endif 