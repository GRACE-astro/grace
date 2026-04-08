/**
 * @file import_kadath.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * Subject to GPL and adapted from the work of author(s)/maintainer(s):
 * Samuel David Tootle <tootle@itp.uni-frankfurt.de>
 * Ludwig Jens Papenfort <papenfort@th.physik.uni-frankfurt.de>
 * @brief 
 * @date 2024-08-29
 * The following is a minimal header file to include the necessary
 * enumerators and function defintions
 * that reflect those used by the exporters included in the FUKA branch of Kadath.
 * This makes it possible to only need to compile the application with 
 * the static libary stored in ${HOME_KADATH}/lib
 * without needing to include or compile with Kadath header files.
 * This is especially important since features in FUKA require C++17.
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

#ifndef GRACE_IMPORT_KADATH_HH
#define GRACE_IMPORT_KADATH_HH

#include <array>
#include <string>
#include <vector>
#include <functional>

// choose the serial or OpenMP parallelized mode - depends crucially on the setup of the underlying kadath library
#define KADATH_EXPORTERS_PARALLEL 

#ifndef KADATH_EXPORTERS_PARALLEL
#define KADATH_EXPORTERS_SERIAL
#endif 

// enumeration for quantities to export to evolution codes
enum sim_vac_quants {
  K_ALPHA,
  K_BETAX,
  K_BETAY,
  K_BETAZ,
  K_GXX,
  K_GXY,
  K_GXZ,
  K_GYY,
  K_GYZ,
  K_GZZ,
  K_KXX,
  K_KXY,
  K_KXZ,
  K_KYY,
  K_KYZ,
  K_KZZ,
  NUM_VOUT
};

// chained enums together to include matter quantities
enum sim_matter_quants {
  K_RHO=NUM_VOUT,
  K_EPS,
  K_PRESS,
  K_VELX,
  K_VELY,
  K_VELZ,
  NUM_OUT
};


constexpr const size_t VAC = 1;
constexpr const size_t MATTER = 2;
constexpr const size_t NUM_MATTER = 6; 

//*****************************************************************************************************
/**
 * @brief  general wrapper for ID choice: copies the coordinates, picks the ID type and fills the array of references to the state of auxiliary quants
 * @param kadath_id The type of the ID: BH/NS/BBH/BNS/BHNS
 * @param kadath_id_file   The full path to the Kadath .info ID file
 * @param state_ref references (via reference_wrapper) to individual entries (field, point) in the auxiliary state array
 * @param pcoords   The (by assumption global & Cartesian!) coordinates in Kadath space 
 * @param nfields   The number of fields to fill out (extra fields are needed for hydro import)
*/
void KadathImporter(const std::string kadath_id, const std::string  filename,
                    const std::vector<double> & xx, 
                    const std::vector<double> & yy, 
                    const std::vector<double> & zz,
                    Kokkos::View<double *****, grace::default_space>& ddata, 
                    const int nfields, const int npoints, size_t nx, size_t ny, size_t nz, size_t ngz) ;
//}

#endif /* GRACE_IMPORT_KADATH_HH */
