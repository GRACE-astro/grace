/*
 * Copyright (C) 2022 Carlo Musolino
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include "types.hh"
#include "commstack.hh"

#include <vector>
#include <cassert> 

namespace healpix {

  class healpix_data_t {
  public :
    std::vector<double> data ;  
    bool integrate_, output_ ;
    std::string output_var_ = "" ; 
    std::string input_var_;
    
    healpix_data_t( size_t lsh, bool integrate, bool output, char const* input_var,
		    char const * output_var="" ): input_var_(input_var), integrate_(integrate), output_(output), output_var_(output_var)
    {
      data.resize(lsh) ;
    }

    ~healpix_data_t () {} ;

  // pointer to the memory address where the i-th GRACE variable 
  // is it even needed in the re-design? 

  //   CCTK_REAL* outpointer(const cGH* const cctkGH) const {
  //     int const out_index = CCTK_VarIndex(output_var_.c_str() ) ;
  //     if( out_index < 0 ) {
	// CCTK_VWarn(CCTK_WARN_ABORT , __LINE__, __FILE__, CCTK_THORNSTRING,
	// 	   "Couldn't get variable index for variable %s", output_var_.c_str() ) ;
  //     }
  //     return (CCTK_REAL*) CCTK_VarDataPtrB(cctkGH,0,out_index, nullptr) ;
  //   }
    
  } ;
  
}
