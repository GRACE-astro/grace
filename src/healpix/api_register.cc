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
#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Functions.h"
#include "cctk_Parameters.h"

#include "inc/types.hh"
#include "inc/healpix_sphere.hh"
#include "inc/healpix.hh"

#include <cassert>
#include <string>
#include <vector>

using namespace healpix ; 

extern "C" CCTK_INT healpix_register_integral( const CCTK_POINTER_TO_CONST varname,
					       const CCTK_POINTER_TO_CONST outname,
					       const CCTK_INT output_ ,
					       const CCTK_INT det_id )

{
  DECLARE_CCTK_PARAMETERS ;

  const char* _varname = (char*) varname ;
  const char* _outname = (char*) outname ;


  assert(_varname); assert(_outname);

  assert( det_id < num_slices ) ;

  bool output = static_cast<bool>(output_) ;
  bool integrate = true ; 

  std::string varname_lowercase(_varname);
  for (int i=0; _varname[i]; ++i)
    {
      varname_lowercase[i] = tolower(_varname[i]);
    }
  
  std::string outname_lowercase(_outname);
  for (int i=0; _outname[i]; ++i)
    {
      outname_lowercase[i] = tolower(_outname[i]);
    }

  healpix_err_t herr ;
  int var_index ;
  std::make_tuple(var_index, herr) = slices[det_id].register_var( integrate,
								  output,
								  varname_lowercase.c_str(),
								  outname_lowercase.c_str() ) ;

  if ( ! herr == healpix_err_t::NO_ERROR ) {
    CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
		 "Registration returned error code %d on slice %d",det_id,herr) ;  
    }

  
  
  if ( verbose || veryverbose || verbose_register ) {
    CCTK_VInfo(CCTK_THORNSTRING,
	       "Registered integrated variable on sphere %d: ",det_id);

    CCTK_VInfo(CCTK_THORNSTRING,
	       "GF: %s | Scalar: %s",_varname,_outname ) ;

    if ( output )
      CCTK_VInfo(CCTK_THORNSTRING,
		 "Output requested for variable on sphere %d ",det_id);
    else
      CCTK_VInfo(CCTK_THORNSTRING,
		 "Output not requested for variable on sphere %d ",det_id);

  }
  
  return var_index ; 
  
}


extern "C" CCTK_INT healpix_register_output( const CCTK_POINTER_TO_CONST varname,
					     const CCTK_INT det_id )

{
  DECLARE_CCTK_PARAMETERS ;

  const char* _varname = (char*) varname ;
 
  assert(_varname);

  assert( det_id < num_slices ) ;
  
  bool integrate = false ; bool output = true ;

  std::string varname_lowercase(_varname);
  for (int i=0; _varname[i]; ++i)
    {
      varname_lowercase[i] = tolower(_varname[i]);
    }
  
 
  healpix_err_t herr ;
  int var_index ;
  std::make_tuple(var_index, herr) = slices[det_id].register_var( integrate,
								  output,
								  varname_lowercase.c_str(), "" ) ;

  if ( ! herr == healpix_err_t::NO_ERROR ) {
    CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__, CCTK_THORNSTRING,
		 "Registration returned error code %d on slice %d",det_id,herr) ;  
    }

  
  
  if ( verbose || veryverbose || verbose_register ) {
    CCTK_VInfo(CCTK_THORNSTRING,
	       "Registered interpolated variable on sphere %d: ",det_id);

    CCTK_VInfo(CCTK_THORNSTRING,
	       "GF: %s",_varname) ;
  }
  
  return var_index ; 
  
}
