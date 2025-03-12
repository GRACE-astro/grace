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


extern "C" void healpix_output(CCTK_ARGUMENTS) {
  using namespace healpix;

  DECLARE_CCTK_ARGUMENTS ;
  DECLARE_CCTK_PARAMETERS ;

  if ( cctk_iteration % output_every | output_every == 0 )
    return ;
  
  healpix_err_t herr ; 
  
  if ( verbose || veryverbose ) 
    CCTK_INFO("Performing output of requested GFs on spherical surfaces") ;

  for( auto &slice : slices ) {

    if ( veryverbose )
      CCTK_VInfo(CCTK_THORNSTRING, "Sphere %d: %d variables registered for output",slice.get_id(), slice.get_num_vars_output() ) ;
    
    herr = slice.output_all(CCTK_PASS_CTOC) ;

    if ( ! herr == healpix_err_t::NO_ERROR ) {
      CCTK_VWarn(CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
		 "Output returned error code %d on slice %d",slice.get_id(),herr) ;  
    }

    if ( veryverbose )
      CCTK_VInfo(CCTK_THORNSTRING, "Sphere %d: output done.",slice.get_id() ) ;
  }

  return ;

}

