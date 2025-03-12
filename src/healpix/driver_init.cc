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
#include "inc/output_utils.hh"
#include "inc/healpix_sphere.hh"
#include "inc/commstack.hh"

#include <vector>

namespace healpix {
  int healpix_sphere_t::surface_count_ = 0 ;

  bool symm::ref_x = false ;
  bool symm::ref_y = false ;
  bool symm::ref_z = false ; 

// this is where we hold all the spheres
std::vector<healpix_sphere_t> slices  ;

}

using namespace healpix ;

extern "C" void healpix_init(CCTK_ARGUMENTS) {
  
  
  DECLARE_CCTK_ARGUMENTS ;
  DECLARE_CCTK_PARAMETERS ;
 
  if ( verbose )
    CCTK_INFO("Initializing output files and spheres") ;

  commstack_t commstack(cctkGH); 

  distrib_t distrib_;

  if( CCTK_Equals(distrib_method, "split")  ) {
    distrib_ = SPLIT ; 
  } else if ( CCTK_Equals(distrib_method, "const")  ) {
    distrib_ = CONST ;
  } else if ( CCTK_Equals(distrib_method, "single")  ) {
    distrib_ = SINGLE ;
  }
  
  for ( int i =0; i < num_slices ; i++ ) {
    slices.push_back( healpix_sphere_t( cctkGH,
					i,
					commstack,
					distrib_,
					slice_radius[i],
					slice_resolution[i] ) );

    outpututils::InitFile(CCTK_PASS_CTOC,slices[i] ) ;
  }

  // detect symmetries
  if ( CCTK_IsThornActive("ReflectionSymmetry")  !=0 ) {

    if( CCTK_Equals(CCTK_ParameterValString("reflection_x","ReflectionSymmetry"),"yes") ){
      symm::ref_x = true ;
      if ( veryverbose )
	CCTK_INFO("x-reflection symmetry is active") ; 
    }
    if( CCTK_Equals(CCTK_ParameterValString("reflection_y","ReflectionSymmetry"),"yes") ){
      symm::ref_y = true ;
      if ( veryverbose )
	CCTK_INFO("y-reflection symmetry is active") ;
    }
    if( CCTK_Equals(CCTK_ParameterValString("reflection_z","ReflectionSymmetry"),"yes") ){
      symm::ref_z = true ;
      if ( veryverbose )
	CCTK_INFO("z-reflection symmetry is active") ;
    } 
  }

  if ( CCTK_IsThornActive("RotatingSymmetry180") )
    CCTK_VWarn( CCTK_WARN_ALERT, __LINE__, __FILE__, CCTK_THORNSTRING,
                "Warning! Rotation symmetry detected. This is not fully supported by this thorn.");
  
  if ( verbose )
    CCTK_VInfo(CCTK_THORNSTRING, "Initialised %d spherical surfaces", healpix_sphere_t::surface_count_ ) ;

  return ; 

}
