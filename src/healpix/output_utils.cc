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
#include "cctk_Parameters.h"

#include "inc/output_utils.hh"

#include <hdf5.h>
#include <cstdlib>
#include <cassert>

#include <util_String.h>

namespace healpix {
namespace outpututils {


  void InitFile(CCTK_ARGUMENTS, healpix_sphere_t const& slice ) {
    DECLARE_CCTK_ARGUMENTS ;
    DECLARE_CCTK_PARAMETERS ;

    int const myProc = CCTK_MyProc(cctkGH) ;

    int const det_id = slice.get_id() ; 
    CCTK_REAL const det_radius = slice.get_radius() ;
    
    char *fn = nullptr;

    hid_t file_id, dspace_id, dset_id, group_id ;

    hsize_t size{1};
    herr_t h5err ;

    // the master rank takes care of HDF5 I/O
    // Here notice that it may not hold any information regarding
    // this particular surface. But it doesn't matter since we're
    // only writing its radius now.
    if ( myProc == 0 )
      {
	Util_asprintf(&fn, "%s/healpix_det_%d_surf.h5",out_dir,det_id) ;
	assert(fn) ; 

	hid_t is_hdf5;
	H5E_BEGIN_TRY { is_hdf5 = H5Fopen(fn,
                                          H5F_ACC_RDWR,
                                          H5P_DEFAULT ) ; }
	H5E_END_TRY ;
	// if the file exists we have nothing to initialize
	if ( is_hdf5 >= 0)
	  return ;
	
	file_id = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) ;
	assert( file_id >= 0 ) ; assert(file_id>=0);

	dspace_id = H5Screate_simple( 1, &size, nullptr) ;
	dset_id = H5Dcreate2( file_id, "/radius", H5T_NATIVE_DOUBLE, dspace_id,
			      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) ; assert(dset_id>=0);

	h5err = H5Dwrite( dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
			  H5P_DEFAULT, &det_radius ) ; assert(h5err>=0);

	h5err = H5Dclose(dset_id) ; assert(h5err>=0);
	h5err = H5Sclose(dspace_id) ; assert(h5err>=0);

	 // create /data group
        group_id = H5Gcreate(file_id,"data",H5P_DEFAULT,H5P_DEFAULT,
                             H5P_DEFAULT); assert(group_id>=0); assert(group_id>=0);

        h5err = H5Gclose(group_id);assert(h5err>=0);

        h5err = H5Fclose(file_id) ;assert(h5err>=0);
	
      }

    return ; 
  }

  void CreateIter(CCTK_ARGUMENTS, int det_id ) {
    DECLARE_CCTK_ARGUMENTS;
    DECLARE_CCTK_PARAMETERS;
    hid_t file_id, dspace_id, dset_id, group_id, dgroup_id ;
    herr_t h5err ;

    char *fn ;
    Util_asprintf(&fn,"%s/healpix_det_%d_surf.h5",out_dir,det_id) ;

    file_id = H5Fopen( fn,
		       H5F_ACC_RDWR,
		       H5P_DEFAULT ) ; assert(file_id>=0);

    group_id = H5Gopen( file_id,
			"data",
			H5P_DEFAULT ) ; assert(group_id>=0);

    char grpname[200] ;
    sprintf( grpname, "%d", cctk_iteration ) ;

    dgroup_id = H5Gcreate( group_id,
			   grpname, H5P_DEFAULT,
			   H5P_DEFAULT, H5P_DEFAULT ) ;

    hsize_t npoints = 1 ;
    CCTK_REAL time = cctk_time ; 
    dspace_id =  H5Screate_simple( 1, &npoints, nullptr ) ; assert(dspace_id>=0);

    dset_id   = H5Dcreate2( dgroup_id, "time", H5T_NATIVE_DOUBLE, dspace_id,
			    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) ; assert(dset_id>=0);

    h5err = H5Dwrite( dset_id , H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &time ) ; 
    assert( h5err >= 0 ) ;

    h5err = H5Dclose(dset_id) ; assert( h5err >= 0 ) ;
    h5err = H5Sclose(dspace_id); assert( h5err >= 0 ) ;
    
    h5err = H5Gclose(group_id); assert( h5err >= 0 ) ;
    h5err = H5Gclose(dgroup_id); assert( h5err >= 0 ) ;
    h5err = H5Fclose(file_id); assert( h5err >= 0 ) ;

    return ;
  }
  
  void WriteField(CCTK_ARGUMENTS, CCTK_REAL * data_buffer, size_t gsh, int det_id, std::string const& vname ) {
    DECLARE_CCTK_ARGUMENTS;
    DECLARE_CCTK_PARAMETERS;
    
    hid_t file_id, dspace_id, dset_id, group_id, dgroup_id ;
    hsize_t npoints{gsh} ;
    herr_t h5err ;

    char *fn ;
    Util_asprintf(&fn,"%s/healpix_det_%d_surf.h5",out_dir,det_id) ;

    file_id = H5Fopen( fn,
		       H5F_ACC_RDWR,
		       H5P_DEFAULT ) ; assert(file_id>=0);

    group_id = H5Gopen( file_id,
			"data",
			H5P_DEFAULT ) ; assert(group_id>=0) ;

    char grpname[200] ;
    sprintf( grpname, "%d", cctk_iteration ) ;

    dgroup_id = H5Gopen( group_id,
			 grpname,
			 H5P_DEFAULT ) ; assert(dgroup_id>=0);

    dspace_id =  H5Screate_simple( 1, &npoints, nullptr ) ; assert(dspace_id>=0);

    dset_id   = H5Dcreate2( dgroup_id, vname.c_str(), H5T_NATIVE_DOUBLE, dspace_id,
			    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) ; assert(dset_id>=0) ;
    
    h5err = H5Dwrite( dset_id , H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, reinterpret_cast<void const *> (data_buffer) ) ;
    assert( h5err >= 0 ) ;

    h5err = H5Dclose(dset_id) ; assert( h5err >= 0 ) ;
    h5err = H5Sclose(dspace_id); assert( h5err >= 0 ) ;
    h5err = H5Gclose(dgroup_id); assert( h5err >= 0 ) ;
    h5err = H5Gclose(group_id); assert( h5err >= 0 ) ;
    h5err = H5Fclose(file_id); assert( h5err >= 0 ) ;

    return ;
    
  }

}
}
