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
#include "inc/healpix_sphere.hh"
#include "inc/output_utils.hh"

#include <util_Table.h>
#include <util_String.h>

namespace healpix {
  
  std::tuple<int,healpix_err_t> healpix_sphere_t::register_var(bool integrate, bool output, char const * input_var, char const * output_var="") {
    if ( integrate )
      num_vars_integrate_ ++ ;
    if ( output )
      num_vars_output_ ++ ;
    data.push_back( healpix_data_t(lsh_, integrate,output,input_var,output_var) );    
    return std::make_tuple(data.size(), NO_ERROR) ;
  }

  void healpix_sphere_t::fill_coordinates() {
    tx.resize(lsh_); ty.resize(lsh_); tz.resize(lsh_);

#pragma omp parallel for schedule(static)
    for( long ipix=0; ipix < lsh_; ipix++) {
      double vec[3] ;
      // for now we hard-code this, then we'll see! 
      pix2vec_ring(nsides_, ipix + lbnd_, vec) ;

      tx[ipix] = symm::ref_x ? fabs(radius_ * vec[0]) : radius_ * vec[0] ;
      ty[ipix] = symm::ref_y ? fabs(radius_ * vec[1]) : radius_ * vec[1] ;
      tz[ipix] = symm::ref_z ? fabs(radius_ * vec[2]) : radius_ * vec[2] ;
    }

  }

  // Very simple round-robin load balancin that
  // splits the point across ALL ranks.
  // An immediate improvement ( which would make the
  // code usage a bit harder perhaps ) would be
  // to add the possibility of only using a user set number of
  // processors 
  void healpix_sphere_t::balance_load(cGH* cctkGH) {
    
    int const nProcs = commstack_.get_nProcs() ; 
    int const myProc = commstack_.get_myProc(cctkGH) ; 

    // round-robin distribution among processes

    lsh_ = gsh_ / nProcs ;

    int const mod = gsh_ % nProcs ;
    if ( myProc < mod ){
      lsh_++ ; 
    }

    lbnd_ = (gsh_/nProcs) * myProc ;
    lbnd_ += (myProc > mod) ? mod : myProc ; 
    ubnd_ = lbnd_ + lsh_ ;
  }

  

  // computations

  // Interpolate data ! 
  healpix_err_t healpix_sphere_t::interpolate_all(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTS ;
    DECLARE_CCTK_PARAMETERS ;

    int const ninputs = data.size() ;

    if ( !ninputs )
      return NO_ERROR ;

    int const interp_handle = CCTK_InterpHandle (interpolator) ;
    assert ( interp_handle >= 0 ) ;
    int const options_table =
      Util_TableCreateFromString ( interpolator_options ) ;
    assert ( options_table >= 0 ) ;
    int const coord_handle = CCTK_CoordSystemHandle (coord_system) ;

    CCTK_INT input_array_indices[ninputs] ;
    CCTK_INT output_array_types[ninputs] ;
    void *  output_arrays[ninputs] ;
    
    for( int i=0 ; i < ninputs; i++) {      
      input_array_indices[i] = CCTK_VarIndex(data[i].input_var_.c_str()) ;
      output_array_types[i] = CCTK_VARIABLE_REAL ;
      output_arrays[i] = reinterpret_cast<void *> (data[i].data.data() ) ; //yuhuuu
    }

    void const * const interp_coords[] = {
      reinterpret_cast<void *> ( tx.data() ),
      reinterpret_cast<void *> ( ty.data() ),
      reinterpret_cast<void *> ( tz.data() ) } ;
      
    
    int ierr = CCTK_InterpGridArrays( cctkGH,
				      3,
				      interp_handle, options_table, coord_handle,
				      lsh_, CCTK_VARIABLE_REAL, interp_coords,
				      ninputs, input_array_indices,
				      ninputs, output_array_types, output_arrays ) ;

    Util_TableDestroy (options_table);
    
    if ( ierr < 0 ) {
      return INTERP_ERROR ; 
    }

    return NO_ERROR ;
  }


  healpix_err_t healpix_sphere_t::integrate_all(CCTK_ARGUMENTS) {
    // We just do local reductions for each variable,
    // push it to the commstack buffer,
    // call Allreduce ( through commstack ) and write results
    DECLARE_CCTK_ARGUMENTS ;

    commstack_.clear_buffer() ; 
    
    // Loop through variables registered on this surface
    for( auto const& var : data) {
      // nothing to do
      if( ! var.integrate_ )
	continue ;
      CCTK_REAL tmp_buffer{0} ; 
#pragma omp parallel for schedule(static) reduction(+:tmp_buffer)
      for( int i=0 ; i<lsh_ ; i++ ) 
	tmp_buffer += var.data[i] * surface_elem_ ;

      // the ordering of the data elements in the sphere->data
      // vector is crucial.. This might be a problem and we should
      // check this more strongly perhaps! 
      commstack_.push(tmp_buffer) ;
    }
    
    commstack_.reduce(MPI_SUM) ;

    // Finally write back to carpet
    int ibuf{0} ; 
    for(auto const& var: data) {
      if( ! var.integrate_ )
	continue ;
      //CCTK_REAL * varptr = (CCTK_REAL*) CCTK_VarDataPtr(cctkGH, 0, var.output_var_.c_str());
      CCTK_REAL * varptr = var.outpointer(cctkGH) ; 
      *varptr = commstack_.reduced_val(ibuf) ;
      ibuf++ ;
    }
    
    return NO_ERROR ; 
  }
  
  healpix_err_t healpix_sphere_t::output_all(CCTK_ARGUMENTS) {
    // we need to gather the data for which output is requested.
    // then the master rank will put it into an hdf5 .
    DECLARE_CCTK_ARGUMENTS ; 
    int const myProc = commstack_.get_myProc(cctkGH) ;

    // Create hdf5 group for current iteration, write current time
    if ( myProc == 0 )
      outpututils::CreateIter(CCTK_PASS_CTOC,id_) ; 
    
    for( auto & var : data ) {
      if ( ! var.output_ )
	continue ;

      CCTK_REAL* buffer = (CCTK_REAL*) malloc ( gsh_ * sizeof(CCTK_REAL) );	
      commstack_.allgather(reinterpret_cast<void const *> (var.data.data()), buffer, lsh_, lbnd_ ) ;
      
      if ( myProc == 0 ) {
	outpututils::WriteField(CCTK_PASS_CTOC, buffer, gsh_, id_, var.input_var_) ;
      }
      
      free(buffer) ;
    }

    return NO_ERROR ;
    
  }

}
