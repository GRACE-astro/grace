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

#include "healpix_sphere_data.hh"
#include "commstack.hh"
#include "chealpix.h"
#include "types.hh"

#include <memory>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <cmath>

#include <iostream>

namespace healpix {
  
  class healpix_sphere_t
  {
  public :
    healpix_sphere_t(int const id,
		     commstack_t& cs,
		     distrib_t distrib, 
		     double radius,
		     size_t nsides ) : id_(id), nsides_(nsides), radius_(radius), commstack_(cs), distrib_(distrib) 
    {
      surface_count_ ++ ;
      npoints_ = nside2npix(nsides) ;
      surface_elem_ = 4. * M_PI * radius_ * radius / npoints_ ;
      num_vars_integrate_ = 0 ;
      num_vars_output_ = 0 ;
      gsh_ = npoints_ ; 
      
      if ( distrib == CONST ) {
	// This is easy !
	if ( commstack_.get_myProc() == 0 ) {
	  lsh_ = npoints_ ;
	  lbnd_ = 0; ubnd_ = gsh_ ;
	} else {
	  lsh_ = 0;
	  lbnd_= ubnd_ = 0 ;
	}
      } else if ( distrib == SINGLE ) {
	if ( commstack_.get_myProc() == id_ % commstack_.get_nProcs() ) {
	  lsh_ = npoints_ ;
	  lbnd_ = 0; ubnd_ = gsh_ ;
	} else {
	  lsh_ = 0;
	  lbnd_ = ubnd_ = 0; 
	}
      } else if ( distrib == SPLIT ) {
	// round robing point distribution,
	// assumes that gsh_ is set! 
	balance_load() ; 
      }
      
      fill_coordinates() ;  
    }
    
    ~healpix_sphere_t() {} ;
    
    // getters
    int get_id() const { return id_ ; } ;
    size_t get_nsides() const { return nsides_ ; } ;
    size_t get_npoints() const { return npoints_ ; } ;
    int get_num_vars() const { return data.size() ; } ;
    int get_num_vars_integrate() const { return num_vars_integrate_ ; } ;
    int get_num_vars_output() const { return num_vars_output_ ; } ;
    double get_radius() const { return radius_ ; } ;
    double get_surface_elem() const { return surface_elem_ ; } ;
    // register 
    std::tuple<int,healpix_err_t> register_var(bool integrate, bool output, char const * input_var, char const * output_var) ;
    // comptations
    healpix_err_t interpolate_all() ;
    healpix_err_t integrate_all() ;
    // output
    healpix_err_t output_all() ; 
    
  private:
    void fill_coordinates() ;
    void balance_load(cGH* cctkGH) ;
    
    // bookkeeping
    int id_ ; 
    double radius_ ;
    double surface_elem_ ;
    size_t nsides_ ;
    size_t npoints_ ;
    int num_vars_integrate_ ;
    int num_vars_output_ ;
    // MPI shenanigans
    commstack_t commstack_ ;
    distrib_t distrib_ ;
    size_t gsh_ ;
    size_t lsh_ ;
    size_t lbnd_,ubnd_ ;
    // cartesian coordinates 
    std::vector<double> tx ;
    std::vector<double> ty ;
    std::vector<double> tz ;
    // payload 
    std::vector<healpix_data_t> data ;
  public:
    static int surface_count_ ; 
  };
}
