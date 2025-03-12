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

#include <mpi.h>
#include <stdlib.h>
#include <vector>
#include <cassert>


namespace healpix {

  class commstack_t {
    MPI_Comm comm_; 
    size_t nProcs;
    
    bool reduced_ ;
    std::vector<double> buffer_ ;
    
  public :
    commstack_t() {
      // TODO: do we want to support partial communicators?
      // prob more trouble than it's worth! 
      comm_ = MPI_COMM_WORLD ;
      MPI_Comm_size(comm_, &nProcs) ;
    } ;

    ~ commstack_t() {} ;
    
    size_t get_nProcs() const {
      return nProcs;
    } ;

    size_t get_myProc() const {
      int rank;
      MPI_Comm_rank(&rank) ;
      return rank;
    } ; 

    void push( double val ) ;

    void clear_buffer() ;
    
    void reduce(MPI_Op op_) ;

    void allgather(void const* buffer_send, void* buffer_recv , int const lsh, int const lbnd) const;

    CCTK_REAL reduced_val( const int i) const {
      assert(reduced_) ;
      return buffer_[i] ;
    }
    
    CCTK_REAL buffer_val ( const int i) const {
      return buffer_[i] ;
    }
  };
}
