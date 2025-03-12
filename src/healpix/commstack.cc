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

#include "inc/commstack.hh"
#include "inc/types.hh"

namespace healpix {

  void commstack_t::push(CCTK_REAL val) {
    buffer_.push_back(val) ;
    reduced_ = false ;
  }

  void commstack_t::clear_buffer() {
    reduced_ = false ;
    buffer_.clear() ;
  }
  
  void commstack_t::reduce(MPI_Op op_) {
    
    std::vector<CCTK_REAL> dummy(buffer_.size(), 0) ;
    
    MPI_Allreduce( &buffer_.front(), &dummy.front(), buffer_.size(),
		   MPI_DOUBLE, op_, comm_ ) ;

    buffer_ = dummy;

    reduced_ = true;
    
  }

  
  void commstack_t::allgather(void const *buffer_send, void *buffer_recv, CCTK_INT const lsh_, CCTK_INT const lbnd_ ) const {
    
    // first we gather up all the local buffer sizes
    // from all processes
    int cnt_recv[nProcs] ;
    MPI_Allgather(&lsh_, 1, MPI_INT, 
		  &cnt_recv, 1, MPI_INT, comm_ ) ;

    // Then we recieve the lbnds ... i.e. the displacements for the data that will be gathered 
    int displacements[nProcs] ;
    MPI_Allgather(&lbnd_, 1, MPI_INT,
		  &displacements, 1, MPI_INT, comm_ );

    // now comes the real deal 
    MPI_Allgatherv( buffer_send, lsh_,
		    MPI_DOUBLE,
		    buffer_recv,
		    cnt_recv,
		    displacements,
		    MPI_DOUBLE,
		    comm_);

    return ; 
  }
}
