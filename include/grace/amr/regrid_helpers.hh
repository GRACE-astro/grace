/**
 * @file regrid_helpers.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2025-07-31
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_AMR_REGRID_HELPERS_CPP
#define GRACE_AMR_REGRID_HELPERS_CPP

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <Kokkos_Core.hpp>

#include <grace/parallel/mpi_wrappers.hh>

#include <tuple> // need make_pair 

#include <iostream> 

namespace grace { namespace amr {

template< typename ViewT > 
parallel::grace_transfer_context_t 
grace_transfer_fixed_begin(
    const p4est_gloidx_t * dest_gfq,
    const p4est_gloidx_t * src_gfq,
    sc_MPI_Comm mpicomm, int tag,
    ViewT& src, ViewT& dest, size_t data_stride
)
{
    parallel::grace_transfer_context_t context ; 

    if (data_stride==0) {
        // return empty context
        return context; 
    }
    /* Fetch communicator properties */
    int nproc = parallel::mpi_comm_size(mpicomm) ;
    int rank  = parallel::mpi_comm_rank(mpicomm) ;

    /**************************************/
    /* Fetch relevant quadrant offsets in */
    /* current (src) and updated (dest)   */
    /* partitions of the forest.          */
    /**************************************/
    auto const dest_begin = dest_gfq[rank]   ;
    auto const src_begin  = src_gfq[rank]    ;
    auto const dest_end   = dest_gfq[rank+1] ;
    auto const src_end    = src_gfq[rank+1]  ;

    /**************************************/
    /* Offsets and length for local copy  */
    /**************************************/
    size_t dest_cp_offset{0UL}, src_cp_offset{0UL} ;
    size_t cp_len{0UL} ; 

    /****************************************/
    /* First and last ranks containing data */
    /* we need.                             */
    /****************************************/
    p4est_gloidx_t      first_sender, last_sender ;

    /* Placeholders for rank-by-rank transfers */
    p4est_gloidx_t      gbegin, gend;

    /* Data size */
    size_t transfer_size{0UL} ; 

    /* If this rank is not empty */
    if (dest_begin < dest_end) {

        /* Check that the destination View makes sense */
        ASSERT(dest.size() > 0, 
               "Destination data pointer is null.") ;
        
        /****************************************************/
        /* Find process containing the first local quadrant */
        /* in the old partition (remember that the tree is  */
        /* always "contiguous" and that the rank-quadrant   */
        /* relationship is monotonic).                      */
        /****************************************************/
        first_sender = p4est_bsearch_partition (dest_begin, src_gfq, nproc);
        ASSERT( 0<= first_sender && first_sender < nproc,
                "First sender index out of bounds: "
                << first_sender << " for nproc = " << nproc) ;
        /****************************************************/
        /* Find process containing the last local quadrant  */
        /* in the old partition (ditto the remark above)    */
        /****************************************************/
        last_sender =
            p4est_bsearch_partition (dest_end - 1, &src_gfq[first_sender],
                                     nproc - first_sender) + first_sender;
        ASSERT(first_sender <= last_sender && last_sender < nproc,
               "Last sender index out of bounds: "
               << last_sender << " for nproc = " << nproc) ;

        /****************************************************/
        /* Set up the rank-specific quadrant range endpoint */
        /* as beginning.                                    */
        /****************************************************/
        gend = dest_begin ; 

        /* Fetch vector of receive requests */
        auto& rq = context._recv_requests ; 

        /* Quadrant offset */
        size_t rb = 0UL ; 

        /****************************************************/
        /* Loop over all senders and post receive requests  */
        /****************************************************/
        for ( int q = first_sender; q <= last_sender; ++q) {
            /* Start at previous end     */
            gbegin = gend ; 
            /* End at next process begin */
            gend = src_gfq[q+1] ;
            /* If this process goes till the end of the forest */
            /* it better be the last one                       */
            if (gend>dest_end)  { 
                ASSERT( q==last_sender,
                        "Last sender index does not match the end of the "
                        "destination data: " << gend << " != " << dest_end) ;
                gend = dest_end ; 
            }
            ASSERT(q == first_sender || q == last_sender ?
                    gbegin < gend : gbegin <= gend,
                    "Invalid range for sender " << q) ;
            /* Process q is empty */       
            if ( gbegin == gend ) {
                ASSERT( first_sender < q && q < last_sender,
                        "Invalid range for sender " << q
                        << " with gbegin = " << gbegin
                        << " and gend = " << gend) ;
            } else { 
                /* Number of elements that need to be transferred */
                transfer_size = (size_t) (gend-gbegin) * data_stride ; 
                /* Are we sending to ourselves? */
                if ( q == rank ) {
                    /**************************************/
                    /* Yes, store data for later copy     */
                    /**************************************/
                    /* NB this is the number of QUADRANTS */
                    /**************************************/
                    cp_len = (size_t) (gend-gbegin) ; 
                    /**************************************/
                    /* NB this is the QUADRANT offset     */
                    /**************************************/
                    dest_cp_offset = rb ;  
                } else {
                    /**************************************/
                    /* No, post receive request           */
                    /**************************************/
                    rq.push_back(sc_MPI_Request{}) ; 
                    parallel::mpi_irecv(
                        dest.data() + rb * data_stride, transfer_size, q, tag, mpicomm, &(rq.back())
                    ) ;
                }
                /* New quad offset       */
                rb += (size_t) (gend-gbegin) ;  
            }
        }
    } /* if ( dest_begin < dest_end ) */

    /****************************************/
    /* First and last ranks to whom we must */
    /* send data.                            */
    /****************************************/
    p4est_gloidx_t first_receiver, last_receiver ; 

    /****************************************/
    /* Figure out who to send to            */
    /****************************************/
    if ( src_begin < src_end ) { 
        /* Check that the source View makes sense */
        ASSERT(src.size() > 0, 
               "Source data pointer is null.") ;
        
        /****************************************************/
        /* Find process containing the first old local      */
        /* quadrant in the new partition (ditto the remark  */
        /* above).                                          */
        /****************************************************/
        first_receiver = p4est_bsearch_partition (src_begin, dest_gfq, nproc);
        ASSERT( 0<= first_receiver && first_receiver < nproc,
                "First receiver index out of bounds: "
                << first_receiver << " for nproc = " << nproc) ;
        /****************************************************/
        /* Find process containing the last old local       */
        /* quadrant in the old partition (ditto the remark  */
        /* above).                                          */
        /****************************************************/
        last_receiver =
            p4est_bsearch_partition (src_end - 1, &dest_gfq[first_receiver],
                                    nproc - first_receiver) + first_receiver;
        ASSERT(first_receiver <= last_receiver && last_receiver < nproc,
               "Last receiver index out of bounds: "
               << last_receiver << " for nproc = " << nproc) ;
        

        /****************************************************/
        /* Set up the rank-specific quadrant range endpoint */
        /* as beginning.                                    */
        /****************************************************/
        gend = src_begin ; 

        /* Fetch vector of send requests */
        auto& rq = context._send_requests ;


        /* Quadrant offset */
        size_t rb = 0UL ; 
        

        /****************************************************/
        /* Loop over all senders and post send requests     */
        /****************************************************/
        for ( int q = first_receiver; q <= last_receiver; ++q) {

            /* Start at previous end     */
            gbegin = gend ; 

            /* End at next process begin */
            gend = dest_gfq[q+1] ;

            /* If this process goes till the end of the forest */
            /* it better be the last one                       */
            if (gend > src_end) {
                ASSERT( q==last_receiver,
                        "Last receiver index does not match the end of the "
                        "source data: " << gend << " != " << src_end) ;
                gend = src_end ; 
            }
            ASSERT(q == first_receiver || q == last_receiver ?
                    gbegin < gend : gbegin <= gend,
                    "Invalid range for receiver " << q) ;
            /* Process q is empty */       
            if ( gbegin == gend ) {
                ASSERT( first_receiver < q && q < last_receiver,
                        "Invalid range for receiver " << q) ; 
                //rq.push_back(sc_MPI_REQUEST_NULL) ; 
            } else { 
                /* Number of elements that need to be transferred */
                transfer_size = (size_t) (gend-gbegin) * data_stride ; 
                /* Are we sending to ourselves? */
                if ( q == rank ) {
                    /**************************************/
                    /* Yes, store data for later copy     */
                    /**************************************/
                    /* Length of copy was set before, we  */
                    /* just check that we agree on "send" */
                    /* and "receive" side.                */
                    /**************************************/
                    ASSERT(cp_len == (size_t) (gend-gbegin), 
                           "Source data pointer does not match the expected length: "
                           << cp_len << " != " << (size_t) (gend-gbegin)) ; 
                    /**************************************/
                    /* NB this is the QUADRANT offset     */
                    /**************************************/
                    src_cp_offset = rb ; 
                    /**************************************/
                    /*           Null request             */
                    /**************************************/
                    //rq.push_back(sc_MPI_REQUEST_NULL) ; 
                } else {
                    /**************************************/
                    /* No, post receive request           */
                    /**************************************/
                    rq.push_back(sc_MPI_Request{}) ; 
                    parallel::mpi_isend (
                        src.data() + rb * data_stride , transfer_size, q, tag, mpicomm, &(rq.back())
                    ) ;
                }
                /* New quad offset       */
                rb += (size_t) (gend-gbegin) ; 
            }

        }
    }   /* if ( src_begin < src_end ) */

    GRACE_TRACE("Posted {} send and {} receive requests.", context._send_requests.size(), context._recv_requests.size());
    /****************************************/
    if ( cp_len > 0 ) {
        auto src_view = Kokkos::subview(
            src, VEC( Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL() ), 
            Kokkos::ALL(), std::make_pair(src_cp_offset, src_cp_offset + cp_len) ) ;
        auto dest_view = Kokkos::subview(
            dest, VEC( Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL() ), 
            Kokkos::ALL(), std::make_pair(dest_cp_offset, dest_cp_offset + cp_len) ) ;
        Kokkos::deep_copy(default_execution_space{}, dest_view, src_view) ;
    } 

    return std::move(context) ;
}


void grace_transfer_fixed_end(
    parallel::grace_transfer_context_t& context
)
{
    GRACE_TRACE("Waiting for {} senders and {} receivers to complete.", context._recv_requests.size(), context._send_requests.size());
    parallel::mpi_waitall(context) ;
} // grace_transfer_fixed_end

}} /* namespace grace::amr */


#endif /*GRACE_AMR_REGRID_HELPERS_CPP*/