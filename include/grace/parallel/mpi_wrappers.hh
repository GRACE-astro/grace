/**
 * @file mpi_wrappers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief add a thin c++ wrapper around mpi calls.
 * @version 0.1
 * @date 2023-03-01
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

#ifndef GRACE_MPI_WRAPPERS_HH
#define GRACE_MPI_WRAPPERS_HH

#include <grace_config.h>

#include <grace/errors/assert.hh>
#include <grace/utils/type_traits.hh>
/// This header contains
/// a `mock` MPI implementation
/// in case we don't actually link to
/// MPI. 
#include <sc.h>

#include <vector>

#ifdef GRACE_ENABLE_OLD_SC_VERS
#define sc_MPI_LONG_DOUBLE MPI_LONG_DOUBLE 
#else 
#define sc_MPI_UNSIGNED_LONG_LONG MPI_UNSIGNED_LONG_LONG
#endif 

#define mpi_comm sc_MPI_Comm
#define mpi_op sc_MPI_Op 

#define mpi_any_tag sc_MPI_ANY_TAG

#define mpi_comm_null sc_MPI_COMM_NULL
#define mpi_comm_self sc_MPI_COMM_SELF
#define mpi_comm_world sc_MPI_COMM_WORLD

#define mpi_sum sc_MPI_SUM
#define mpi_max sc_MPI_MAX 
#define mpi_min sc_MPI_MIN 
#define mpi_prod sc_MPI_PROD 
#define mpi_land sc_MPI_LAND 
#define mpi_band sc_MPI_BAND 
#define mpi_lor sc_MPI_LOR 
#define mpi_bor sc_MPI_BOR 
#define mpi_lxor sc_MPI_LXOR 
#define mpi_bxor sc_MPI_bxor 
#define mpi_minloc sc_MPI_MINLOC 
#define mpi_maxloc sc_MPI_MAXLOC
#define mpi_replace sc_MPI_REPLACE 

namespace parallel {

enum GRACE_MPI_Tags_t
{
    GRACE_PARTITION_TAG=0,
    GRACE_HALO_EXCHANGE_TAG,
    GRACE_N_MPI_TAGS 
} ; 

namespace detail {
    template< typename T >
    struct mpi_type_utils {
        using type=meta::no_such_type;
        static inline sc_MPI_Datatype get_type() ; 
    } ; 

    #define MPI_PRIMITIVE_TYPE(T, MPI_T) \
            template<> \
            inline sc_MPI_Datatype mpi_type_utils<T>::get_type() { \
            return MPI_T; \
            } 
    
    MPI_PRIMITIVE_TYPE(char,                  sc_MPI_CHAR) ; 
    MPI_PRIMITIVE_TYPE(short,                 sc_MPI_SHORT) ; 
    MPI_PRIMITIVE_TYPE(int,                   sc_MPI_INT) ; 
    MPI_PRIMITIVE_TYPE(long,                  sc_MPI_LONG) ; 
    MPI_PRIMITIVE_TYPE(unsigned,              sc_MPI_UNSIGNED) ; 
    MPI_PRIMITIVE_TYPE(unsigned long,         sc_MPI_UNSIGNED_LONG) ;
    MPI_PRIMITIVE_TYPE(long long,             sc_MPI_LONG_LONG_INT) ;
    MPI_PRIMITIVE_TYPE(unsigned long long,    sc_MPI_UNSIGNED_LONG_LONG) ;
    MPI_PRIMITIVE_TYPE(float,                 sc_MPI_FLOAT) ;
    MPI_PRIMITIVE_TYPE(double,                sc_MPI_DOUBLE) ;
    MPI_PRIMITIVE_TYPE(long double,           sc_MPI_LONG_DOUBLE) ;
    #undef MPI_PRIMITIVE_TYPE
}

void mpi_init(int* argc, char *** argv) ;

void mpi_finalize() ;

[[noreturn]] void mpi_abort(sc_MPI_Comm comm, int error_code) ;

int mpi_comm_size(sc_MPI_Comm comm=sc_MPI_COMM_WORLD) ; 

int mpi_comm_rank(sc_MPI_Comm comm=sc_MPI_COMM_WORLD) ; 

void mpi_barrier(sc_MPI_Comm comm=sc_MPI_COMM_WORLD) ;

sc_MPI_Comm inline get_comm_world() { return sc_MPI_COMM_WORLD ; }  

template<typename T>
static inline 
void mpi_gather(T* send_buffer,
                int send_count,
                T* recv_buffer,
                int recv_count,
                int root=0,
                sc_MPI_Comm comm = MPI_COMM_WORLD)
{ 
    int mpi_retval = sc_MPI_Gather( static_cast<void*>(send_buffer),
                                    send_count,
                                    detail::mpi_type_utils<T>::get_type(),
                                    static_cast<void*>(recv_buffer),
                                    recv_count, detail::mpi_type_utils<T>::get_type(),
                                    root, comm) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_gather call failed.") ; 
}

template<typename T>
static inline 
void mpi_gatherv(T* send_buffer,
                 int send_count,
                 T* recv_buffer,
                 int* recv_counts,
                 int* recv_offsets,
                 int root=0,
                 sc_MPI_Comm comm = MPI_COMM_WORLD) 
{

    int mpi_retval = sc_MPI_Gatherv(static_cast<void*>(send_buffer),
                                    send_count,
                                    detail::mpi_type_utils<T>::get_type(),
                                    recv_buffer,
                                    recv_counts,
                                    recv_offsets,
                                    detail::mpi_type_utils<T>::get_type(), 
                                    root,
                                    comm );
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_gatherv call failed.") ; 
}  

template<typename T>
static inline 
void mpi_allgather(T* send_buffer,
                   int send_count,
                   T* recv_buffer,
                   int recv_count,
                   sc_MPI_Comm comm = MPI_COMM_WORLD)
{
    int mpi_retval = sc_MPI_Allgather( static_cast<void*>(send_buffer),
                                       send_count,
                                       detail::mpi_type_utils<T>::get_type(),
                                       static_cast<void*>(recv_buffer),
                                       recv_count,
                                       detail::mpi_type_utils<T>::get_type(),
                                       comm ) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_allgather call failed.") ; 
}

template<typename T>
static inline 
void mpi_allgatherv(T* send_buffer,
                    int send_count,
                    T* recv_buffer,
                    int* recv_counts,
                    int* recv_offsets,
                    sc_MPI_Comm comm = MPI_COMM_WORLD)
{
    int mpi_retval = sc_MPI_Allgatherv( static_cast<void*>(send_buffer),
                                       send_count,
                                       detail::mpi_type_utils<T>::get_type(),
                                       static_cast<void*>(recv_buffer),
                                       recv_counts,
                                       recv_offsets,
                                       detail::mpi_type_utils<T>::get_type(),
                                       comm ) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_allgatherv call failed.") ; 
}

template<typename T>
static inline 
void mpi_alltoall(T* send_buffer,
                  int send_count,
                  T* recv_buffer,
                  int count_recv,
                  sc_MPI_Comm comm = MPI_COMM_WORLD)
{
    
    int mpi_retval = sc_MPI_Alltoall( static_cast<void*>(send_buffer),
                                       send_count,
                                       detail::mpi_type_utils<T>::get_type(),
                                       static_cast<void*>(recv_buffer),
                                       count_recv,
                                       detail::mpi_type_utils<T>::get_type(),
                                       comm ) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_alltoall call failed.") ; 
}


template<typename T>
static inline 
void mpi_bcast(T* send_buffer,
               int send_count,
               int root,
               sc_MPI_Comm comm = MPI_COMM_WORLD) 
{

    int mpi_retval = sc_MPI_Bcast( static_cast<void*>(send_buffer),
                                   send_count,
                                   detail::mpi_type_utils<T>::get_type(),
                                   root,
                                   comm ) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_bcast call failed.") ; 
}

template<typename T>
static inline 
void mpi_reduce(T* send_buffer,
                T* recv_buffer,
                int count,
                sc_MPI_Op op,
                int root=0,
                sc_MPI_Comm comm=MPI_COMM_WORLD) 
{
    ASSERT_DBG( (send_buffer != nullptr) or (count == 0), 
                "Trying to reduce more than zero"
                " data elements via dangling pointer.") ; 
    int mpi_retval = sc_MPI_Reduce( static_cast<void*>(send_buffer),
                                    static_cast<void*>(recv_buffer),
                                    count,
                                    detail::mpi_type_utils<T>::get_type(),
                                    op,
                                    root,
                                    comm ) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_reduce call failed.") ; 
}

template<typename T>
static inline 
void mpi_allreduce(T* send_buffer,
                   T* recv_buffer,
                   int count,
                   sc_MPI_Op op,
                   sc_MPI_Comm comm=MPI_COMM_WORLD)
{
    
    ASSERT_DBG( (send_buffer != nullptr) or (count == 0), 
                "Trying to reduce more than zero"
                " data elements via dangling pointer.");
    ASSERT_DBG( (recv_buffer != nullptr) or (count == 0), 
                "Trying to store allreduce result from "
                "more than zero data elements on dangling pointer.") ;
    int mpi_retval = sc_MPI_Allreduce(static_cast<void*>(send_buffer),
                                      static_cast<void*>(recv_buffer),
                                      count,
                                      detail::mpi_type_utils<T>::get_type(),
                                      op,
                                      comm ) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_reduce call failed.") ; 
}

template<typename T>
static inline 
void mpi_send(T* send_buffer, int size,
              int dest,
              int tag,
              sc_MPI_Comm comm=MPI_COMM_WORLD) 
{
    #ifndef SC_ENABLE_MPI
    ASSERT(0, 
           "Please make sure that a real MPI implementation "
           "is linked before attempting to call mpi_send "
           "(build sc with --enable-mpi)") ;
    #endif 
    ASSERT_DBG( (send_buffer!=nullptr) or (size==0), 
                "Attempting to mpi_send more than zero "
                "elements from a dangling pointer." ) ;
    int mpi_retval = sc_MPI_Send( static_cast<void*>(send_buffer),
                                  size,
                                  detail::mpi_type_utils<T>::get_type(),
                                  dest,
                                  tag,
                                  comm) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_send call failed.") ;
}

template<typename T>
static inline 
sc_MPI_Status mpi_recv(T* recv_buffer, int size,
              int source,
              int tag,
              sc_MPI_Comm comm=MPI_COMM_WORLD)
{
    #ifndef SC_ENABLE_MPI
    ASSERT(0, 
           "Please make sure that a real MPI implementation "
           "is linked before attempting to call mpi_recv "
           "(build sc with --enable-mpi)") ;
    #endif
    ASSERT_DBG( (recv_buffer!=nullptr) or (size==0), 
                "Attempting to mpi_recv more than zero "
                "elements into a dangling pointer." ) ;
    sc_MPI_Status status ;
    int mpi_retval = sc_MPI_Recv( static_cast<void*>(recv_buffer),
                                  size,
                                  detail::mpi_type_utils<T>::get_type(),
                                  source,
                                  tag,
                                  comm, &status) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_recv call failed.") ; 
    return status ;
}

template<typename T>
static inline 
void mpi_isend(T* send_buffer, int size,
               int dest,
               int tag,
               sc_MPI_Comm comm,
               sc_MPI_Request* request)
{
    #ifndef SC_ENABLE_MPI
    ASSERT(0, 
           "Please make sure that a real MPI implementation"
           "is linked before attempting to call mpi_isend"
           "(build sc with --enable-mpi)") ;
    #endif
    ASSERT_DBG( (send_buffer!=nullptr) or (size==0), 
                "Attempting to mpi_isend more than zero "
                "elements from a dangling pointer." ) ;
    int mpi_retval = sc_MPI_Isend( static_cast<void*>(send_buffer),
                                   size,
                                   detail::mpi_type_utils<T>::get_type(),
                                   dest,
                                   tag,
                                   comm, request) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_isend call failed.") ;
}

template<typename T>
static inline 
void mpi_irecv(T* recv_buffer, int size,
              int source,
              int tag,
              sc_MPI_Comm comm,
              sc_MPI_Request* request) 
{
    #ifndef SC_ENABLE_MPI
    ASSERT(0, 
           "Please make sure that a real MPI implementation"
           "is linked before attempting to call mpi_irecv"
           "(build sc with --enable-mpi)") ;
    #endif
    ASSERT_DBG( (recv_buffer!=nullptr) or (size==0), 
                "Attempting to mpi_irecv more than zero "
                "elements into a dangling pointer." ) ;
    int mpi_retval = sc_MPI_Irecv( static_cast<void*>(recv_buffer),
                                  size,
                                  detail::mpi_type_utils<T>::get_type(),
                                  source,
                                  tag,
                                  comm, request) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_irecv call failed.") ;
}

void mpi_waitall(std::vector<sc_MPI_Request>& requests);


}

#endif 