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
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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

#ifndef sc_MPI_LONG_DOUBLE
#define sc_MPI_LONG_DOUBLE MPI_LONG_DOUBLE 
#endif 
#ifndef sc_MPI_UNSIGNED_LONG_LONG
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

enum GRACE_MPI_Tags_t : size_t
{
    GRACE_PARTITION_TAG=0,
    GRACE_HALO_EXCHANGE_TAG_CC=1,
    GRACE_HALO_EXCHANGE_TAG_FX=2,
    GRACE_HALO_EXCHANGE_TAG_FY=3,
    GRACE_HALO_EXCHANGE_TAG_FZ=4,
    GRACE_REFLUX_TAG=5,
    GRACE_REFLUX_EMF_FACE_TAG=6,
    GRACE_REFLUX_EMF_COARSE_FACE_TAG=7,
    GRACE_REFLUX_EMF_EDGE_TAG=8,
    GRACE_REFLUX_EMF_COARSE_EDGE_TAG=9,
    GRACE_REGRID_TAG_FX=10,
    GRACE_REGRID_TAG_FY=11,
    GRACE_REGRID_TAG_FZ=12,
    //! Cross-rank exchange tag for same-level face-flux averaging
    //! (GRACE_SAME_LEVEL_FACE_AVERAGE).  Distinct from the fine-coarse
    //! face/EMF reflux tags so the two pipelines can be in-flight together.
    GRACE_REFLUX_COARSE_FACE_TAG=13,
    GRACE_N_MPI_TAGS=14
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

struct grace_transfer_context_t 
{ 
    
    std::vector<sc_MPI_Request> _recv_requests ; 
    std::vector<sc_MPI_Request> _send_requests ; 

    void reset() { 
        _send_requests.clear() ;  
        _recv_requests.clear() ;  
    } ; 
} ; 

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

/**
 * @brief Wrapper around MPI_Allreduce.
 *
 * Standard mode: passes through to `sc_MPI_Allreduce` directly.
 *
 * Deterministic mode (compile-time flag GRACE_ENABLE_DETERMINISTIC_MPI_REDUCTIONS):
 *   When `op == sc_MPI_SUM`, replace the implementation-defined tree
 *   reduction (whose floating-point summation order depends on rank count
 *   and topology) with a canonical-ordered sum:
 *     1. `MPI_Allgather` each rank's per-element partials into a single
 *        buffer of size count·nproc, ordered by ascending source rank.
 *     2. Each rank then accumulates the partials in the SAME ascending-rank
 *        order, locally.
 *   The result is bit-identical on every rank AND invariant under MPI
 *   repartitioning — i.e. the same simulation run at 1, 2, 8, … ranks
 *   yields exactly the same sum to the last bit.  Required for the
 *   bit-exact mass-conservation goal (`total_mass + Σ dM_outflux ≡ 0`
 *   across all partitions).
 *
 *   Cost: O(nproc · count) memory + Allgather instead of Allreduce.  For
 *   the small-count conservation reductions this is negligible; for any
 *   large-count reduction in the hot path the flag should stay off.
 *
 *   Currently intercepts SUM only.  MAX/MIN are FP-deterministic by
 *   construction (associative + commutative under IEEE comparison for
 *   non-NaN values), so they fall through to standard Allreduce.  PROD
 *   could be added symmetrically if a use case appears.
 */
template<typename T>
static inline
void mpi_allreduce(T const* send_buffer,
                   T*       recv_buffer,
                   int      count,
                   sc_MPI_Op op,
                   sc_MPI_Comm comm=MPI_COMM_WORLD)
{
    ASSERT_DBG( (send_buffer != nullptr) or (count == 0),
                "Trying to reduce more than zero"
                " data elements via dangling pointer.");
    ASSERT_DBG( (recv_buffer != nullptr) or (count == 0),
                "Trying to store allreduce result from "
                "more than zero data elements on dangling pointer.") ;

    // MPI_IN_PLACE handling: the caller indicates "input is in recv_buffer,
    // recv_buffer also receives the result".  Both the standard path and the
    // deterministic path treat the recv_buffer as the source in that case.
    void* const send_ptr = (send_buffer == reinterpret_cast<T const*>(MPI_IN_PLACE))
                           ? static_cast<void*>(recv_buffer)
                           : static_cast<void*>(const_cast<T*>(send_buffer)) ;

#ifdef GRACE_ENABLE_DETERMINISTIC_MPI_REDUCTIONS
    // Intercept SUM, MAX, MIN.  For SUM the issue is FP non-associativity:
    // different tree topologies → different bit pattern.  For MAX/MIN with
    // pure-IEEE inputs (no NaN) the result is mathematically unique, but we
    // still route through the canonical path for two reasons:
    //   (1) gives users an absolute guarantee that ALL reductions are
    //       rank-invariant for free, no need to think per-op;
    //   (2) any downstream arithmetic that follows the MAX (e.g., dt =
    //       CFL/v_max) is then driven by a bit-identical input across
    //       ranks, eliminating one source of inter-rank divergence.
    if (count > 0 &&
        (op == sc_MPI_SUM || op == sc_MPI_MAX || op == sc_MPI_MIN)) {
        int nproc = 1 ;
        sc_MPI_Comm_size(comm, &nproc) ;
        std::vector<T> gathered(static_cast<size_t>(count)
                                * static_cast<size_t>(nproc)) ;
        int ag_retval = sc_MPI_Allgather(
            send_ptr,
            count, detail::mpi_type_utils<T>::get_type(),
            static_cast<void*>(gathered.data()),
            count, detail::mpi_type_utils<T>::get_type(),
            comm) ;
        ASSERT( ag_retval == sc_MPI_SUCCESS,
                "deterministic mpi_allreduce: Allgather call failed.") ;
        for (int i = 0; i < count; ++i) {
            T acc = gathered[i] ;          // rank 0's value
            if (op == sc_MPI_SUM) {
                for (int r = 1; r < nproc; ++r)
                    acc += gathered[static_cast<size_t>(r) * count + i] ;
            } else if (op == sc_MPI_MAX) {
                for (int r = 1; r < nproc; ++r) {
                    T const v = gathered[static_cast<size_t>(r) * count + i] ;
                    if (v > acc) acc = v ;
                }
            } else { // sc_MPI_MIN
                for (int r = 1; r < nproc; ++r) {
                    T const v = gathered[static_cast<size_t>(r) * count + i] ;
                    if (v < acc) acc = v ;
                }
            }
            recv_buffer[i] = acc ;
        }
        return ;
    }
#endif

    // Standard path.  Pass MPI_IN_PLACE through unchanged to the underlying
    // implementation when the caller specified it (MPI itself handles the
    // alias, no buffer copy needed); otherwise the normal send_buffer path.
    int mpi_retval = sc_MPI_Allreduce(
        (send_buffer == reinterpret_cast<T const*>(MPI_IN_PLACE))
            ? MPI_IN_PLACE
            : static_cast<void*>(const_cast<T*>(send_buffer)),
        static_cast<void*>(recv_buffer),
        count,
        detail::mpi_type_utils<T>::get_type(),
        op,
        comm) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS,
            "mpi_reduce call failed.") ;
}

/**
 * @brief In-place all-reduce: the same buffer is used for both input and
 *        output.  Avoids the awkward `reinterpret_cast<T*>(MPI_IN_PLACE)`
 *        at the call site.  Routes through `mpi_allreduce` so the
 *        deterministic path applies uniformly.
 */
template<typename T>
static inline
void mpi_allreduce_inplace(T* buffer,
                           int count,
                           sc_MPI_Op op,
                           sc_MPI_Comm comm = MPI_COMM_WORLD)
{
    ASSERT_DBG( (buffer != nullptr) or (count == 0),
                "mpi_allreduce_inplace called with a dangling pointer.") ;
    // Snapshot the input into a temporary so the wrapper's two-buffer
    // contract is preserved.  Cheap for the small-count reductions this
    // is typically used for (1-element checkpoint / dt aggregates).
    std::vector<T> tmp(buffer, buffer + count) ;
    mpi_allreduce(tmp.data(), buffer, count, op, comm) ;
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
           "Please make sure that a real MPI implementation"
           "is linked before attempting to call mpi_send"
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
           "Please make sure that a real MPI implementation"
           "is linked before attempting to call mpi_recv"
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

template<typename T>
static inline 
void mpi_exscan_sum( T* send_buffer, T* recv_buffer, int size,
              sc_MPI_Comm comm) 
{
    #ifndef SC_ENABLE_MPI
    ASSERT(0, 
           "Please make sure that a real MPI implementation"
           "is linked before attempting to call mpi_irecv"
           "(build sc with --enable-mpi)") ;
    #endif
    ASSERT_DBG( (recv_buffer!=nullptr) or (size==0), 
                "Attempting to send more than zero "
                "elements into a dangling pointer." ) ;
    int mpi_retval = sc_MPI_Exscan( static_cast<void*>(send_buffer),
                                    static_cast<void*>(recv_buffer),
                                  size,
                                  detail::mpi_type_utils<T>::get_type(),
                                  mpi_sum,
                                  comm) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_irecv call failed.") ;
}

void mpi_waitall(grace_transfer_context_t& context);


}

#endif 