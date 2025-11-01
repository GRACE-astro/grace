#include <grace/parallel/mpi_wrappers.hh>

namespace parallel {

void mpi_init(int* argc, char *** argv) 
{
    #ifndef SC_ENABLE_MPI
    WARN(1, 
         "MPI is currently not active. All MPI calls"
         "are being emulated in serial execution.");
    #endif

    int mpi_retval = sc_MPI_Init(argc, argv) ;
    /*ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_init failed.") ;*/
}

void mpi_finalize() 
{
    int mpi_retval = sc_MPI_Finalize() ;
    /*ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "Anomalous termination of MPI process"
            "mpi_finalize call failed.") ;*/
}

[[noreturn]] void 
mpi_abort(sc_MPI_Comm comm, int error_code) 
{
    sc_MPI_Abort(comm, error_code) ;
    // just so the compiler does not complain
    std::abort() ;
}

void
mpi_barrier(sc_MPI_Comm comm)
{
    int mpi_retval = sc_MPI_Barrier(comm) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_barrier call failed.") ;
}

int mpi_comm_size(sc_MPI_Comm comm)
{   
    int nProcs; 
    int mpi_retval = sc_MPI_Comm_size(comm, &nProcs) ;
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_comm_size call failed.") ;
    return nProcs ;
}

int mpi_comm_rank(sc_MPI_Comm comm)
{
    int rank;
    int mpi_retval = sc_MPI_Comm_rank(comm, &rank);
    ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_comm_rank call failed.") ;
    return rank ;
}

void mpi_waitall(grace_transfer_context_t& context)
{
    // mpi err code
    int mpi_retval; 
    if (context._recv_requests.size() > 0) {
        /* Wait for all receive requests to complete */
        mpi_retval = sc_MPI_Waitall(
            context._recv_requests.size(), 
            context._recv_requests.data(), 
            MPI_STATUSES_IGNORE
        ) ;
        ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_waitall call failed with status " << mpi_retval << ".") ;
    }
    if (context._send_requests.size() > 0) {
        /* Wait for all send requests to complete */
        mpi_retval = sc_MPI_Waitall(
            context._send_requests.size(), 
            context._send_requests.data(), 
            MPI_STATUSES_IGNORE
        ) ;
        ASSERT( mpi_retval == sc_MPI_SUCCESS, 
            "mpi_waitall call failed with status " << mpi_retval << ".") ;
    }
    context.reset() ; 
}

}

