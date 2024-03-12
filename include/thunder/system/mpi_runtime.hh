#ifndef INCLUDE_THUNDER_SYSTEM_MPI_RUNTIME
#define INCLUDE_THUNDER_SYSTEM_MPI_RUNTIME

#include <thunder_config.h>

#include <thunder/parallel/mpi_wrappers.hh>
#include <thunder/utils/singleton_holder.hh> 
#include <thunder/utils/creation_policies.hh>
#include <thunder/utils/lifetime_tracker.hh> 

namespace thunder {

class mpi_runtime_impl_t 
{
 private:
    
    mpi_runtime_impl_t(int argc, char* argv[] ) {
        parallel::mpi_init(&argc, &argv) ; 
    }
    ~mpi_runtime_impl_t() {
        parallel::mpi_finalize() ; 
    } 

    friend class utils::singleton_holder<mpi_runtime_impl_t,memory::default_create> ; 
    friend class memory::new_delete_creator<mpi_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access

    static constexpr size_t longevity = MPI_RUNTIME ; 

} ; 

using mpi_runtime = utils::singleton_holder<mpi_runtime_impl_t,memory::default_create> ;

} /* namespace thunder */

#endif /* INCLUDE_THUNDER_SYSTEM_MPI_RUNTIME */
