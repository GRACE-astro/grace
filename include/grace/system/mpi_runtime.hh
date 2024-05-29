#ifndef INCLUDE_GRACE_SYSTEM_MPI_RUNTIME
#define INCLUDE_GRACE_SYSTEM_MPI_RUNTIME

#include <grace_config.h>

#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/singleton_holder.hh> 
#include <grace/utils/creation_policies.hh>
#include <grace/utils/lifetime_tracker.hh>
#include <grace/utils/inline.h>

#include <grace/config/config_parser.hh>

namespace grace {
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Utility class that ensures MPI is initialized and finalized at appropriate times.
 * \ingroup system 
 */
class mpi_runtime_impl_t 
{
 private:
    int _master_rank ;     //!< The master rank is the one which is allowed to print to stdout 
 public:
    GRACE_ALWAYS_INLINE int master_rank() const { return _master_rank ; }
 private:
    //*****************************************************************************************************
    /**
     * @brief (Never) construct a new <code>mpi_runtime_impl_t</code> object
     */
    mpi_runtime_impl_t(int argc, char* argv[] ) {
        parallel::mpi_init(&argc, &argv) ; 
        auto& params = grace::config_parser::get() ; 
        _master_rank = params["system"]["master_rank"].as<int>() ; 
    }
    //*****************************************************************************************************
    /**
     * @brief (Never) destroy the <code>mpi_runtime_impl_t</code> object
     * 
     */
    ~mpi_runtime_impl_t() {
        parallel::mpi_finalize() ; 
    } 
    //*****************************************************************************************************
    friend class utils::singleton_holder<mpi_runtime_impl_t,memory::default_create> ;           //!< Give access
    friend class memory::new_delete_creator<mpi_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access
    //*****************************************************************************************************
    static constexpr size_t longevity = MPI_RUNTIME ; //!< Schedule destruction
    //*****************************************************************************************************
} ; 
//*****************************************************************************************************
/**
 * @brief Proxy for mpi runtime
 */
using mpi_runtime = utils::singleton_holder<mpi_runtime_impl_t,memory::default_create> ;
//*****************************************************************************************************
//*****************************************************************************************************
} /* namespace grace */

#endif /* INCLUDE_GRACE_SYSTEM_MPI_RUNTIME */
