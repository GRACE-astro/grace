#ifndef INCLUDE_GRACE_SYSTEM_P4EST_RUNTIME
#define INCLUDE_GRACE_SYSTEM_P4EST_RUNTIME

#include <grace/amr/p4est_headers.hh>
#include <grace_config.h>

#include <grace/utils/singleton_holder.hh> 
#include <grace/utils/creation_policies.hh>
#include <grace/utils/lifetime_tracker.hh> 
#include <grace/system/print.hh>
#include <spdlog/spdlog.h>
namespace grace {

namespace detail {
static void grace_sc_log_hijacker(FILE* log_stream,
                                    const char* filename,
                                    int lineno,
                                    int package,
                                    int category,
                                    int priority,
                                    const char* msg)
{
    GRACE_VERBOSE(msg) ;
}
}

class p4est_runtime_impl_t 
{
 private:
    
    p4est_runtime_impl_t() {
        p4est_init(detail::grace_sc_log_hijacker, SC_LP_DEFAULT) ; 
    }
    ~p4est_runtime_impl_t() { } 

    friend class utils::singleton_holder<p4est_runtime_impl_t,memory::default_create> ; //!< Give access
    friend class memory::new_delete_creator<p4est_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access

    static constexpr size_t longevity = P4EST_RUNTIME ; 

} ; 

using p4est_runtime = utils::singleton_holder<p4est_runtime_impl_t,memory::default_create> ;

} /* namespace grace */

#endif /* INCLUDE_p4est_SYSTEM_P4EST_RUNTIME */
