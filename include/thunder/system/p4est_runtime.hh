#ifndef INCLUDE_THUNDER_SYSTEM_P4EST_RUNTIME
#define INCLUDE_THUNDER_SYSTEM_P4EST_RUNTIME

#include <thunder/amr/p4est_headers.hh>
#include <thunder_config.h>

#include <thunder/utils/singleton_holder.hh> 
#include <thunder/utils/creation_policies.hh>
#include <thunder/utils/lifetime_tracker.hh> 
#include <thunder/system/print.hh>
#include <spdlog/spdlog.h>
namespace thunder {

namespace detail {
static void thunder_sc_log_hijacker(FILE* log_stream,
                                    const char* filename,
                                    int lineno,
                                    int package,
                                    int category,
                                    int priority,
                                    const char* msg)
{
    THUNDER_VERBOSE(msg) ;
}
}

class p4est_runtime_impl_t 
{
 private:
    
    p4est_runtime_impl_t() {
        p4est_init(detail::thunder_sc_log_hijacker, SC_LP_DEFAULT) ; 
    }
    ~p4est_runtime_impl_t() { } 

    friend class utils::singleton_holder<p4est_runtime_impl_t,memory::default_create> ; //!< Give access
    friend class memory::new_delete_creator<p4est_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access

    static constexpr size_t longevity = P4EST_RUNTIME ; 

} ; 

using p4est_runtime = utils::singleton_holder<p4est_runtime_impl_t,memory::default_create> ;

} /* namespace thunder */

#endif /* INCLUDE_p4est_SYSTEM_P4EST_RUNTIME */
