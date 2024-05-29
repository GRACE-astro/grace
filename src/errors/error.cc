#include <grace/errors/error.hh>

#include<csignal>
#include<signal.h>

namespace detail {
//! \cond grace_detail
/**
 * @brief Magma signal handler. Catches signals 
 *        and aborts with <code>ERROR</code>.
 * 
 * @param sg signal caught. 
 */
[[noreturn]] void grace_signal_handler(int sg){
    switch (sg) {

        case SIGFPE:
            ERROR("Invalid floating point operation!") ;
            break;
        case SIGBUS:
            ERROR("Invalid memory access!") ;
            break;
        case SIGSEGV:
            ERROR("Segmentation fault.");
            break;
        case SIGINT:
            ERROR("Interrup signal caught.") ;
            break ;
        case SIGTERM:
            ERROR("Termination signal caught.") ;
            break ; 
        default:
            ERROR("Unknown signal caught.") ;
            break ;
    }

    std::abort() ; 
}
}
/**
 * Install signal handler with <code>sigaction</code>.
 */
void install_signal_handlers() {
    struct sigaction signal_handler ;
    signal_handler.sa_handler = detail::grace_signal_handler;

    if ( sigaction(SIGBUS, &signal_handler, nullptr) < 0 )
    {
        ERROR("Unable to install signal handler for SIGBUS.") ;
    } else if ( sigaction(SIGFPE, &signal_handler, nullptr) < 0 )
    {
        ERROR("Unable to install signal handler for SIGFPE.") ;
    } else if ( sigaction(SIGSEGV, &signal_handler, nullptr) < 0 )
    {
        ERROR("Unable to install signal handler for SIGSEGV.") ;
    } else if ( sigaction(SIGINT, &signal_handler, nullptr) < 0 )
    {
        ERROR("Unable to install signal handler for SIGINT.") ;
    } else if ( sigaction(SIGTERM, &signal_handler, nullptr) < 0 )
    {
        ERROR("Unable to install signal handler for SIGTERM.") ;
    }
}
//! \endcond