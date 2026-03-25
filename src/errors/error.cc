#include <grace/errors/error.hh>

#include <csignal>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

namespace detail {
//! \cond grace_detail
/**
 * @brief GRACE signal handler. Catches signals 
 *        and aborts with <code>ERROR</code>.
 * 
 * @param sg signal caught. 
 */
[[noreturn]] static void grace_signal_handler(int sig, siginfo_t* info, void*) {
    void* trace[64];
    int trace_size = backtrace(trace, 64);

    const char* sig_name = strsignal(sig);

    // Header
    write(STDERR_FILENO, "\n=== FATAL SIGNAL ===\n", 22);
    write(STDERR_FILENO, sig_name, strlen(sig_name));
    write(STDERR_FILENO, "\n", 1);

    // Fault address (if available)
    if (sig == SIGSEGV || sig == SIGBUS) {
        char buf[128];
        int len = snprintf(buf, sizeof(buf),
                           "Fault address: %p\n", info->si_addr);
        write(STDERR_FILENO, buf, len);
    }

    write(STDERR_FILENO, "Backtrace:\n", 11);

    // Print raw backtrace
    backtrace_symbols_fd(trace, trace_size, STDERR_FILENO);

    write(STDERR_FILENO, "====================\n", 21);

    _exit(128 + sig);
}
}
/**
 * Install signal handler with <code>sigaction</code>.
 */
void install_signal_handlers() {
    struct sigaction sa{};
    sa.sa_sigaction = detail::grace_signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;

    sigemptyset(&sa.sa_mask);

    const int fatal_signals[] = {
        SIGSEGV,
        SIGBUS,
        SIGFPE,
        SIGILL,
        SIGABRT
    };

    for (int s : fatal_signals) {
        sigaction(s, &sa, nullptr);
    }
}
//! \endcond