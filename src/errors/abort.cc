/**
 * @file abort.cc
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Implementation of <code>abort_with_error_message</code>.
 * @date 2023-03-14
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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
#include <array>
#include <cstdlib>
#include <execinfo.h>
// link.h is not available on all platforms
#if __has_include(<link.h>)
#include <link.h>
#endif
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thunder/errors/abort.hh>

//! \cond thunder_detail

//! utility for printing backtraces.
//! \ingroup system
struct backtrace_handler {} ;

#if __has_include(<link.h>)
/**
 * @brief demangle addresses coming from 
 *        loaded ELF shared objects.
 * @param addr address in the executable.
 * @return void* de-mangled address readable by <code>addr2line</code>.
 */
void* convert_to_virtual_memory_address(const void* addr) 
{
    Dl_info dlip; 
    link_map* _link_map = nullptr; 

    dladdr1(addr, &dlip,
           reinterpret_cast<void**>(&_link_map),
           RTLD_DL_LINKMAP) ;

    return reinterpret_cast<void*>(reinterpret_cast<std::size_t>(addr)-_link_map->l_addr) ;
}
#endif
/**
 * @brief Print a stack trace with a maximum depth of 13.
 */
std::ostream& operator<<(std::ostream& os, backtrace_handler const& /*unused*/)
{
    constexpr size_t max_stack_depth = 20; 

    std::array<void*, max_stack_depth> trace_elements{} ;

    int element_count = backtrace(trace_elements.data(), max_stack_depth); 

    std::unique_ptr<char*, decltype(free)*> stack_syms {
        backtrace_symbols(trace_elements.data(), element_count), free 
    } ;

    for( int i=3; i<element_count; ++i) {
        #if __has_include(<link.h>)
        Dl_info _info ; 
        const auto i_st = static_cast<size_t>(i) ;
        if(dladdr(trace_elements[i_st], &_info) != 0 ) {
            void* vma = convert_to_virtual_memory_address(trace_elements[i_st]) ; 
            os << stack_syms.get()[i] << " Address (to be read by addr2line): " << vma << '\n' ;
        } else {
            os << stack_syms.get()[i] << '\n' ;
        }
        #else // has link.h
        os << stack_syms.get()[i] << '\n' ;
        #endif // has link.h
    }
    return os ;
}
//! \endcond thunder_detail

/**
 * This function uses <code>backtrace_handler</code> to append
 * a stack trace to the user-provided error message.
 */
[[noreturn]] void abort_with_message( const char* expression,
                                      const char* file,
                                      int line,
                                      const char* function,
                                      const std::string& message  )
{
    std::ostringstream os ;
    os << '\n'
       << "========== ASSERTION FAILED ==========\n"
       << expression << " not fulfilled \n"
       << "File " << file << '\n'
       << "Line " << line << '\n'
       << "Function " << function << '\n'
       << "Stack trace follows:\n"
       << backtrace_handler{} << "\n"
       << message << "\n"
       << "======================================\n\n" ;
    throw std::runtime_error(os.str()) ; 
} 

/**
 * This function uses <code>backtrace_handler</code> to append
 * a stack trace to the user-provided error message.
 */
[[noreturn]] void abort_with_message( const char* file,
                                      int line,
                                      const char* function,
                                      const std::string& message  )
{
    std::ostringstream os ;
    os << '\n'
       << "========== FATAL ERROR ==========\n"
       << message << '\n'
       << "File " << file << '\n'
       << "Line " << line << '\n'
       << "Function " << function << '\n'
       << "Stack trace follows:\n"
       << backtrace_handler{} 
       << "=================================\n\n" ;
    throw std::runtime_error(os.str()) ; 
}
