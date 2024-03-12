#include <sstream>
#include <stdio.h>
#include <stdarg.h>

#include <thunder/errors/warn.hh>


void emit_warning_with_message(int warning_level, 
                               const char* file,
                               int line,
                               const char* function,
                               const char* fmt,
                               ... )
{
    std::ostringstream os ; 
    os << '\n'
       << "========== WARNING LEVEL " << warning_level << " ==========\n"
       << "File " << file << '\n'
       << "Line " << line << '\n'
       << "Function " << function << '\n'
       << fmt ; 

    va_list p; va_start(p, fmt);
    vfprintf(stderr, os.str().c_str(), p);
    va_end(p);
    
}