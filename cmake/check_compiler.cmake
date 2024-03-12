
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message("Using Intel compiler.")
    if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2021.7.0)
        message(FATAL_ERROR "Intel compiler version must be at least 2021.7.0")
    endif()
    add_compile_options(-diag-disable=10441)
endif()