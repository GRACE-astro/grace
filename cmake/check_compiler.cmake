# Handle HIP compiler
if("${CMAKE_CXX_COMPILER}" MATCHES "hipcc")
    message(STATUS "Compiling with hipcc.")
    add_compile_options(-fgpu-rdc)
    add_link_options(-fgpu-rdc --hip-link)
endif()

# Link filesystem library if necessary
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
    message(STATUS "Added -lstdc++fs to CMAKE_EXE_LINKER_FLAGS (GCC < 9.0).")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_LIBRARY STREQUAL "libstdc++" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
        message(STATUS "Added -lstdc++fs to CMAKE_EXE_LINKER_FLAGS (Clang with libstdc++ < 9.0).")
    endif()
endif()

# Check for Intel compiler
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(STATUS "Using Intel compiler.")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2021.7.0)
        message(FATAL_ERROR "Intel compiler version must be at least 2021.7.0.")
    endif()
    add_compile_options(-diag-disable=10441)
endif()

# Check for Nvidia HPC compiler
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
    message(STATUS "Using Nvidia HPC compiler.")
    add_compile_options(--expt-relaxed-constexpr -Xcudafe "--diag_suppress=20091")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
    message(STATUS "Added -lstdc++fs to CMAKE_EXE_LINKER_FLAGS for NVHPC.")
endif()
