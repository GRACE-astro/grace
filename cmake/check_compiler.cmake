if("${CMAKE_CXX_COMPILER}" MATCHES "hipcc")
    message(STATUS "Compiling with hipcc")
    # Gate -fgpu-rdc to CXX compilation only: AMD Cray toolchains (and similar
    # wrappers) drive C compilation through a `cc` frontend that does not
    # understand the flag, even when CXX is hipcc.  The generator expression
    # keeps the flag off the C compile lines.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fgpu-rdc>)
    add_link_options(-fgpu-rdc --hip-link)
    string(REPLACE "-fno-gpu-rdc" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if ("${CMAKE_CXX_COMPILER}" MATCHES "(mpiicpx|dpcpp)")
  message(STATUS "Compiling with ${CMAKE_CXX_COMPILER} (SYCL backend)")
  # Same CXX-only gating as the HIP block: keep SYCL flags off C compile
  # lines so toolchains that wrap C through a non-SYCL frontend don't choke.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsycl>
                      $<$<COMPILE_LANGUAGE:CXX>:-fsycl-rdc>
                      $<$<COMPILE_LANGUAGE:CXX>:-fsycl-unnamed-lambda>)
  add_link_options(-fsycl)
  string(REPLACE "-fno-sycl-rdc" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message("Using Intel compiler.")
    if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2021.7.0)
        message(FATAL_ERROR "Intel compiler version must be at least 2021.7.0")
    endif()
    add_compile_options(-diag-disable=10441)
endif()
