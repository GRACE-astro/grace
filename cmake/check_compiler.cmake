if("${CMAKE_CXX_COMPILER}" MATCHES "hipcc")
    message(STATUS "Compiling with hipcc")
    add_compile_options(-fgpu-rdc)
    add_link_options(-fgpu-rdc --hip-link)
    string(REPLACE "-fno-gpu-rdc" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if ("${CMAKE_CXX_COMPILER}" MATCHES "(mpiicpx|dpcpp)")
  message(STATUS "Compiling with ${CMAKE_CXX_COMPILER} (SYCL backend)")
  # should we add -fsycl-targets=spir64_gen ? it's the default I think
  add_compile_options(-fsycl -fsycl-rdc -fsycl-unnamed-lambda) # should we do the unnamed lambdas?
  add_link_options(-fsycl -fsycl-link) 
  string(REPLACE "-fno-sycl-rdc" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message("Using Intel compiler.")
    if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2021.7.0)
        message(FATAL_ERROR "Intel compiler version must be at least 2021.7.0")
    endif()
    add_compile_options(-diag-disable=10441)
endif()