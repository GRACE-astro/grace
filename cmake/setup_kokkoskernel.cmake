if (NOT KOKKOS_KERNELS_ROOT)
    set(KOKKOS_KERNELS_ROOT "$ENV{KOKKOS_KERNELS_ROOT}")
endif()
message(STATUS "KokkosKernels root: ${KOKKOS_KERNELS_ROOT}")

find_package(KokkosKernels REQUIRED
    PATHS
        ${KOKKOS_KERNELS_ROOT}/lib/cmake/KokkosKernels
        ${KOKKOS_KERNELS_ROOT}/lib64/cmake/KokkosKernels
)

# Debug what targets are available
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    if(_variableName MATCHES "KokkosKernels")
        message(STATUS "${_variableName}=${${_variableName}}")
    endif()
endforeach()

# Check what targets were created
if(TARGET KokkosKernels::kokkoskernels)
    message(STATUS "Found target: KokkosKernels::kokkoskernels")
elseif(TARGET KokkosKernels::KokkosKernels) 
    message(STATUS "Found target: KokkosKernels::KokkosKernels")
elseif(TARGET Kokkos::kokkoskernels)
    message(STATUS "Found target: Kokkos::kokkoskernels")
else()
    message(STATUS "No standard KokkosKernels target found")
endif()