# Kokkos dependency resolution.
#
#   GRACE_USE_BUNDLED_DEPS = ON  →  build extern/kokkos in-tree.  Inherits any
#                                   `Kokkos_ENABLE_*` cache vars set on the
#                                   GRACE configure line (-DKokkos_ENABLE_CUDA=ON,
#                                   -DKokkos_ARCH_HOPPER90=ON, etc.).
#   GRACE_USE_BUNDLED_DEPS = OFF →  system install (find_package, honours
#                                   KOKKOS_ROOT env var).
if(GRACE_USE_BUNDLED_DEPS)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/kokkos/CMakeLists.txt")
        message(FATAL_ERROR
            "GRACE_USE_BUNDLED_DEPS=ON but extern/kokkos is empty.  Run:\n"
            "  git submodule update --init --recursive extern/kokkos")
    endif()
    # UX convenience: mirror GRACE_ENABLE_<BACKEND> into the matching
    # `Kokkos_ENABLE_*` cache var if the user hasn't set it explicitly.
    # That way `-DGRACE_ENABLE_CUDA=ON` alone is sufficient for the bundled
    # build; the user only needs to add `-DKokkos_ARCH_*=ON` for the GPU
    # arch (which GRACE can't infer).
    macro(_grace_mirror_kokkos_backend GRACE_VAR KOKKOS_VAR)
        if(${GRACE_VAR} AND NOT DEFINED ${KOKKOS_VAR})
            set(${KOKKOS_VAR} ON CACHE BOOL "auto-mirrored from ${GRACE_VAR}" FORCE)
            message(STATUS "Auto-set ${KOKKOS_VAR}=ON to match ${GRACE_VAR}")
        endif()
    endmacro()
    _grace_mirror_kokkos_backend(GRACE_ENABLE_CUDA   Kokkos_ENABLE_CUDA)
    _grace_mirror_kokkos_backend(GRACE_ENABLE_HIP    Kokkos_ENABLE_HIP)
    _grace_mirror_kokkos_backend(GRACE_ENABLE_SYCL   Kokkos_ENABLE_SYCL)
    _grace_mirror_kokkos_backend(GRACE_ENABLE_OMP    Kokkos_ENABLE_OPENMP)
    _grace_mirror_kokkos_backend(GRACE_ENABLE_SERIAL Kokkos_ENABLE_SERIAL)
    # Relocatable device code (RDC) is required by GRACE — several GRACE
    # device functions (e.g. conservs_to_prims in src/physics/eos/c2p.cpp)
    # are defined in .cpp and called from kernels in other translation
    # units, which only links correctly when the active GPU backend's RDC
    # is enabled.  Mirror the corresponding Kokkos_ENABLE_*_RDC into the
    # cache for the bundled-Kokkos build.
    _grace_mirror_kokkos_backend(GRACE_ENABLE_CUDA Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
    _grace_mirror_kokkos_backend(GRACE_ENABLE_HIP  Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
    _grace_mirror_kokkos_backend(GRACE_ENABLE_SYCL Kokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/extern/kokkos)
else()
    if (NOT KOKKOS_ROOT)
        set(KOKKOS_ROOT "")
        set(KOKKOS_ROOT "$ENV{KOKKOS_ROOT}")
    endif()
    message(STATUS "Kokkos root ${KOKKOS_ROOT}")
    find_package(Kokkos REQUIRED PATHS ${KOKKOS_ROOT}/lib/cmake/Kokkos ${KOKKOS_ROOT}/lib64/cmake/Kokkos)
endif()

option(GRACE_ENABLE_CUDA  "Enable CUDA device support" OFF) 
option(GRACE_ENABLE_HIP   "Enable HIP device support"  OFF)
option(GRACE_ENABLE_SYCL  "Enable SYCL device support"  OFF)
option(GRACE_ENABLE_OMP   "Enable OpenMP threading support" OFF)
option(GRACE_ENABLE_SERIAL "Enable serial execution of ParallelLoops" OFF)

if( (NOT GRACE_ENABLE_CUDA)  AND (NOT GRACE_ENABLE_HIP) AND (NOT GRACE_ENABLE_SYCL) AND (NOT GRACE_ENABLE_OMP) )
    message(STATUS "No backend selectend, enabling serial execution")
    set(GRACE_ENABLE_SERIAL ON)
endif() 

if( GRACE_ENABLE_CUDA AND (NOT Kokkos_ENABLE_CUDA))
    message(FATAL_ERROR "GRACE configured with CUDA support but Kokkos does not support CUDA backend.")
endif()

if( GRACE_ENABLE_HIP AND (NOT Kokkos_ENABLE_HIP))
    message(FATAL_ERROR "GRACE configured with HIP support but Kokkos does not support HIP backend.")
endif()

if( GRACE_ENABLE_SYCL AND (NOT Kokkos_ENABLE_SYCL))
    message(FATAL_ERROR "GRACE configured with SYCL support but Kokkos does not support SYCL backend.")
endif()
if( GRACE_ENABLE_OMP AND (NOT Kokkos_ENABLE_OPENMP))
    message(FATAL_ERROR "GRACE configured with OMP support but Kokkos does not support OMP backend.")
endif()

if( GRACE_ENABLE_SERIAL AND (NOT Kokkos_ENABLE_SERIAL))
    message(FATAL_ERROR "GRACE configured with SERIAL support but Kokkos does not support SERIAL backend.")
endif()

if( GRACE_ENABLE_OMP )
    find_package(OpenMP REQUIRED)
endif()

if ( GRACE_ENABLE_CUDA )
    # `--expt-relaxed-constexpr` is an nvcc-only flag; clang-CUDA allows
    # constexpr host/device calls without it (and rejects the flag with
    # an error).  Only add it when the C++ compiler is *not* clang.
    if ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
        message(STATUS "clang-CUDA detected; skipping --expt-relaxed-constexpr (clang allows it by default)")
    else()
        add_compile_options(--expt-relaxed-constexpr)
    endif()
endif()
