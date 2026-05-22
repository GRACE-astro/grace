
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
#cmake_policy(SET CMP0144 NEW)

project(GRACE VERSION 0.5 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(code_modules)
include(check_compiler)
include(prohibit_in_source_build)
include(setup_flags)

# --- Options ---
# Metric-evolution scheme (compile-time).
#   COWLING — frozen background metric (no spacetime evolution).
#   Z4       — Z4c evolution of the spacetime metric.
set(GRACE_METRIC_EVOL "COWLING" CACHE STRING
    "Metric evolution scheme (COWLING|Z4)")
set_property(CACHE GRACE_METRIC_EVOL PROPERTY STRINGS COWLING Z4)
if(NOT GRACE_METRIC_EVOL MATCHES "^(COWLING|Z4)$")
    message(FATAL_ERROR
        "GRACE_METRIC_EVOL=${GRACE_METRIC_EVOL} is not one of COWLING, Z4.")
endif()
message(STATUS "Metric evolution: ${GRACE_METRIC_EVOL}")

# FD truncation orders (compile-time).
#
# GRACE_Z4C_DER_ORDER controls the truncation order of every Z4c RHS
# finite-difference operator (centered 1st, biased L1/R1, centered 2nd
# diag/mixed, upwind 1st, Kreiss-Oliger).  Supported orders: {2, 4, 6}.
# All operators are emitted by ~/grace_notebooks/Z4c/fd_emit.py with
# symmetry-equivariant grouping baked into the generated C99 expressions;
# extending beyond order 6 requires re-running the codegen with a larger
# ORDERS tuple.
#
# GRACE_MATTER_METRIC_DER_ORDER controls the FD order used by matter
# subsystems (GRMHD, M1) when sampling spatial derivatives of the metric
# for geometric source terms.  Defaults to GRACE_Z4C_DER_ORDER so the
# matter and metric live on the same dispersion curve at all k — this is
# the structural cure for the order-mismatch wake artefact diagnosed in
# the May-2026 BNS ripple discussion (centered FD on Z4c at order 6 plus
# matter sources at order 4 produces a coherent f-mode ringing at the NS
# surface that survives reconstruction-limiter changes).  Override only
# with cause.
set(GRACE_Z4C_DER_ORDER "6" CACHE STRING
    "Z4c FD truncation order (2, 4, or 6)")
set_property(CACHE GRACE_Z4C_DER_ORDER PROPERTY STRINGS 2 4 6)
if(NOT GRACE_Z4C_DER_ORDER MATCHES "^(2|4|6)$")
    message(FATAL_ERROR
        "GRACE_Z4C_DER_ORDER=${GRACE_Z4C_DER_ORDER} is not one of {2, 4, 6}.")
endif()

set(GRACE_MATTER_METRIC_DER_ORDER "${GRACE_Z4C_DER_ORDER}" CACHE STRING
    "FD order for matter-side metric derivatives (2, 4, or 6; defaults to GRACE_Z4C_DER_ORDER)")
set_property(CACHE GRACE_MATTER_METRIC_DER_ORDER PROPERTY STRINGS 2 4 6)
if(NOT GRACE_MATTER_METRIC_DER_ORDER MATCHES "^(2|4|6)$")
    message(FATAL_ERROR
        "GRACE_MATTER_METRIC_DER_ORDER=${GRACE_MATTER_METRIC_DER_ORDER} is not one of {2, 4, 6}.")
endif()
if(NOT GRACE_MATTER_METRIC_DER_ORDER STREQUAL GRACE_Z4C_DER_ORDER)
    message(WARNING
        "GRACE_MATTER_METRIC_DER_ORDER (${GRACE_MATTER_METRIC_DER_ORDER}) differs "
        "from GRACE_Z4C_DER_ORDER (${GRACE_Z4C_DER_ORDER}). "
        "This introduces an order mismatch in matter<->metric coupling. "
        "Expect a small but coherent ringing artefact at NS surfaces in BNS "
        "configurations (see May-2026 wake-ripple analysis).  Only set this "
        "deliberately for diagnostic A/B comparisons.")
endif()
message(STATUS "Z4c FD order              : ${GRACE_Z4C_DER_ORDER}")
message(STATUS "Matter metric-deriv order : ${GRACE_MATTER_METRIC_DER_ORDER}")

set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")

option(GRACE_BUILD_DOCS "Build the docs" OFF)
option(GRACE_BUILD_DOCS_ONLY "Only build docs" OFF)
if(GRACE_BUILD_DOCS OR GRACE_BUILD_DOCS_ONLY ) 
    add_subdirectory(doc)
endif()

if( NOT GRACE_BUILD_DOCS_ONLY )

# -----------------------------------------------------------------------------
# Bundled dependencies
# -----------------------------------------------------------------------------
# When ON, build Kokkos / p4est / Catch2 / spdlog / yaml-cpp from the in-tree
# git submodules under `extern/` instead of linking against system installs.
# Recommended for laptop / CI / first-time-contributor builds — gives a
# `git clone --recursive && cmake && make` experience.  Cluster builds should
# leave OFF and rely on arch-tuned `module load` paths (Kokkos in particular
# wants per-cluster CUDA/HIP arch flags that we can't bake in here).
# -----------------------------------------------------------------------------
option(GRACE_USE_BUNDLED_DEPS
       "Build Kokkos/p4est/Catch2/spdlog/yaml-cpp from in-tree extern/ submodules"
       OFF)
if(GRACE_USE_BUNDLED_DEPS)
    message(STATUS "Bundled-dependency mode: deps will build from extern/ submodules")
endif()

include(setup_mpi)
find_package(OpenMP REQUIRED)
include(setup_p4est)
# P4_TO_P8 is added HERE, AFTER the (potentially-bundled) p4est subdirectory
# is consumed, so its 2D source files compile cleanly.  Targets registered
# after this point (GRACE's own libraries + tests) will pick up the
# definition for the 3D p8est headers they consume.  See the matching note
# in cmake/setup_flags.cmake.
#
# Note: defined with an EMPTY replacement (`=`) rather than the implicit `=1`
# that `add_compile_definitions(P4_TO_P8)` would emit.  p4est's own
# `p4est_to_p8est.h` does `#define P4_TO_P8` (no value); without the empty-
# replacement match here, the two #defines disagree and GCC emits a
# `-Wredefined` warning at every translation unit that includes a p8est
# header.
if(GRACE_3D)
    add_compile_definitions(P4_TO_P8=)
endif()
find_package(LibXml2 REQUIRED)
find_package(ZLIB REQUIRED)
include(setup_kokkos)
include(setup_yaml)
include(setup_spdlog)
include(setup_HDF5)

option( GRACE_ENABLE_VTK "Enable VTK support" OFF)
if ( GRACE_ENABLE_VTK )
    message(STATUS "VTK support enabled")
    include(setup_vtk)
endif() 

option( GRACE_ENABLE_LORENE "Lorene support" OFF )
if ( GRACE_ENABLE_LORENE )
    message(STATUS "LORENE support enabled")
    find_package(LORENE REQUIRED)
endif() 

option( GRACE_ENABLE_TWO_PUNCTURES "TwoPunctures support" OFF )
if ( GRACE_ENABLE_TWO_PUNCTURES )
    message(STATUS "TwoPunctures support enabled")
    find_package(TwoPunctures REQUIRED)
endif() 

option( GRACE_ENABLE_FUKA "FUKA support" OFF )

if( GRACE_ENABLE_FUKA )
    # This package contains its dependencies 
    # NB: HOME_KADATH must be set somewhere for this
    # to be found 
    find_package(Kadath REQUIRED)
endif()

option(GRACE_ENABLE_PROFILING "Build profiling hooks" OFF)
if( GRACE_ENABLE_PROFILING )
    message(STATUS "Profiling enabled.")
    if( GRACE_ENABLE_HIP )
        include(setup_rocmprofiler)
    elseif( GRACE_ENABLE_CUDA )
        message(FATAL_ERROR "CUDA GPU profiling is not supported yet.")
    endif()
else()
    message(STATUS "Profiling disabled.")
endif()

option(GRACE_ENABLE_PARTICLES "Build the GRACE particle subsystem (tracers, MC, PIC)" OFF)
if( GRACE_ENABLE_PARTICLES )
    message(STATUS "Particle subsystem enabled (Kokkos+MPI only — no external particle library)")
endif()

# Deterministic MPI reductions: intercept mpi_allreduce(SUM) and replace
# with an MPI_Allgather + canonical-ordered local sum.  Result is bit-
# identical across all ranks and reproducible across partitions.  Required
# for the bit-exact mass-conservation goal.  Costs an Allgather + O(nproc)
# memory per reduction; negligible for the small-count diagnostic sums it
# targets.  Leave OFF by default — performance impact on the hot path
# (if any large-count Allreduce sits there) would be real.
option(GRACE_ENABLE_DETERMINISTIC_MPI_REDUCTIONS
       "Use canonical-ordered (rank-invariant, bit-deterministic) sums in mpi_allreduce(SUM)"
       OFF)
if( GRACE_ENABLE_DETERMINISTIC_MPI_REDUCTIONS )
    message(STATUS "Deterministic MPI sum reductions enabled — bit-exact across partitions.")
endif()

# Same-level flux averaging.  After each substep's reflux at fine-coarse
# interfaces, ALSO exchange and average face fluxes at same-level interior
# faces so both sides apply the SAME bit-pattern.  Combined with the existing
# fine-coarse reflux and deterministic MPI sums, this gives the bit-exact
# mass-conservation property:  total_mass(t) + Σ dM_outflux(t) ≡ M(0).
# Eliminates the linear-in-time FMA-roundoff drift that would otherwise
# accumulate (~ulp · ⟨|F|⟩ · A_total · N_step relative).  Costs are similar
# in scale to the existing reflux machinery: an extra send/recv per pair of
# same-level interior faces.  Leave OFF by default — opt in for runs where
# bit-exact conservation matters.
option(GRACE_SAME_LEVEL_FLUX_AVERAGE
       "Average face fluxes at same-level interior faces after reflux — bit-exact mass conservation"
       OFF)
# Internal define keeps the historical `_FACE_AVERAGE` name so the in-source
# `#ifdef`s in refluxing.cpp / amr_ghosts_construct_reflux_buffers.cpp don't
# need to be renamed in lockstep with user-facing knob changes.
set(GRACE_SAME_LEVEL_FACE_AVERAGE ${GRACE_SAME_LEVEL_FLUX_AVERAGE})
if( GRACE_SAME_LEVEL_FLUX_AVERAGE )
    message(STATUS "Same-level flux averaging enabled — bit-exact mass conservation across rank boundaries.")
endif()

# Reconstructor selection.  Compile-time choice; substituted into
# GRACE_RECONSTRUCTION_T in grace_config.h.in.
#   WENOZ      — 5th-order WENO-Z (default, production)
#   MC2        — 2nd-order monotonized-central slope-limited
#   MINMOD     — 2nd-order minmod slope-limited
#   DONOR_CELL — 1st-order piecewise-constant (provably L/R-symmetric;
#                use for symmetry-preservation diagnostics, not production)
set(GRACE_RECONSTRUCTION "WENOZ" CACHE STRING
    "Spatial reconstructor (WENOZ|MC2|MINMOD|DONOR_CELL)")
set_property(CACHE GRACE_RECONSTRUCTION PROPERTY STRINGS
    WENOZ MC2 MINMOD DONOR_CELL)

if(GRACE_RECONSTRUCTION STREQUAL "WENOZ")
    set(GRACE_RECONSTRUCTION_T "weno_reconstructor_t<5>")
elseif(GRACE_RECONSTRUCTION STREQUAL "MC2")
    set(GRACE_RECONSTRUCTION_T "slope_limited_reconstructor_t<MCbeta>")
elseif(GRACE_RECONSTRUCTION STREQUAL "MINMOD")
    set(GRACE_RECONSTRUCTION_T "slope_limited_reconstructor_t<minmod>")
elseif(GRACE_RECONSTRUCTION STREQUAL "DONOR_CELL")
    set(GRACE_RECONSTRUCTION_T "donor_cell_reconstructor_t")
else()
    message(FATAL_ERROR
        "GRACE_RECONSTRUCTION=${GRACE_RECONSTRUCTION} is not one of WENOZ, MC2, MINMOD, DONOR_CELL.")
endif()
message(STATUS "Reconstructor: ${GRACE_RECONSTRUCTION} (${GRACE_RECONSTRUCTION_T})")

# Thermodynamic primitive reconstructed at cell faces.
#   TEMP  — reconstruct temperature T (default, GRACE convention).  For
#           cold polytropic equilibria, T scales co-linearly with rho along
#           the cold sequence, so a scale-invariant limiter (WENO-Z) gives
#           bit-exact preservation of K = P/rho^Gamma at faces.
#   PRESS — reconstruct pressure P.  Continuous across contact
#           discontinuities (advantageous for BNS merger contact /
#           shocked post-merger), but does not preserve cold-K
#           exactly because the limiter sees a different smoothness
#           indicator than rho does.  Requires the inverse EOS hook
#           eps_h_csnd2_temp_entropy__press_rho_ye to compute (eps, T, ...)
#           from (P, rho, Ye); tabulated EOS currently aborts in this hook.
set(GRACE_RECON_THERMO "TEMP" CACHE STRING
    "Thermodynamic primitive reconstructed at faces (TEMP|PRESS)")
set_property(CACHE GRACE_RECON_THERMO PROPERTY STRINGS TEMP PRESS)

if(GRACE_RECON_THERMO STREQUAL "TEMP")
    set(GRACE_RECON_THERMO_TEMP ON)
elseif(GRACE_RECON_THERMO STREQUAL "PRESS")
    set(GRACE_RECON_THERMO_PRESS ON)
else()
    message(FATAL_ERROR
        "GRACE_RECON_THERMO=${GRACE_RECON_THERMO} is not one of TEMP, PRESS.")
endif()
message(STATUS "Reconstructed thermo primitive: ${GRACE_RECON_THERMO}")

# Select a per-architecture performance-tuning header.  The chosen file is
# included into grace_config.h and defines kernel LaunchBounds / tile macros.
# Each source file keeps `#ifndef GRACE_* / #define ... default / #endif`
# fallbacks, so if a tuning header does not set a macro (or the generic
# header is chosen) the source-level default applies.  LaunchBounds is a
# no-op on CPU backends, so headers are safe to include unconditionally.
#
# Selection order:
#   1. User-provided -DGRACE_PERF_TUNING=<name> — short name matching a
#      subdirectory of performance_tuning/ (e.g. Mi300A, A100).  Useful when
#      Kokkos_ARCH_* doesn't propagate out of an installed Kokkos into the
#      parent CMake (common with find_package(Kokkos)).
#   2. Auto-detect from Kokkos_ARCH_* cache variables.
#   3. Fall through to generic (no arch detected).
set(GRACE_PERF_TUNING "" CACHE STRING
    "Select performance-tuning header by name (Mi300A|Mi250X|A100|H100|generic). Leave empty to auto-detect from Kokkos_ARCH_*.")

if(GRACE_PERF_TUNING)
    set(_grace_tuning_candidate "${CMAKE_SOURCE_DIR}/performance_tuning/${GRACE_PERF_TUNING}/tuning.h")
    if(GRACE_PERF_TUNING STREQUAL "generic")
        set(_grace_tuning_candidate "${CMAKE_SOURCE_DIR}/performance_tuning/generic.h")
    endif()
    if(NOT EXISTS "${_grace_tuning_candidate}")
        message(FATAL_ERROR
            "GRACE_PERF_TUNING=${GRACE_PERF_TUNING} but no header at ${_grace_tuning_candidate}. "
            "Available: Mi300A, Mi250X, A100, H100, generic.")
    endif()
    set(GRACE_PERF_TUNING_HEADER "${_grace_tuning_candidate}")
    message(STATUS "Perf-tuning header: ${GRACE_PERF_TUNING} (user-selected)")
elseif(Kokkos_ARCH_AMD_GFX942)
    set(GRACE_PERF_TUNING_HEADER "${CMAKE_SOURCE_DIR}/performance_tuning/Mi300A/tuning.h")
    message(STATUS "Perf-tuning header: Mi300A (gfx942)")
elseif(Kokkos_ARCH_AMD_GFX90A)
    set(GRACE_PERF_TUNING_HEADER "${CMAKE_SOURCE_DIR}/performance_tuning/Mi250X/tuning.h")
    message(STATUS "Perf-tuning header: Mi250X (gfx90a)")
elseif(Kokkos_ARCH_AMPERE80)
    set(GRACE_PERF_TUNING_HEADER "${CMAKE_SOURCE_DIR}/performance_tuning/A100/tuning.h")
    message(STATUS "Perf-tuning header: A100 (sm_80)")
elseif(Kokkos_ARCH_HOPPER90)
    set(GRACE_PERF_TUNING_HEADER "${CMAKE_SOURCE_DIR}/performance_tuning/H100/tuning.h")
    message(STATUS "Perf-tuning header: H100 (sm_90)")
else()
    set(GRACE_PERF_TUNING_HEADER "${CMAKE_SOURCE_DIR}/performance_tuning/generic.h")
    message(STATUS "Perf-tuning header: generic (no arch detected — override with -DGRACE_PERF_TUNING=<name>)")
endif()

configure_file("${CMAKE_SOURCE_DIR}/include/grace/grace_config.h.in" grace_config.h)

configure_file( "${CMAKE_SOURCE_DIR}/include/grace/config/code_modules.h.in" code_modules.h )
include_directories( "${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(src)


add_executable(grace src/grace.cpp)
target_include_directories(grace PRIVATE "${HEADER_DIR}")
target_link_libraries(grace PRIVATE 
        grace_objects
        yaml_cpp::yaml
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        Kokkos::kokkos 
        p4est::p4est 
        p4est::sc
        spdlog::spdlog
        HDF5::HDF5
	    ZLIB::ZLIB
        # FUKA handling:
        $<$<BOOL:${GRACE_ENABLE_FUKA}>:Kadath::kadath>
        # Maybe we should remove this altogether 
        $<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling>
        $<$<BOOL:${GRACE_ENABLE_VTK}>:VTK::VTK>
        # LORENE 
        $<$<BOOL:${GRACE_ENABLE_LORENE}>:LORENE::LORENE>
        # TwoPunctures
        $<$<BOOL:${GRACE_ENABLE_TWO_PUNCTURES}>:TwoPunctures::TwoPunctures>
        )
if ( GRACE_ENABLE_VTK )
    vtk_module_autoinit(
        TARGETS grace
        MODULES ${VTK_LIBRARIES}
    )
endif() 

option(GRACE_ENABLE_TESTING "Build the tests" ON)
if(GRACE_ENABLE_TESTING)
    # Test parfiles live under `test/configs/` in source, are copied as-is
    # into the matching path in the build tree so ctest invocations of the
    # form `./<test> --grace-parfile ./configs/<name>.yaml` resolve from
    # `<build>/test/`.  Glob keeps this list maintenance-free: a new yaml
    # file in `test/configs/` is picked up on the next configure run.
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test/configs)
    file(GLOB _grace_test_parfiles
        CONFIGURE_DEPENDS
        RELATIVE "${CMAKE_SOURCE_DIR}/test/configs"
        "${CMAKE_SOURCE_DIR}/test/configs/*.yaml")
    foreach(_pf ${_grace_test_parfiles})
        configure_file("${CMAKE_SOURCE_DIR}/test/configs/${_pf}"
                       "${CMAKE_BINARY_DIR}/test/configs/${_pf}" COPYONLY)
    endforeach()
    unset(_grace_test_parfiles)
    enable_testing()
    include(setup_catch2)
    add_subdirectory(test)
endif()

option(GRACE_ENABLE_BENCHMARKS "Build the benchmarks" OFF)
if(GRACE_ENABLE_BENCHMARKS)
add_subdirectory(profiling)
endif()

#=====================================================================
# GRACE configuration summary
#
# Printed to the console at the end of configure (so it survives the
# `-- ` noise above) and written to
# `${CMAKE_BINARY_DIR}/grace_config_summary.txt` for later inspection
# (e.g. archived alongside benchmark output).
#=====================================================================

# Derive the backend label from the GRACE_ENABLE_* flags set by setup_kokkos.
if(GRACE_ENABLE_CUDA)
    set(_grace_backend "CUDA")
elseif(GRACE_ENABLE_HIP)
    set(_grace_backend "HIP")
elseif(GRACE_ENABLE_SYCL)
    set(_grace_backend "SYCL")
elseif(GRACE_ENABLE_OMP)
    set(_grace_backend "OpenMP")
else()
    set(_grace_backend "Serial")
endif()

# Coordinate system label.
if(GRACE_SPHERICAL_COORDINATES)
    set(_grace_coords "spherical")
else()
    set(_grace_coords "cartesian")
endif()

# Helper: stringify a boolean CMake variable as ON / OFF.
macro(_grace_bool_label var out)
    if(${var})
        set(${out} "ON")
    else()
        set(${out} "OFF")
    endif()
endmacro()

_grace_bool_label(GRACE_ENABLE_PARTICLES    _grace_particles)
_grace_bool_label(GRACE_ENABLE_M1           _grace_m1)
_grace_bool_label(GRACE_ENABLE_FOFC         _grace_fofc)
_grace_bool_label(GRACE_ENABLE_LORENE       _grace_lorene)
_grace_bool_label(GRACE_ENABLE_FUKA         _grace_fuka)
_grace_bool_label(GRACE_ENABLE_TWO_PUNCTURES _grace_twopuncture)
_grace_bool_label(GRACE_ENABLE_VTK          _grace_vtk)
_grace_bool_label(GRACE_ENABLE_PROFILING    _grace_profiling)
_grace_bool_label(GRACE_ENABLE_DETERMINISTIC_MPI_REDUCTIONS _grace_det_reductions)
_grace_bool_label(GRACE_SAME_LEVEL_FLUX_AVERAGE _grace_samelevel_avg)
_grace_bool_label(GRACE_FREEZE_HYDRO        _grace_freeze_hydro)

# Flag matter / Z4c FD-order mismatch directly in the summary so it's
# unmissable; matches the warning emitted at configure time.
if(NOT GRACE_MATTER_METRIC_DER_ORDER STREQUAL GRACE_Z4C_DER_ORDER)
    set(_grace_fd_order_warn "  [MISMATCH vs Z4c — see warning above]")
else()
    set(_grace_fd_order_warn "")
endif()

set(_grace_summary
"==========================================================\n"
"            GRACE Configuration Summary\n"
"==========================================================\n"
"  Version             : ${CMAKE_PROJECT_VERSION}\n"
"  Build type          : ${CMAKE_BUILD_TYPE}\n"
"  Backend             : ${_grace_backend}\n"
"  Coordinates         : ${_grace_coords}\n"
"  Dimensions          : ${GRACE_NSPACEDIM}\n"
"  Perf-tuning header  : ${GRACE_PERF_TUNING_HEADER}\n"
"  ---------------------------------------------------------\n"
"  FOFC                : ${_grace_fofc}\n"
"  Metric evolution    : ${GRACE_METRIC_EVOL}\n"
"  Z4c FD order        : ${GRACE_Z4C_DER_ORDER}\n"
"  Matter metric-d ord : ${GRACE_MATTER_METRIC_DER_ORDER}${_grace_fd_order_warn}\n"
"  EMF scheme          : ${GRACE_EMF_SCHEME}\n"
"  Riemann solver      : ${GRACE_RIEMANN_SOLVER}\n"
"  Reconstruction      : ${GRACE_RECONSTRUCTION} (${GRACE_RECONSTRUCTION_T})\n"
"  Reconstructed thermo: ${GRACE_RECON_THERMO}\n"
"  Freeze hydro        : ${_grace_freeze_hydro}\n"
"  ---------------------------------------------------------\n"
"  Particles           : ${_grace_particles}\n"
"  M1 radiation        : ${_grace_m1}\n"
"  ---------------------------------------------------------\n"
"  LORENE ID           : ${_grace_lorene}\n"
"  FUKA ID             : ${_grace_fuka}\n"
"  TwoPunctures ID     : ${_grace_twopuncture}\n"
"  ---------------------------------------------------------\n"
"  VTK output          : ${_grace_vtk}\n"
"  Profiling           : ${_grace_profiling}\n"
"  Deterministic MPI Σ : ${_grace_det_reductions}\n"
"  Same-level flux avg : ${_grace_samelevel_avg}\n"
"==========================================================\n"
"  Source dir          : ${CMAKE_SOURCE_DIR}\n"
"  Build dir           : ${CMAKE_BINARY_DIR}\n"
"==========================================================\n"
)

# Glue list -> single string (string(JOIN ...) requires CMake 3.12+; we're on 3.22).
string(JOIN "" _grace_summary_str ${_grace_summary})

message(NOTICE "\n${_grace_summary_str}")

file(WRITE "${CMAKE_BINARY_DIR}/grace_config_summary.txt" "${_grace_summary_str}")
message(STATUS "Configuration summary written to ${CMAKE_BINARY_DIR}/grace_config_summary.txt")

endif()
