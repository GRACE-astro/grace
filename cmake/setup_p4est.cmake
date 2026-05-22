# p4est dependency resolution.
#
#   GRACE_USE_BUNDLED_DEPS = ON  →  in-tree submodule (extern/p4est).
#   GRACE_USE_BUNDLED_DEPS = OFF →  system install (find_package, honours
#                                   P4EST_ROOT env var).

if(GRACE_USE_BUNDLED_DEPS)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/p4est/CMakeLists.txt")
        message(FATAL_ERROR
            "GRACE_USE_BUNDLED_DEPS=ON but extern/p4est is empty.  Run:\n"
            "  git submodule update --init --recursive extern/p4est")
    endif()
    # p4est build options that GRACE needs:
    #   * 3D variant (p8est) — GRACE only uses the 3D library.
    #   * MPI on for both p4est and libsc.  The variable name has shifted
    #     across p4est versions (`MPI`, `mpi`, `P4EST_ENABLE_MPI`,
    #     `SC_ENABLE_MPI`); set them all to be safe.
    #   * Tests + examples off for p4est, libsc, and the default CTest
    #     `BUILD_TESTING` (set ON by `include(CTest)` higher up).  The
    #     tests are pure C and inherit GRACE-wide HIP/SYCL link flags via
    #     directory scope, which they can't link against — disabling them
    #     also makes the bundled build smaller and faster.
    set(P4EST_ENABLE_BUILD_3D ON  CACHE BOOL "" FORCE)
    set(P4EST_ENABLE_MPI      ON  CACHE BOOL "" FORCE)
    set(SC_ENABLE_MPI         ON  CACHE BOOL "" FORCE)
    set(MPI                   ON  CACHE BOOL "" FORCE)
    set(mpi                   ON  CACHE BOOL "" FORCE)
    set(P4EST_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
    set(P4EST_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    set(SC_BUILD_TESTING      OFF CACHE BOOL "" FORCE)
    set(SC_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
    # Override the default CTest variable just inside the p4est subdir;
    # we restore it after add_subdirectory so GRACE's own tests are
    # unaffected.
    set(_grace_saved_build_testing "${BUILD_TESTING}")
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/extern/p4est)
    set(BUILD_TESTING "${_grace_saved_build_testing}" CACHE BOOL "" FORCE)
    unset(_grace_saved_build_testing)
    # Bundled p4est exposes plain `p4est` and `sc` targets (no namespace).
    # GRACE's link blocks (and many downstream users) expect the namespaced
    # `p4est::p4est` / `p4est::sc` that the system find_package config
    # provides.  Bridge the difference with ALIAS targets.
    if(TARGET p4est AND NOT TARGET p4est::p4est)
        add_library(p4est::p4est ALIAS p4est)
    endif()
    if(TARGET sc AND NOT TARGET p4est::sc)
        add_library(p4est::sc ALIAS sc)
    endif()
else()
    if(NOT P4EST_ROOT)
        set(P4EST_ROOT "$ENV{P4EST_ROOT}")
    endif()
    find_package(p4est REQUIRED)
endif()
