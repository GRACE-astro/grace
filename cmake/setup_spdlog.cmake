# spdlog dependency resolution.
#
#   GRACE_USE_BUNDLED_DEPS = ON  →  in-tree submodule (extern/spdlog).
#   GRACE_USE_BUNDLED_DEPS = OFF →  system install (find_package, honours
#                                   SPDLOG_ROOT env var).
#
# Both paths leave the `spdlog::spdlog` target available to downstream
# link blocks.

if(GRACE_USE_BUNDLED_DEPS)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/spdlog/CMakeLists.txt")
        message(FATAL_ERROR
            "GRACE_USE_BUNDLED_DEPS=ON but extern/spdlog is empty.  Run:\n"
            "  git submodule update --init --recursive extern/spdlog")
    endif()
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_BENCH   OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/extern/spdlog)
else()
    if(NOT SPDLOG_ROOT)
        set(SPDLOG_ROOT "$ENV{SPDLOG_ROOT}")
    endif()
    message(STATUS "Searching path ${SPDLOG_ROOT}")
    find_package(spdlog REQUIRED PATHS "${SPDLOG_ROOT}")
    message(STATUS "spdlog libraries: ${SPDLOG_LIBRARIES}")
    message(STATUS "spdlog includes: ${SPDLOG_INCLUDE_DIR}")
    if(NOT TARGET spdlog::spdlog)
        add_library(spdlog::spdlog IMPORTED INTERFACE)
        set_property(TARGET spdlog::spdlog APPEND PROPERTY
                     INTERFACE_INCLUDE_DIRECTORIES "${SPDLOG_INCLUDE_DIRS}")
        set_property(TARGET spdlog::spdlog APPEND PROPERTY
                     INTERFACE_LINK_LIBRARIES "${SPDLOG_LIBRARIES}")
    endif()
endif()
