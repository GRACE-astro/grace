# Catch2 dependency resolution.
#
#   GRACE_USE_BUNDLED_DEPS = ON  →  in-tree submodule (extern/Catch2).
#   GRACE_USE_BUNDLED_DEPS = OFF →  system install (find_package, honours
#                                   CATCH2_ROOT env var).

if(GRACE_USE_BUNDLED_DEPS)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/Catch2/CMakeLists.txt")
        message(FATAL_ERROR
            "GRACE_USE_BUNDLED_DEPS=ON but extern/Catch2 is empty.  Run:\n"
            "  git submodule update --init --recursive extern/Catch2")
    endif()
    set(CATCH_BUILD_TESTING            OFF CACHE BOOL "" FORCE)
    set(CATCH_BUILD_EXAMPLES           OFF CACHE BOOL "" FORCE)
    set(CATCH_BUILD_EXTRA_TESTS        OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_DOCS             OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/extern/Catch2)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/extern/Catch2/extras)
    include(Catch)
else()
    if (NOT CATCH2_ROOT)
        set(CATCH2_ROOT "$ENV{CATCH2_ROOT}")
    endif()
    find_package(Catch2 PATHS "${CATCH2_ROOT}" NO_DEFAULT_PATH)
    find_package(Catch2 REQUIRED)
    set(catch2_SOURCE_DIR "${CATCH2_ROOT}")
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
endif()
