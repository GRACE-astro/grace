# yaml-cpp dependency resolution.
#
#   GRACE_USE_BUNDLED_DEPS = ON  →  in-tree submodule (extern/yaml-cpp).
#   GRACE_USE_BUNDLED_DEPS = OFF →  system install (find_package, honours
#                                   YAML_ROOT env var or ~/libs/yaml-cpp-install
#                                   fallback for legacy installs).
#
# Both paths register the alias `yaml_cpp::yaml` for use throughout GRACE's
# link blocks — call sites do not need to know which path provided it.

if(GRACE_USE_BUNDLED_DEPS)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/yaml-cpp/CMakeLists.txt")
        message(FATAL_ERROR
            "GRACE_USE_BUNDLED_DEPS=ON but extern/yaml-cpp is empty.  Run:\n"
            "  git submodule update --init --recursive extern/yaml-cpp")
    endif()
    # Disable yaml-cpp's own tools/tests so they don't pollute the build.
    set(YAML_CPP_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB  OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/extern/yaml-cpp)
    # yaml-cpp 0.9.0 exposes `yaml-cpp::yaml-cpp` as an ALIAS pointing at
    # the underlying `yaml-cpp` target.  CMake forbids ALIAS→ALIAS, so we
    # have to alias the underlying target directly.  Earlier versions
    # don't have `yaml-cpp::yaml-cpp` at all — the elseif handles those.
    if(TARGET yaml-cpp)
        add_library(yaml_cpp::yaml ALIAS yaml-cpp)
    elseif(TARGET yaml-cpp::yaml-cpp)
        add_library(yaml_cpp::yaml ALIAS yaml-cpp::yaml-cpp)
    endif()
else()
    if(NOT YAML_ROOT)
        set(YAML_ROOT "$ENV{YAML_ROOT}")
    endif()
    if(YAML_ROOT)
        set(yaml-cpp_DIR "${YAML_ROOT}/lib64/cmake/yaml-cpp")
    else()
        set(yaml-cpp_DIR "$ENV{HOME}/libs/yaml-cpp-install/lib64/cmake/yaml-cpp")
    endif()
    message(STATUS "Looking for yaml-cpp config at: ${yaml-cpp_DIR}")
    find_package(yaml-cpp REQUIRED)
    message(STATUS "yaml-cpp found: ${yaml-cpp_FOUND}")
    message(STATUS "yaml-cpp version: ${yaml-cpp_VERSION}")
    if(TARGET yaml-cpp::yaml-cpp)
        message(STATUS "Using modern yaml-cpp::yaml-cpp target")
        add_library(yaml_cpp::yaml ALIAS yaml-cpp::yaml-cpp)
    else()
        message(STATUS "Using legacy yaml-cpp variables")
        message(STATUS "yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")
        message(STATUS "yaml-cpp includes: ${YAML_CPP_INCLUDE_DIR}")
        if(NOT TARGET yaml_cpp::yaml)
            add_library(yaml_cpp::yaml IMPORTED INTERFACE)
            set_property(TARGET yaml_cpp::yaml APPEND PROPERTY
                         INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIRS}")
            set_property(TARGET yaml_cpp::yaml APPEND PROPERTY
                         INTERFACE_LINK_LIBRARIES "${YAML_CPP_LIBRARIES}")
        endif()
    endif()
endif()
