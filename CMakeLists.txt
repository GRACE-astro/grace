
cmake_minimum_required(VERSION 3.14)

project(thunder VERSION 0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


include(check_compiler)

include(prohibit_in_source_build)

include(setup_flags)

include(setup_mpi)

find_package(p4est REQUIRED)

include(setup_gsl)
include(setup_yaml)
include(setup_vtk)
include(setup_kokkos)


configure_file( "${CMAKE_SOURCE_DIR}/include/thunder/config/code_modules.h.in" code_modules.h )
include_directories( "${CMAKE_CURRENT_BINARY_DIR}")

configure_file("${CMAKE_SOURCE_DIR}/include/thunder/thunder_config.h.in" thunder_config.h)

add_subdirectory(src)

option(thunder_TESTS "Build the tests" ON)
if(thunder_TESTS)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test/configs)
    configure_file("${CMAKE_SOURCE_DIR}/configs/basic_config.yaml" 
                  "${CMAKE_BINARY_DIR}/test/configs/basic_config.yaml" COPYONLY)
    enable_testing()
    include(setup_catch2)
    add_subdirectory(test)
endif()
