
cmake_minimum_required(VERSION 3.14)

project(thunder VERSION 0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


include(check_compiler)

include(prohibit_in_source_build)

include(setup_flags)

include(setup_mpi)

find_package(OpenMP REQUIRED)

find_package(p4est REQUIRED)

include(setup_gsl)
include(setup_yaml)
include(setup_vtk)
include(setup_kokkos)

configure_file( "${CMAKE_SOURCE_DIR}/include/thunder/config/code_modules.h.in" code_modules.h )
include_directories( "${CMAKE_CURRENT_BINARY_DIR}")

configure_file("${CMAKE_SOURCE_DIR}/include/thunder/thunder_config.h.in" thunder_config.h)

set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")

add_subdirectory(src)

add_executable(thunder src/thunder.cpp)
target_include_directories(thunder PRIVATE "${HEADER_DIR}")
target_link_libraries(thunder PRIVATE 
        thunder_objects
        $<TARGET_OBJECTS:thunder_volume_output>
        yaml_cpp::yaml
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        Kokkos::kokkos 
        p4est::p4est 
        p4est::sc
        VTK::VTK
        )
vtk_module_autoinit(
    TARGETS thunder
    MODULES ${VTK_LIBRARIES}
)

option(thunder_TESTS "Build the tests" ON)
if(thunder_TESTS)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test/configs)
    configure_file("${CMAKE_SOURCE_DIR}/configs/basic_config.yaml" 
                  "${CMAKE_BINARY_DIR}/test/configs/basic_config.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/regrid_test_cartesian.yaml" 
                  "${CMAKE_BINARY_DIR}/test/configs/regrid_test_cartesian.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/regrid_test_spherical.yaml" 
                   "${CMAKE_BINARY_DIR}/test/configs/regrid_test_spherical.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/regrid_test_spherical_log.yaml" 
                   "${CMAKE_BINARY_DIR}/test/configs/regrid_test_spherical_log.yaml" COPYONLY)
    enable_testing()
    include(setup_catch2)
    add_subdirectory(test)
endif()
