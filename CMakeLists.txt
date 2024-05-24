
cmake_minimum_required(VERSION 3.14)

project(thunder VERSION 0.5 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(code_modules)
include(check_compiler)
include(prohibit_in_source_build)
include(setup_flags)
include(setup_mpi)
find_package(OpenMP REQUIRED)
find_package(p4est REQUIRED)
find_package(LibXml2 REQUIRED)
include(setup_kokkos)
include(setup_gsl)
include(setup_yaml)
include(setup_vtk)
include(setup_spdlog)
include(setup_HDF5)

configure_file( "${CMAKE_SOURCE_DIR}/include/thunder/config/code_modules.h.in" code_modules.h )
include_directories( "${CMAKE_CURRENT_BINARY_DIR}")

configure_file("${CMAKE_SOURCE_DIR}/include/thunder/thunder_config.h.in" thunder_config.h)

set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")

add_subdirectory(src)

add_executable(thunder src/thunder.cpp)
target_include_directories(thunder PRIVATE "${HEADER_DIR}")
target_link_libraries(thunder PRIVATE 
        thunder_objects
        $<TARGET_OBJECTS:thunder_vtk_output>
        $<TARGET_OBJECTS:thunder_hdf5_output>
        $<TARGET_OBJECTS:thunder_cell_output>
        yaml_cpp::yaml
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        Kokkos::kokkos 
        p4est::p4est 
        p4est::sc
        VTK::VTK
        spdlog::spdlog
        HDF5::HDF5 
        )
vtk_module_autoinit(
    TARGETS thunder
    MODULES ${VTK_LIBRARIES}
)

option(THUNDER_ENABLE_TESTING "Build the tests" ON)
if(THUNDER_ENABLE_TESTING)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test/configs)
    configure_file("${CMAKE_SOURCE_DIR}/configs/basic_config.yaml" 
                  "${CMAKE_BINARY_DIR}/test/configs/basic_config.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/regrid_test_cartesian.yaml" 
                  "${CMAKE_BINARY_DIR}/test/configs/regrid_test_cartesian.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/regrid_test_spherical.yaml" 
                   "${CMAKE_BINARY_DIR}/test/configs/regrid_test_spherical.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/regrid_test_spherical_log.yaml" 
                   "${CMAKE_BINARY_DIR}/test/configs/regrid_test_spherical_log.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/scalar_reductions_test.yaml" 
                   "${CMAKE_BINARY_DIR}/test/configs/scalar_reductions_test.yaml" COPYONLY)
    enable_testing()
    include(setup_catch2)
    add_subdirectory(test)
endif()

option(THUNDER_ENABLE_BENCHMARKS "Build the benchmarks" ON)
if(THUNDER_ENABLE_BENCHMARKS)
add_subdirectory(profiling)
endif()
