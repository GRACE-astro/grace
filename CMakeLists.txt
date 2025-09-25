
cmake_minimum_required(VERSION 3.14)

project(GRACE VERSION 0.5 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(code_modules)
include(check_compiler)
include(prohibit_in_source_build)
include(setup_flags)

configure_file( "${CMAKE_SOURCE_DIR}/include/grace/config/code_modules.h.in" code_modules.h )
include_directories( "${CMAKE_CURRENT_BINARY_DIR}")

configure_file("${CMAKE_SOURCE_DIR}/include/grace/grace_config.h.in" grace_config.h)

set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")

option(GRACE_BUILD_DOCS "Build the docs" OFF)
option(GRACE_BUILD_DOCS_ONLY "Only build docs" OFF)
if(GRACE_BUILD_DOCS OR GRACE_BUILD_DOCS_ONLY ) 
    add_subdirectory(doc)
endif()

if( NOT GRACE_BUILD_DOCS_ONLY )

include(setup_mpi)
find_package(OpenMP REQUIRED)
find_package(p4est REQUIRED)
find_package(LibXml2 REQUIRED)
find_package(ZLIB REQUIRED)
include(setup_kokkos)
#include(setup_gsl)
include(setup_yaml)
include(setup_vtk)
include(setup_spdlog)
include(setup_HDF5)

if( GRACE_ENABLE_KADATH )
    # note that order of linking will be important for the final executable!
    find_package(Kadath REQUIRED)
    find_package(GSL REQUIRED)
    find_package(FFTW REQUIRED COMPONENTS DOUBLE)
    if(GRACE_USE_MKL) 
        find_package(MKL REQUIRED)
        # make sure we link the MKL's Kadath executables do;
        # (probably redundant; also, it doesn't pick out LP64 over ILP64 reliably)
        set(MKL_INTERFACE_LAYER LP64)  
    else()
        # Platform-dependent BLAS/ScaLAPACK handling
        if(CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment" OR DEFINED ENV{CRAY_LIBSCI_PREFIX_DIR})
            # Cray system: use LibSci
            if(DEFINED ENV{CRAY_LIBSCI_PREFIX_DIR})
            link_directories($ENV{CRAY_LIBSCI_PREFIX_DIR}/lib)
            endif()
            set(SCALAPACK_LIBRARIES sci_cray_mpi)
            
            set(BLAS_LIBRARIES "${SCALAPACK_LIBRARIES}")
            set(OPENBLAS_FOUND OFF)
            set(OpenBLAS_LIB "") # silently turn off - Cray's SciLib already has BLAS routines
            set(SCALAPACK_FOUND ON)
        else() 
            # non-cray system
            # find openblas and netlib-scalapack independently
            find_package(OPENBLAS REQUIRED)
            find_package(SCALAPACK REQUIRED)
        endif()
    endif()
endif()


option(GRACE_ENABLE_PROFILING "Build profiling hooks" ON)
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

add_subdirectory(src)

add_executable(grace src/grace.cpp)
target_include_directories(grace PRIVATE "${HEADER_DIR}")
target_link_libraries(grace PRIVATE 
        grace_objects
        $<TARGET_OBJECTS:grace_vtk_output>
        $<TARGET_OBJECTS:grace_hdf5_output>
        $<TARGET_OBJECTS:grace_cell_output>
        yaml_cpp::yaml
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        Kokkos::kokkos 
        p4est::p4est 
        p4est::sc
        VTK::VTK
        spdlog::spdlog
        HDF5::HDF5
	ZLIB::ZLIB
        # Kadath handling:
        $<$<BOOL:${GRACE_ENABLE_KADATH}>:Kadath::kadath>
        $<$<BOOL:${GRACE_ENABLE_KADATH}>:${GSL_LIBRARY}>
        $<$<BOOL:${GRACE_ENABLE_KADATH}>:${FFTW_LIB}>
        # Libraries to link if GRACE_ENABLE_KADATH and GRACE_USE_MKL are both true
        $<$<AND:$<BOOL:${GRACE_ENABLE_KADATH}>,$<BOOL:${GRACE_USE_MKL}>>:
        ${MKL_SCALAPACK_LIBRARY}
        ${MKL_INTERFACE_LIBRARY}
        ${MKL_SEQUENTIAL_LAYER_LIBRARY}
        ${MKL_CORE_LIBRARY}
        ${MKL_BLACS_LIBRARY}
        >
        # Libraries to link if GRACE_ENABLE_KADATH is true and GRACE_USE_MKL is false
        $<$<AND:$<BOOL:${GRACE_ENABLE_KADATH}>,$<NOT:$<BOOL:${GRACE_USE_MKL}>>>:
        ${SCALAPACK_LIBRARIES}        
        ${OpenBLAS_LIB}
        >
        # remaining conditional libraries
        $<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling>
        )
vtk_module_autoinit(
    TARGETS grace
    MODULES ${VTK_LIBRARIES}
)

option(GRACE_ENABLE_TESTING "Build the tests" ON)
if(GRACE_ENABLE_TESTING)
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
    configure_file("${CMAKE_SOURCE_DIR}/configs/pwpoly_test_config.yaml" 
                   "${CMAKE_BINARY_DIR}/test/configs/pwpoly_test_config.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/test/sly.h5" 
                   "${CMAKE_BINARY_DIR}/test/sly.h5" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/c2p_test_gamma2.yaml" 
                   "${CMAKE_BINARY_DIR}/test/configs/c2p_test_gamma2.yaml" COPYONLY)
    configure_file("${CMAKE_SOURCE_DIR}/configs/c2p_test_sly4_pw.yaml" 
                   "${CMAKE_BINARY_DIR}/test/configs/c2p_test_sly4_pw.yaml" COPYONLY)
    enable_testing()
    include(setup_catch2)
    add_subdirectory(test)
endif()

option(GRACE_ENABLE_BENCHMARKS "Build the benchmarks" OFF)
if(GRACE_ENABLE_BENCHMARKS)
add_subdirectory(profiling)
endif()

endif()
