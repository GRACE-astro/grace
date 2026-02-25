
cmake_minimum_required(VERSION 3.14)

project(GRACE VERSION 0.5 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(code_modules)
include(check_compiler)
include(prohibit_in_source_build)
include(setup_flags)

# --- Options ---
option(GRACE_ENABLE_COWLING_METRIC "Use Cowling approximation (default)" ON)
option(GRACE_ENABLE_Z4C_METRIC     "Use full Z4c evolution" OFF)

if(GRACE_ENABLE_Z4C_METRIC)
    message(STATUS "GRACE will use full Z4c metric (GRACE_ENABLE_Z4C_METRIC=ON).")
    set(GRACE_ENABLE_COWLING_METRIC OFF)
elseif(GRACE_ENABLE_COWLING_METRIC)
    message(STATUS "GRACE will use full Cowling metric (GRACE_ENABLE_COWLING=ON).")
    set(GRACE_ENABLE_Z4C_METRIC OFF)
else()
    message(STATUS "No option explicitly enabled; defaulting to Cowling metric.")
    set(GRACE_ENABLE_COWLING_METRIC ON)
endif()

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
    # note that order of linking will be important for the final executable!
    find_package(Kadath REQUIRED)
    find_package(GSL REQUIRED)
    find_package(FFTW REQUIRED COMPONENTS DOUBLE)

    find_package(MKL QUIET) 
    if(MKL_FOUND)
        set(MKL_INTERFACE_LAYER LP64)  
    	message(STATUS "Using Intel MKL")
    else()
        # Platform-dependent BLAS/ScaLAPACK handling
        if(CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment" OR DEFINED ENV{CRAY_LIBSCI_PREFIX_DIR} OR DEFINED ENV{CRAY_LIBSCI_BASE_DIR})
            find_package(SCICRAY REQUIRED)
    	    message(STATUS "Using Cray's SciLib")
            # Cray system: use LibSci
            set(BLAS_LIBRARIES ${SCICRAY_LIBRARIES})
            set(SCALAPACK_LIBRARIES ${SCICRAY_LIBRARIES})
            set(OPENBLAS_FOUND OFF) 
            set(OpenBLAS_LIB "") # silently turn off - Cray's SciLib already has BLAS routines
            set(SCALAPACK_FOUND ON)
        else() 
    	    message(STATUS "Using OpenBLAS and ScaLAPACK")
            # non-Cray system
            # find openblas and netlib-scalapack independently
            find_package(OPENBLAS REQUIRED)
            find_package(SCALAPACK REQUIRED)
        endif()
    endif()
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
        $<$<BOOL:${GRACE_ENABLE_FUKA}>:${GSL_LIBRARY}>
        $<$<BOOL:${GRACE_ENABLE_FUKA}>:${FFTW_LIB}>
        # Libraries to link if GRACE_ENABLE_FUKA and MKL_FOUND are both true
        $<$<AND:$<BOOL:${GRACE_ENABLE_FUKA}>,$<BOOL:${MKL_FOUND}>>:
        ${MKL_SCALAPACK_LIBRARY}
        ${MKL_INTERFACE_LIBRARY}
        ${MKL_SEQUENTIAL_LAYER_LIBRARY}
        ${MKL_CORE_LIBRARY}
        ${MKL_BLACS_LIBRARY}
        >
        # Libraries to link to otherwise 
        $<$<AND:$<BOOL:${GRACE_ENABLE_FUKA}>,$<NOT:$<BOOL:${MKL_FOUND}>>>:
        ${SCALAPACK_LIBRARIES}        
        ${OpenBLAS_LIB}
        >
        $<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling>
        $<$<BOOL:${GRACE_ENABLE_VTK}>:VTK::VTK>
        $<$<BOOL:${GRACE_ENABLE_LORENE}>:LORENE::LORENE>
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
