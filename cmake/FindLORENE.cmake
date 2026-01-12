#[=======================================================================[.rst:
Findlorene
-------

Finds the LORENE library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``LORENE::LORENE``
  The LORENE library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``LORENE_FOUND``
  True if the system has the LORENE library.
``LORENE_INCLUDE_DIRS``
  Include directories needed to use LORENE.
``LORENE_LIBRARIES``
  Libraries needed to link to LORENE.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``LORENE_INCLUDE_DIR``
  The directory containing Lorene headers.
``LORENE_LIBRARY``
  The path to the LORENE library.

#]=======================================================================]
if (NOT LORENE_ROOT)
    set(LORENE_ROOT "")
    set(LORENE_ROOT "$ENV{LORENE_ROOT}")
endif()

message("Looking for LORENE in ${LORENE_ROOT}")

find_path(
    LORENE_INCLUDE_DIR
    NAMES unites.h 
    PATHS ${LORENE_ROOT}
    PATH_SUFFIXES C++/Include 
)

find_path(
    LORENE_EXPORT_INCLUDE_DIR
    NAMES bin_ns.h
    PATHS ${LORENE_ROOT}
    PATH_SUFFIXES Export/C++/Include
)

find_library( LORENE_LIBRARY
    NAMES lorene
    PATHS ${LORENE_ROOT}
    PATH_SUFFIXES Lib
)

find_library( LORENE_EXPORT_LIBRARY
    NAMES lorene_export
    PATHS ${LORENE_ROOT}
    PATH_SUFFIXES Lib
)

find_library( LORENE_FF_LIBRARY
    NAMES lorenef77
    PATHS ${LORENE_ROOT}
    PATH_SUFFIXES Lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LORENE
  FOUND_VAR LORENE_FOUND
  REQUIRED_VARS
    LORENE_LIBRARY
    LORENE_EXPORT_LIBRARY
    LORENE_FF_LIBRARY
    LORENE_INCLUDE_DIR
    LORENE_EXPORT_INCLUDE_DIR
)

# LORENE depends on GSL LAPACK and BLAS, so 
# we pull them in too 
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Try FFTW explicitly
find_package(FFTW)

# FFTW is required unless MKL is the BLAS provider
#if (NOT FFTW_FOUND AND NOT ENV{MKLROOT})
#    message(FATAL_ERROR
#        "LORENE requires FFTW, but neither FFTW nor MKL was found")
#endif()
enable_language(Fortran)
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

find_package(GSL REQUIRED)

if (LORENE_FOUND)
    set( LORENE_LIBRARIES ${LORENE_LIBRARY})
    set( LORENE_INCLUDE_DIRS "${LORENE_INCLUDE_DIR};${LORENE_EXPORT_INCLUDE_DIR}")
endif() 
message(STATUS "LORENE libs: ${LORENE_LIBRARIES}")
message(STATUS "LORENE include: ${LORENE_INCLUDE_DIRS}")

if (LORENE_FOUND AND NOT TARGET LORENE::LORENE)
    add_library(LORENE::core STATIC IMPORTED)
    set_target_properties(LORENE::core PROPERTIES
        IMPORTED_LOCATION "${LORENE_LIBRARY}"   # liblorene.a
    )

    add_library(LORENE::export STATIC IMPORTED)
    set_target_properties(LORENE::export PROPERTIES
        IMPORTED_LOCATION "${LORENE_EXPORT_LIBRARY}" # liblorene_export.a
    )

    add_library(LORENE::ff STATIC IMPORTED)
    set_target_properties(LORENE::ff PROPERTIES
        IMPORTED_LOCATION "${LORENE_FF_LIBRARY}" # liblorene_export.a
    )

    add_library(LORENE::LORENE INTERFACE IMPORTED)

    target_link_libraries(LORENE::LORENE INTERFACE
        LORENE::core
        LORENE::ff
        LORENE::export
        ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
    )

    target_include_directories(LORENE::LORENE INTERFACE
      ${LORENE_INCLUDE_DIR}
      ${LORENE_EXPORT_INCLUDE_DIR}
    )

    target_link_libraries(
      LORENE::LORENE INTERFACE 
      GSL::gsl
    )

    target_link_libraries(
        LORENE::LORENE INTERFACE 
        BLAS::BLAS
        LAPACK::LAPACK
        $<$<BOOL:${FFTW_FOUND}>:FFTW::FFTW>
    )

endif() 

mark_as_advanced(
    LORENE_INCLUDE_DIR 
    LORENE_LIBRARY
)