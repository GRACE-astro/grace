#[=======================================================================[.rst:
FindKadath
-------

Finds the Kadath library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``Kadath::kadath``
  The Kadath library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Kadath_FOUND``
  True if the system has the Kadath library.
``Kadath_INCLUDE_DIRS``
  Include directories needed to use Kadath.
``Kadath_LIBRARIES``
  Libraries needed to link to Kadath.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Kadath_INCLUDE_DIR``
  The directory containing ``Kadath.hpp``.
``Kadath_LIBRARY``
  The path to the Kadath library (there is only one, in fact).

#]=======================================================================]

# if(NOT HOME_KADATH)
#     set(HOME_KADATH    "")
#     set(HOME_KADATH    "$ENV{HOME_KADATH}") 
# endif()
# message(STATUS "Looking for Kadath in ${HOME_KADATH}")

# # find_path( Kadath_INCLUDE_DIRS
# #     NAMES  exporter_utilities.hpp kadath.hpp
# #     PATHS ${HOME_KADATH}
# #     PATH_SUFFIXES include include/Kadath_point_h
# # )

# # First find the main include directory
# find_path(Kadath_INCLUDE_DIR_MAIN
#     NAMES kadath.hpp
#     PATHS ${HOME_KADATH}
#     PATH_SUFFIXES include
# )

# # Optionally check for the Kadath_point_h subdirectory
# find_path(Kadath_INCLUDE_DIR_POINT_H
#     NAMES exporter_utilities.hpp
#     PATHS ${HOME_KADATH}
#     PATH_SUFFIXES include/Kadath_point_h
# )

# set(Kadath_INCLUDE_DIRS ${Kadath_INCLUDE_DIR_MAIN} ${Kadath_INCLUDE_DIR_POINT_H})

# find_library( Kadath_LIBRARY
#     NAMES kadath
#     PATHS ${HOME_KADATH} #${CMAKE_SOURCE_DIR}/extern/Kadath # note that if we'd like to compile Kadath along, we'd need to include more libs (GSL,FFTW,Scalapack,Lapack) in the build step 
#     PATH_SUFFIXES local/lib lib
# )


# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(Kadath
#   FOUND_VAR Kadath_FOUND
#   REQUIRED_VARS
#     Kadath_LIBRARY
#     Kadath_INCLUDE_DIR_MAIN
#     Kadath_INCLUDE_DIR_POINT_H
# )


# if(Kadath_FOUND AND NOT TARGET Kadath::kadath)
#     add_library(Kadath::kadath UNKNOWN IMPORTED)
#     set_target_properties(Kadath::kadath PROPERTIES
#     IMPORTED_LOCATION "${Kadath_LIBRARY}"
#     INTERFACE_INCLUDE_DIRECTORIES "${Kadath_INCLUDE_DIRS}")
# endif()

# message(STATUS "Kadath library: ${Kadath_LIBRARY}")
# message(STATUS "Kadath include: ${Kadath_INCLUDE_DIRS}")

# Modified by Carlo Musolino <carlo.musolino@aei.mpg.de> check if HOME_KADATH
# exists before using it. Also, package dependencies inside imported target.



if( NOT HOME_KADATH )
  if( EXISTS "$ENV{HOME_KADATH}" )
    file( TO_CMAKE_PATH "$ENV{HOME_KADATH}" HOME_KADATH )
  else()
    message( FATAL_ERROR "Kadath requested but HOME_KADATH is not set" )
  endif()
endif()

message( STATUS "Looking for Kadath in ${HOME_KADATH}" )

# The main Kadath headers are in Kadath_point_h
find_path(Kadath_INCLUDE_DIR_MAIN
    NAMES kadath.hpp
    PATHS ${HOME_KADATH}
    PATH_SUFFIXES include/Kadath_point_h
)

# The utility header is in top-level include
find_path(Kadath_INCLUDE_DIR_UTIL
    NAMES exporter_utilities.hpp
    PATHS ${HOME_KADATH}
    PATH_SUFFIXES include
)

# Combine both include dirs (both required)
set(Kadath_INCLUDE_DIRS ${Kadath_INCLUDE_DIR_MAIN} ${Kadath_INCLUDE_DIR_UTIL})

# Library
find_library(Kadath_LIBRARY
    NAMES kadath
    PATHS ${HOME_KADATH}
    PATH_SUFFIXES lib local/lib
)

# Require both include dirs and library
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Kadath
  FOUND_VAR Kadath_FOUND
  REQUIRED_VARS Kadath_LIBRARY Kadath_INCLUDE_DIR_MAIN Kadath_INCLUDE_DIR_UTIL
)

# Now we find the dependencies 
find_package(GSL REQUIRED)
find_package(Boost REQUIRED)
find_package(FFTW REQUIRED COMPONENTS DOUBLE)

set( MKL_INTERFACE    "lp64" CACHE STRING  "MKL interface layer (lp64 or ilp64)" )
set( ENABLE_BLACS     ON     CACHE BOOL    "Enable BLACS support in MKL"          )
set( ENABLE_SCALAPACK ON     CACHE BOOL    "Enable ScaLAPACK support in MKL"      )

find_package(MKL CONFIG QUIET 
    PATHS 
        $ENV{MKLROOT}/lib/cmake 
        ${MKL_ROOT}/lib/cmake
)
if( MKL_FOUND )
  if( MKL_VERSION VERSION_LESS "2021.0" )
    message( WARNING "MKL < 2021 detected: MKL::MKL_SCALAPACK target may not exist, consider upgrading" )
  endif()
  message( STATUS "Kadath: using Intel MKL for BLAS/ScaLAPACK" )
elseif( CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment"
     OR DEFINED ENV{CRAY_LIBSCI_PREFIX_DIR}
     OR DEFINED ENV{CRAY_LIBSCI_BASE_DIR} )
  find_package( SCICRAY REQUIRED )
  message( STATUS "Kadath: using Cray LibSci for BLAS/ScaLAPACK" )
else()
  find_package( OpenBLAS  REQUIRED )
  find_package( ScaLAPACK REQUIRED )
  message( STATUS "Kadath: using OpenBLAS + ScaLAPACK" )
endif()

# --- Imported target ---
if( Kadath_FOUND AND NOT TARGET Kadath::kadath )
  add_library( Kadath::kadath UNKNOWN IMPORTED )
  set_target_properties( Kadath::kadath PROPERTIES
    IMPORTED_LOCATION             "${Kadath_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Kadath_INCLUDE_DIRS}"
  )

  target_link_libraries(
    Kadath::kadath INTERFACE 
    FFTW::FFTW 
    GSL::gsl 
    Boost::headers
  )

  if( MKL_FOUND )
    target_link_libraries(
      Kadath::kadath INTERFACE 
      MKL::MKL_SCALAPACK
    )
  elseif( SCICRAY_FOUND )
    target_link_libraries(
      Kadath::kadath INTERFACE 
      ${SCICRAY_LIBRARIES}
    )
  else()
    target_link_libraries(
      Kadath::kadath INTERFACE 
      OpenBLAS::OpenBLAS
      ScaLAPACK::ScaLAPACK
    )
  endif()
endif()

message(STATUS "Found Kadath library and dependencies")