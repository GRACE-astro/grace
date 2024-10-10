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

if(NOT HOME_KADATH)
    set(HOME_KADATH    "")
    set(HOME_KADATH    "$ENV{HOME_KADATH}") 
endif()
message(STATUS "Looking for Kadath in ${HOME_KADATH}")
find_path( Kadath_INCLUDE_DIRS
    NAMES  exporter_utilities.hpp kadath.hpp
    PATHS ${HOME_KADATH}
    PATH_SUFFIXES include include/Kadath_point_h
)



find_library( Kadath_LIBRARY
    NAMES kadath
    PATHS ${HOME_KADATH} #${CMAKE_SOURCE_DIR}/extern/Kadath # note that if we'd like to compile Kadath along, we'd need to include more libs (GSL,FFTW,Scalapack,Lapack) in the build step 
    PATH_SUFFIXES local/lib lib
)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Kadath
  FOUND_VAR Kadath_FOUND
  REQUIRED_VARS
    Kadath_LIBRARY
    Kadath_INCLUDE_DIRS
)

if(Kadath_FOUND)
    set(Kadath_LIBRARY "${Kadath_LIBRARY}")
    set(Kadath_INCLUDE_DIRS "${Kadath_INCLUDE_DIRS}")
endif()

if(Kadath_FOUND AND NOT TARGET Kadath::kadath)
    add_library(Kadath::kadath UNKNOWN IMPORTED)
    set_target_properties(Kadath::kadath PROPERTIES
    IMPORTED_LOCATION "${Kadath_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Kadath_INCLUDE_DIRS}")
endif()

message(STATUS "Kadath library: ${Kadath_LIBRARY}")
message(STATUS "Kadath include: ${Kadath_INCLUDE_DIRS}")