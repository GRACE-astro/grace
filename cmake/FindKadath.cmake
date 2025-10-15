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

if(NOT HOME_KADATH)
    set(HOME_KADATH "$ENV{HOME_KADATH}")
endif()
message(STATUS "Looking for Kadath in ${HOME_KADATH}")

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

# Imported target
if(Kadath_FOUND AND NOT TARGET Kadath::kadath)
    add_library(Kadath::kadath UNKNOWN IMPORTED)
    set_target_properties(Kadath::kadath PROPERTIES
        IMPORTED_LOCATION "${Kadath_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Kadath_INCLUDE_DIRS}"
    )
endif()

message(STATUS "Kadath library: ${Kadath_LIBRARY}")
message(STATUS "Kadath include dirs: ${Kadath_INCLUDE_DIRS}")

