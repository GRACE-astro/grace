#[=======================================================================[.rst:
Findp4est
-------

Finds the p4est library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``p4est::p4est``
  The p4est library
``p4est::sc``
  The sc library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``p4est_FOUND``
  True if the system has the p4est library.
``p4est_INCLUDE_DIRS``
  Include directories needed to use p4est.
``p4est_LIBRARIES``
  Libraries needed to link to p4est.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``p4est_INCLUDE_DIR``
  The directory containing ``p4est.h``.
``p4est_LIBRARY``
  The path to the p4est library.

#]=======================================================================]

if(NOT P4EST_ROOT)
    set(P4EST_ROOT "")
    set(P4EST_ROOT "$ENV{P4EST_ROOT}") 
endif()
message(STATUS "Looking for p4est in ${P4EST_ROOT}")
find_path( p4est_INCLUDE_DIR
    NAMES p4est.h 
    PATHS ${P4EST_ROOT}
    PATH_SUFFIXES local/include include 
)

find_library( p4est_LIBRARY
    NAMES p4est
    PATHS ${P4EST_ROOT} ${CMAKE_SOURCE_DIR}/extern/p4est
    PATH_SUFFIXES local/lib lib
)

find_library( sc_LIBRARY
    NAMES sc
    PATHS ${P4EST_ROOT} ${CMAKE_SOURCE_DIR}/extern/p4est
    PATH_SUFFIXES local/lib lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(p4est
  FOUND_VAR p4est_FOUND
  REQUIRED_VARS
    p4est_LIBRARY
    sc_LIBRARY
    p4est_INCLUDE_DIR
)

if(p4est_FOUND)
    set(p4est_LIBRARIES "${p4est_LIBRARY}")
    set(sc_LIBRARIES "${sc_LIBRARY}")
    set(p4est_INCLUDE_DIRS "${p4est_INCLUDE_DIR}")
    set(sc_INCLUDE_DIRS "${p4est_INCLUDE_DIR}")
endif()

if(p4est_FOUND AND NOT TARGET p4est::p4est)
    add_library(p4est::p4est UNKNOWN IMPORTED)
    set_target_properties(p4est::p4est PROPERTIES
    IMPORTED_LOCATION "${p4est_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${p4est_INCLUDE_DIRS}")
endif()

if(p4est_FOUND AND NOT TARGET p4est::sc)
    add_library(p4est::sc UNKNOWN IMPORTED)
    set_target_properties(p4est::sc PROPERTIES
    IMPORTED_LOCATION "${sc_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${p4est_INCLUDE_DIRS}")
endif()

mark_as_advanced(
    p4est_INCLUDE_DIR
    p4est_LIBRARY
    sc_LIBRARY
)

message(STATUS "p4est libs: ${p4est_LIBRARIES}")
message(STATUS "sc libs: ${sc_LIBRARIES}")
message(STATUS "p4est include: ${p4est_INCLUDE_DIRS}")