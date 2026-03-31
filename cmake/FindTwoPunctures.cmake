#[=======================================================================[.rst:
FindTwoPunctures
-------

Finds the TwoPunctures library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``TwoPunctures::TwoPunctures``
  The TwoPunctures library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``TwoPunctures_FOUND``
  True if the system has the TwoPunctures library.
``TwoPunctures_INCLUDE_DIRS``
  Include directories needed to use TwoPunctures.
``TwoPunctures_LIBRARIES``
  Libraries needed to link to TwoPunctures.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``TwoPunctures_INCLUDE_DIR``
  The directory containing TwoPunctures headers.
``TwoPunctures_LIBRARY``
  The path to the TwoPunctures library.

#]=======================================================================]
if (NOT TwoPunctures_ROOT)
    set(TwoPunctures_ROOT "")
    set(TwoPunctures_ROOT "$ENV{TwoPunctures_ROOT}")
endif()

message("Looking for TwoPunctures in ${TwoPunctures_ROOT}")

find_path(
    TwoPunctures_INCLUDE_DIR
    NAMES TwoPunctures.h
    PATHS ${TwoPunctures_ROOT}
    PATH_SUFFIXES Source
)

find_library( TwoPunctures_LIBRARY
    NAMES twopunctures
    PATHS ${TwoPunctures_ROOT}
    PATH_SUFFIXES Source
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TwoPunctures
  FOUND_VAR TwoPunctures_FOUND
  REQUIRED_VARS
    TwoPunctures_LIBRARY
    TwoPunctures_INCLUDE_DIR
)

find_package(GSL REQUIRED)

if (TwoPunctures_FOUND)
    set( TwoPunctures_LIBRARIES ${TwoPunctures_LIBRARY})
    set( TwoPunctures_INCLUDE_DIRS "${TwoPunctures_INCLUDE_DIR}")
endif() 
message(STATUS "TwoPunctures libs: ${TwoPunctures_LIBRARIES}")
message(STATUS "TwoPunctures include: ${TwoPunctures_INCLUDE_DIRS}")

if (TwoPunctures_FOUND AND NOT TARGET TwoPunctures::TwoPunctures)
    add_library(TwoPunctures::TwoPunctures STATIC IMPORTED)
    set_target_properties(TwoPunctures::TwoPunctures PROPERTIES
        IMPORTED_LOCATION "${TwoPunctures_LIBRARY}"   # libTwoPunctures.a
    )
    target_include_directories(TwoPunctures::TwoPunctures INTERFACE
      ${TwoPunctures_INCLUDE_DIR}
    )
    target_link_libraries(TwoPunctures::TwoPunctures INTERFACE
        GSL::gsl
    )
endif() 

mark_as_advanced(
    TwoPunctures_INCLUDE_DIR 
    TwoPunctures_LIBRARY
)
