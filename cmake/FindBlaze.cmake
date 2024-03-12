
if (NOT BLAZE_ROOT)
    set(BLAZE_ROOT "")
    set(BLAZE_ROOT "$ENV{BLAZE_ROOT}")
endif()

if (BLAZE_ROOT STREQUAL "")
    set(BLAZE_ROOT "${CMAKE_SOURCE_DIR}/extern/blaze")
endif()

find_path( BLAZE_INCLUDE_DIR
    NAMES blaze/Blaze.h 
    PATHS "${BLAZE_ROOT}"
    PATH_SUFFIXES include)

set(BLAZE_INCLUDE_DIRS ${BLAZE_INCLUDE_DIR})
message(STATUS "BLAZE include: ${BLAZE_INCLUDE_DIRS}")
set(BLAZE_VERSION "")

if(EXISTS "${BLAZE_INCLUDE_DIRS}/blaze/system/Version.h")
  # Extract version info from header
  file(READ
    "${BLAZE_INCLUDE_DIRS}/blaze/system/Version.h"
    BLAZE_FIND_HEADER_CONTENTS)

  string(REGEX MATCH "#define BLAZE_MAJOR_VERSION [0-9]+"
    BLAZE_MAJOR_VERSION "${BLAZE_FIND_HEADER_CONTENTS}")
  string(REPLACE "#define BLAZE_MAJOR_VERSION " ""
    BLAZE_MAJOR_VERSION
    ${BLAZE_MAJOR_VERSION})

  string(REGEX MATCH "#define BLAZE_MINOR_VERSION [0-9]+"
    BLAZE_MINOR_VERSION "${BLAZE_FIND_HEADER_CONTENTS}")
  string(REPLACE "#define BLAZE_MINOR_VERSION " ""
    BLAZE_MINOR_VERSION
    ${BLAZE_MINOR_VERSION})

  set(BLAZE_VERSION
    "${BLAZE_MAJOR_VERSION}.${BLAZE_MINOR_VERSION}"
    )
else()
  message(WARNING "Failed to find file "
    "'${BLAZE_INCLUDE_DIRS}/blaze/system/Version.h' "
    "while detecting the Blaze version.")
endif(EXISTS "${BLAZE_INCLUDE_DIRS}/blaze/system/Version.h")

set(Blaze_VERSION ${BLAZE_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Blaze
  FOUND_VAR BLAZE_FOUND
  REQUIRED_VARS BLAZE_INCLUDE_DIR BLAZE_INCLUDE_DIRS
  VERSION_VAR BLAZE_VERSION
  )
mark_as_advanced(BLAZE_INCLUDE_DIR BLAZE_INCLUDE_DIRS
  BLAZE_VERSION BLAZE_MAJOR_VERSION BLAZE_MINOR_VERSION
  Blaze_VERSION
)