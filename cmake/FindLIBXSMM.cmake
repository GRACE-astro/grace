
if (NOT LIBXSMM_ROOT)
    set(LIBXSMM_ROOT "")
    set(LIBXSMM_ROOT "$ENV{LIBXSMM_ROOT}")
endif()

find_path( libxsmm_INCLUDE_DIR
    NAMES libxsmm.h 
    PATHS "${LIBXSMM_ROOT}"
    PATH_SUFFIXES include )

find_library( libxsmm_LIBRARY
    NAMES xsmm
    PATHS "${LIBXSMM_ROOT}"
    PATH_SUFFIXES lib lib64 )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBXSMM
      FOUND_VAR LIBXSMM_FOUND
      REQUIRED_VARS
        libxsmm_LIBRARY
        libxsmm_INCLUDE_DIR
)

if(LIBXSMM_FOUND)
    set(libxsmm_LIBRARIES ${libxsmm_LIBRARY})
    set(libxsmm_INCLUDE_DIRS ${libxsmm_INCLUDE_DIR})
endif()

mark_as_advanced(libxsmm_INCLUDE_DIR
                 libxsmm_LIBRARY)