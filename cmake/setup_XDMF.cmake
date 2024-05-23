if( NOT LIBXDMF_ROOT )
    set(LIBXDMF_ROOT "$ENV{LIBXDMF_ROOT}")
endif()
message(STATUS "Looking for XDMF in ${LIBXDMF_ROOT}")

find_package(XDMF PATHS "${LIBXDMF_ROOT}")
find_package(XDMF REQUIRED)