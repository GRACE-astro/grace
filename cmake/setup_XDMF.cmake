if( NOT LIBXDMF_ROOT )
    set(LIBXDMF_ROOT "$ENV{LIBXDMF_ROOT}")
endif()
message(STATUS "Looking for XDMF in ${LIBXDMF_ROOT}")

find_package(Xdmf PATHS "${LIBXDMF_ROOT}" PATH_SUFFIXES lib/cmake)
find_package(Xdmf REQUIRED)

message(STATUS "LIBXDMF libraries: ${XDMF_LIBRARIES}")
message(STATUS "LIBXDMF include: ${XDMF_INCLUDE_DIRS}")
message(STATUS "LIBXDMF include: ${XDMF_PC_LIBXML_INCLUDE_DIRS}")

if( NOT TARGET XDMF::XDMF)
    add_library(XDMF::XDMF IMPORTED INTERFACE )
    set_property(TARGET XDMF::XDMF PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${XDMF_INCLUDE_DIRS}"
        )
    set_property(TARGET XDMF::XDMF APPEND PROPERTY    
        INTERFACE_LINK_LIBRARIES ${LIBXDMF_ROOT}/lib/libXdmf.so.3 ${LIBXDMF_ROOT}/lib/libXdmfCore.so.3
        )
endif()

