
find_package(LAPACK REQUIRED)

message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")

if(NOT TARGET Lapack::Lapack)
    add_library(Lapack::Lapack IMPORTED INTERFACE)

    set_property( TARGET Lapack::Lapack PROPERTY
        INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES}
        )
    set_property( TARGET Lapack::Lapack PROPERTY    
        INTERFACE_LINK_OPTIONS ${LAPACK_LINKER_FLAGS}
        )
endif()
