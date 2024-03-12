

find_package(BLAS REQUIRED)

message(STATUS "Blas libraries: ${BLAS_LIBRARIES}")

if( NOT TARGET Blas::Blas)
    add_library(Blas::Blas IMPORTED INTERFACE)
    set_property(TARGET Blas::Blas
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}" )
endif()


