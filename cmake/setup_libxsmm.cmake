

find_package(LIBXSMM REQUIRED)

message(STATUS "libxsmm libraries: ${libxsmm_LIBRARIES}")
message(STATUS "libxsmm include: ${libxsmm_INCLUDE_DIRS}")

if(NOT TARGET Libxsmm::Libxsmm )
    add_library(Libxsmm::Libxsmm IMPORTED INTERFACE )
    set_property(TARGET Libxsmm::Libxsmm PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${libxsmm_INCLUDE_DIRS}"
        )
    set_property(TARGET Libxsmm::Libxsmm APPEND PROPERTY    
        INTERFACE_LINK_LIBRARIES ${libxsmm_LIBRARIES}
        )
    set_property(TARGET Libxsmm::Libxsmm APPEND PROPERTY    
        INTERFACE_LINK_LIBRARIES Blas::Blas
        )
endif()