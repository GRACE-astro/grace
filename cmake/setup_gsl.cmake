
find_package(GSL REQUIRED)

if(NOT TARGET GSL::gsl)
    add_library(GSL::gsl IMPORTED INTERFACE)
    set_target_properties(GSL::gsl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GSL_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES ${GSL_LINK_FLAGS} "${GSL_LIBRARIES}"
        INTERFACE_COMPILE_OPTIONS "${GSL_COMPILE_FLAGS}")
endif()

target_link_libraries(GSL::gsl
                      INTERFACE
                      Blas::Blas 
                      )

