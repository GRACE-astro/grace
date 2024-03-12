
find_package(Blaze 3.9 EXACT REQUIRED)

message(STATUS "Blaze include: ${BLAZE_INCLUDE_DIR}")

if( NOT TARGET Blaze::Blaze )
    add_library(Blaze::Blaze IMPORTED INTERFACE)
    set_property(TARGET Blaze::Blaze PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${BLAZE_INCLUDE_DIR})   
endif()

target_link_libraries(
    Blaze::Blaze
    INTERFACE
    Blas::Blas 
    GSL::gsl 
    Lapack::Lapack
)

target_compile_definitions(Blaze::Blaze
    INTERFACE

    BLAZE_BLAS_MODE=1

    BLAZE_BLAS_INCLUDE_FILE=<gsl/gsl_cblas.h>

    BLAZE_DEFAULT_STORAGE_ORDER=blaze::rowMajor

    BLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0

    BLAZE_MPI_PARALLEL_MODE=0

    BLAZE_USE_PADDING=0

    BLAZE_USE_STREAMING=1

    BLAZE_USE_DEFAULT_INITIALIZATION=0

    BLAZE_USE_SLEEF=0

    BLAZE_USE_STRONG_INLINE=1
    BLAZE_USE_ALWAYS_INLINE=1

)

