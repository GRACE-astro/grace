
option( THUNDER_CARTESIAN_COORDINATES "Build the code with cartesian coordinates" OFF) 
option( THUNDER_SPHERICAL_COORDINATES "Build the code with spherical coordinates" ON) 

if( NOT THUNDER_NSPACEDIM )
    set(THUNDER_NSPACEDIM 2)
    message(STATUS "Space dimension (THUNDER_NSPACEDIM) not set, default is 2.")
endif() 

if( THUNDER_NSPACEDIM EQUAL 3 )
    set( THUNDER_3D ON )
endif()

add_compile_options(
    $<$<CONFIG:DEBUG>:-O0>
    $<$<CONFIG:RELEASE>:-O3>
)

set(CMAKE_CXX_FLAGS "-g3 -ldl")

add_compile_definitions(
    $<$<CONFIG:DEBUG>:THUNDER_DEBUG>
)

if(THUNDER_3D)
    add_compile_definitions(
        P4_TO_P8
    )
endif() 
