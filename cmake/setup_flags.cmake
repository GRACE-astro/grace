
option( GRACE_CARTESIAN_COORDINATES "Build the code with cartesian coordinates" ON) 
option( GRACE_SPHERICAL_COORDINATES "Build the code with spherical coordinates" OFF) 

if( GRACE_SPHERICAL_COORDINATES )
    message(STATUS  "Spherical cordinate system enabled.")
    if( GRACE_CARTESIAN_COORDINATES)
        set(GRACE_CARTESIAN_COORDINATES OFF)
        message(STATUS  "Switching off Cartesian coordinate system.")
    endif()
endif() 

if( GRACE_CARTESIAN_COORDINATES )
    message(STATUS  "Cartesian cordinate system enabled.")
    if( GRACE_SPHERICAL_COORDINATES)
        set(GRACE_SPHERICAL_COORDINATES OFF)
        message(STATUS  "Switching off spherical coordinate system.")
    endif()
endif() 

if( NOT GRACE_NSPACEDIM )
    set(GRACE_NSPACEDIM 2)
    message(STATUS "Space dimension (GRACE_NSPACEDIM) not set, default is 2.")
endif() 

if( GRACE_NSPACEDIM EQUAL 3 )
    set( GRACE_3D ON )
endif()

add_compile_options(
    $<$<CONFIG:DEBUG>:-O0>
    $<$<CONFIG:DEBUG>:-gdwarf-4>
    $<$<CONFIG:RELWITHDEBINFO>:-gdwarf-4>
    $<$<CONFIG:RELEASE>:-O3>
    -Wno-deprecated-declarations
)

add_compile_definitions(
    $<$<CONFIG:DEBUG>:GRACE_DEBUG>
)

if(GRACE_3D)
    add_compile_definitions(
        P4_TO_P8
    )
endif() 
