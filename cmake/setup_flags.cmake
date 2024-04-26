
option( THUNDER_CARTESIAN_COORDINATES "Build the code with cartesian coordinates" ON) 
option( THUNDER_SPHERICAL_COORDINATES "Build the code with spherical coordinates" OFF) 

if( THUNDER_SPHERICAL_COORDINATES )
    message(STATUS  "Spherical cordinate system enabled.")
    if( THUNDER_CARTESIAN_COORDINATES)
        set(THUNDER_CARTESIAN_COORDINATES OFF)
        message(STATUS  "Switching off Cartesian coordinate system.")
    endif()
endif() 

if( THUNDER_CARTESIAN_COORDINATES )
    message(STATUS  "Cartesian cordinate system enabled.")
    if( THUNDER_SPHERICAL_COORDINATES)
        set(THUNDER_SPHERICAL_COORDINATES OFF)
        message(STATUS  "Switching off spherical coordinate system.")
    endif()
endif() 

if( NOT THUNDER_NSPACEDIM )
    set(THUNDER_NSPACEDIM 2)
    message(STATUS "Space dimension (THUNDER_NSPACEDIM) not set, default is 2.")
endif() 

if( THUNDER_NSPACEDIM EQUAL 3 )
    set( THUNDER_3D ON )
endif()

add_compile_options(
    $<$<CONFIG:DEBUG>:-O0>
    $<$<CONFIG:DEBUG>:-gdwarf-4>
    $<$<CONFIG:RELWITHDEBINFO>:-gdwarf-4>
    $<$<CONFIG:RELEASE>:-O3>
    -Wno-deprecated-declarations
)

add_compile_definitions(
    $<$<CONFIG:DEBUG>:THUNDER_DEBUG>
)

if(THUNDER_3D)
    add_compile_definitions(
        P4_TO_P8
    )
endif() 
