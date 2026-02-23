
option( GRACE_CARTESIAN_COORDINATES "Build the code with cartesian coordinates" ON) 
message(STATUS  "Cartesian cordinate system enabled.")

set(GRACE_NSPACEDIM 3)
message(STATUS "Space dimension (GRACE_NSPACEDIM) is 3.")
set( GRACE_3D ON )

#add_compile_options(
#    $<$<CONFIG:DEBUG>:-O0>
#    $<$<CONFIG:DEBUG>:-gdwarf-4>
#    $<$<CONFIG:RELWITHDEBINFO>:-gdwarf-4>
#    $<$<CONFIG:RELEASE>:-O3>
#    -Wno-deprecated-declarations
#)

add_compile_definitions(
    $<$<CONFIG:DEBUG>:GRACE_DEBUG>
)

if(GRACE_3D)
    add_compile_definitions(
        P4_TO_P8
    )
endif() 
