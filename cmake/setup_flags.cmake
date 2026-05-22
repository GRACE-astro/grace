
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

# NOTE: `P4_TO_P8` is intentionally NOT added here.  It's a header-toggle
# macro that p4est uses INTERNALLY when compiling its 3D source files —
# if it's globally defined when p4est itself is being built (which
# happens under GRACE_USE_BUNDLED_DEPS=ON via add_subdirectory), p4est's
# 2D source files break with "Including a p4est header with P4_TO_P8
# defined".  We add the definition to GRACE's own targets in
# CMakeLists.txt AFTER `include(setup_p4est)` so the subdirectory build
# is unaffected.  See the matching block at the bottom of CMakeLists.txt.
