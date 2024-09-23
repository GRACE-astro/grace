add_library( grace_objects INTERFACE )

include(register_grace_object)

add_subdirectory(parallel)

add_subdirectory(errors)

add_subdirectory(utils)

add_subdirectory(amr)

add_subdirectory(coordinates)

add_subdirectory(system)

add_subdirectory(data_structures)

add_subdirectory(IO)

if( GRACE_ENABLE_PROFILING )
 add_subdirectory(profiling)
endif()

add_subdirectory(evolution)

add_subdirectory(physics)


