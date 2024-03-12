set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")

add_subdirectory(errors)

add_subdirectory(parallel)

add_subdirectory(utils)

add_subdirectory(amr)

add_subdirectory(system)

add_subdirectory(data_structures)


add_library( thunder_objects INTERFACE )
target_sources( thunder_objects INTERFACE 
                $<TARGET_OBJECTS:thunder_error_handlers>
                $<TARGET_OBJECTS:thunder_parallel>
                $<TARGET_OBJECTS:thunder_singleton>
                $<TARGET_OBJECTS:thunder_runtime_utilities>
                $<TARGET_OBJECTS:thunder_connectivity> 
                $<TARGET_OBJECTS:thunder_forest>
                $<TARGET_OBJECTS:thunder_coordinates>
                $<TARGET_OBJECTS:thunder_init> 
                $<TARGET_OBJECTS:thunder_variable_list>
                $<TARGET_OBJECTS:thunder_variable_registration>)