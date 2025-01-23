set(HEADER_DIR "${PROJECT_SOURCE_DIR}/include/")

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

add_library( grace_objects INTERFACE )
target_sources( grace_objects INTERFACE 
                $<TARGET_OBJECTS:grace_error_handlers>
                #$<TARGET_OBJECTS:grace_device_lowlevel>
                $<TARGET_OBJECTS:grace_parallel>
                $<TARGET_OBJECTS:grace_singleton>
                $<TARGET_OBJECTS:grace_runtime_utilities>
                $<TARGET_OBJECTS:grace_connectivity> 
                $<TARGET_OBJECTS:grace_forest>
                $<TARGET_OBJECTS:grace_amr_functions>
                $<TARGET_OBJECTS:spherical_cell_volume_kernels>
                $<TARGET_OBJECTS:spherical_jacobians_2D>
                $<TARGET_OBJECTS:spherical_jacobians_3D>
                $<TARGET_OBJECTS:spherical_jacobians_inv_2D>
                $<TARGET_OBJECTS:spherical_jacobians_inv_3D>
                $<TARGET_OBJECTS:spherical_jacobians_inv_phys_2D>
                $<TARGET_OBJECTS:spherical_jacobians_inv_phys_3D>
                $<TARGET_OBJECTS:spherical_surface_element_2D>
                $<TARGET_OBJECTS:spherical_surface_element_3D>
                $<TARGET_OBJECTS:spherical_volume_element_3D>
                $<TARGET_OBJECTS:spherical_volume_element_3D_log>
                $<TARGET_OBJECTS:spherical_volume_element_3D_log_exact>
                $<TARGET_OBJECTS:grace_coordinates>
                $<TARGET_OBJECTS:grace_init> 
                $<TARGET_OBJECTS:grace_finalize>
                $<TARGET_OBJECTS:grace_variable_list>
                $<TARGET_OBJECTS:grace_variable_registration>
                $<TARGET_OBJECTS:grace_variable_utils>
		$<TARGET_OBJECTS:grace_vtk_output>
		$<TARGET_OBJECTS:grace_hdf5_output>
		$<TARGET_OBJECTS:grace_cell_output>
                $<TARGET_OBJECTS:grace_output_aux>
                $<TARGET_OBJECTS:grace_cartesian_coordinates>
                $<TARGET_OBJECTS:grace_spherical_coordinates>
                $<TARGET_OBJECTS:grace_coordinate_systems>
                $<TARGET_OBJECTS:grace_regrid> 
                $<TARGET_OBJECTS:grace_BC_helpers>
                $<TARGET_OBJECTS:grace_BC>
                $<TARGET_OBJECTS:grace_scalar_adv>
                $<TARGET_OBJECTS:grace_burgers_eq>
                $<TARGET_OBJECTS:grace_grmhd>
                $<TARGET_OBJECTS:grace_eos>
                $<TARGET_OBJECTS:grace_evolution> 
                $<TARGET_OBJECTS:grace_initial_data> 
                $<TARGET_OBJECTS:grace_c2p>
                $<$<BOOL:${GRACE_ENABLE_PROFILING}>:$<TARGET_OBJECTS:grace_profiling>>)

