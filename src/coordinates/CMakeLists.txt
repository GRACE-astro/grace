add_library( thunder_cartesian_coordinates OBJECT cartesian_coordinate_system.cpp )
target_include_directories( thunder_cartesian_coordinates PRIVATE "${HEADER_DIR}" )
target_compile_options(thunder_cartesian_coordinates PRIVATE -Wno-dangling-field)
target_link_libraries( thunder_cartesian_coordinates PRIVATE 
                        $<TARGET_OBJECTS:thunder_parallel> 
                        $<TARGET_OBJECTS:thunder_singleton> 
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos )

add_library(spherical_cell_volume_kernels OBJECT cell_volume_kernels.cpp)
target_include_directories( spherical_cell_volume_kernels PRIVATE "${HEADER_DIR}" )

add_library( thunder_spherical_coordinates OBJECT spherical_coordinate_system.cpp )
target_include_directories( thunder_spherical_coordinates PRIVATE "${HEADER_DIR}" )
target_compile_options(thunder_spherical_coordinates PRIVATE -Wno-dangling-field)
target_link_libraries( thunder_spherical_coordinates PRIVATE 
                        $<TARGET_OBJECTS:thunder_parallel> 
                        $<TARGET_OBJECTS:spherical_cell_volume_kernels>
                        $<TARGET_OBJECTS:thunder_singleton> 
                        $<TARGET_OBJECTS:thunder_connectivity> 
                        $<TARGET_OBJECTS:thunder_forest>
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos )

add_library( thunder_coordinates OBJECT coordinates.cpp )
target_include_directories( thunder_coordinates PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_coordinates PRIVATE 
                       $<TARGET_OBJECTS:thunder_parallel> 
                       $<TARGET_OBJECTS:thunder_singleton> 
                       $<TARGET_OBJECTS:thunder_connectivity> 
                       $<TARGET_OBJECTS:thunder_forest>
                       $<TARGET_OBJECTS:thunder_amr_functions>
                       $<TARGET_OBJECTS:thunder_cartesian_coordinates>
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est 
                       Kokkos::kokkos )

add_library( thunder_coordinate_systems OBJECT coordinate_systems.cpp )
target_include_directories( thunder_coordinate_systems PRIVATE "${HEADER_DIR}" )
target_link_libraries( thunder_coordinate_systems PRIVATE 
                       $<TARGET_OBJECTS:thunder_parallel> 
                       $<TARGET_OBJECTS:thunder_singleton> 
                       $<TARGET_OBJECTS:thunder_connectivity> 
                       $<TARGET_OBJECTS:thunder_forest>
                       $<TARGET_OBJECTS:thunder_amr_functions>
                       $<TARGET_OBJECTS:thunder_cartesian_coordinates>
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est 
                       Kokkos::kokkos )