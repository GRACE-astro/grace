add_library( grace_connectivity OBJECT connectivity.cpp )
target_include_directories( grace_connectivity PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_connectivity PRIVATE 
                       $<TARGET_OBJECTS:grace_parallel> 
                       $<TARGET_OBJECTS:grace_singleton> 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est 
                       spdlog::spdlog )

add_library( grace_forest OBJECT forest.cpp )
target_include_directories( grace_forest PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_forest PRIVATE 
                       $<TARGET_OBJECTS:grace_parallel> 
                       $<TARGET_OBJECTS:grace_singleton> 
                       $<TARGET_OBJECTS:grace_connectivity> 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est 
                       spdlog::spdlog )

add_library( grace_amr_functions OBJECT amr_functions.cpp )
target_include_directories( grace_amr_functions PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_amr_functions PRIVATE 
                        $<TARGET_OBJECTS:grace_parallel> 
                        $<TARGET_OBJECTS:grace_singleton> 
                        $<TARGET_OBJECTS:grace_connectivity> 
                        $<TARGET_OBJECTS:grace_forest>
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est  
                        spdlog::spdlog )


add_library( grace_regrid OBJECT regrid.cpp )
target_include_directories( grace_regrid PRIVATE "${HEADER_DIR}" )
target_compile_options(grace_regrid PRIVATE -Wno-dangling-field)
target_link_libraries( grace_regrid PRIVATE 
                        $<TARGET_OBJECTS:grace_parallel> 
                        $<TARGET_OBJECTS:grace_singleton> 
                        $<TARGET_OBJECTS:grace_connectivity> 
                        $<TARGET_OBJECTS:grace_forest>
                        $<TARGET_OBJECTS:grace_amr_functions>
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos
                        spdlog::spdlog )

add_library( grace_BC_helpers OBJECT bc_helpers.cpp )
target_include_directories( grace_BC_helpers PRIVATE "${HEADER_DIR}" )
target_compile_options(grace_BC_helpers PRIVATE -Wno-dangling-field)
target_link_libraries( grace_BC_helpers PRIVATE 
                        $<TARGET_OBJECTS:grace_parallel> 
                        $<TARGET_OBJECTS:grace_singleton> 
                        $<TARGET_OBJECTS:grace_connectivity> 
                        $<TARGET_OBJECTS:grace_forest>
                        $<TARGET_OBJECTS:grace_amr_functions>
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )

add_library( grace_BC OBJECT boundary_conditions.cpp )
target_include_directories( grace_BC PRIVATE "${HEADER_DIR}" )
target_compile_options(grace_BC PRIVATE -Wno-dangling-field)
target_link_libraries( grace_BC PRIVATE 
                        $<TARGET_OBJECTS:grace_parallel> 
                        $<TARGET_OBJECTS:grace_singleton> 
                        $<TARGET_OBJECTS:grace_connectivity> 
                        $<TARGET_OBJECTS:grace_forest>
                        $<TARGET_OBJECTS:grace_amr_functions>
                        $<TARGET_OBJECTS:grace_BC_helpers>
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )