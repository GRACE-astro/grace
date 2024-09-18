add_library( grace_connectivity OBJECT connectivity.cpp )
target_include_directories( grace_connectivity PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_connectivity PRIVATE 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est 
                       spdlog::spdlog )

add_library( grace_forest OBJECT forest.cpp )
target_include_directories( grace_forest PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_forest PRIVATE 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est 
                       spdlog::spdlog )

add_library( grace_amr_functions OBJECT amr_functions.cpp )
target_include_directories( grace_amr_functions PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_amr_functions PRIVATE 
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est  
                        spdlog::spdlog )


add_library( grace_regrid OBJECT regrid.cpp )
target_include_directories( grace_regrid PRIVATE "${HEADER_DIR}" )
target_compile_options(grace_regrid PRIVATE -Wno-dangling-field)
target_link_libraries( grace_regrid PRIVATE 
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos
                        spdlog::spdlog )

add_library( grace_BC_helpers OBJECT bc_iterate.cpp bc_copy_ghostzones.cpp bc_restrict_ghostzones.cpp bc_prolongate_ghostzones.cpp bc_helpers.cpp bc_fill_physical_ghostzones.cpp )
target_include_directories( grace_BC_helpers PRIVATE "${HEADER_DIR}" )
target_compile_options(grace_BC_helpers PRIVATE -Wno-dangling-field)
target_link_libraries( grace_BC_helpers PRIVATE 
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
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )