add_library( grace_evolution OBJECT auxiliaries.cpp evolve.cpp find_stable_timestep.cpp )
target_include_directories( grace_evolution PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_evolution PRIVATE 
                       yaml_cpp::yaml
                       MPI::MPI_CXX 
                       p4est::sc 
                       p4est::p4est 
                       Kokkos::kokkos 
                       spdlog::spdlog )
add_library( grace_initial_data OBJECT initial_data.cpp )
target_include_directories( grace_initial_data PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_initial_data PRIVATE 
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )