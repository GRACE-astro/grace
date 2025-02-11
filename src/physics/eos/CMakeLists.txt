add_library(grace_eos OBJECT eos_storage.cpp )
target_include_directories(  grace_eos PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_eos PRIVATE
                       yaml_cpp::yaml
                       MPI::MPI_CXX 
                       p4est::sc 
                       p4est::p4est 
                       Kokkos::kokkos 
                       spdlog::spdlog )
add_library(grace_c2p OBJECT c2p.cpp )
target_include_directories(  grace_c2p PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_c2p PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )
add_library(grace_tab OBJECT tabulated_reader_compose.cpp )
target_include_directories(  grace_tab PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_tab PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )