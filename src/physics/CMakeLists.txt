add_library(grace_scalar_adv OBJECT scalar_advection.cpp )
target_include_directories(  grace_scalar_adv PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_scalar_adv PRIVATE
                       yaml_cpp::yaml
                       MPI::MPI_CXX 
                       p4est::sc 
                       p4est::p4est 
                       Kokkos::kokkos 
                       spdlog::spdlog )
add_library(grace_burgers_eq OBJECT burgers_equation.cpp )
target_include_directories(  grace_burgers_eq PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_burgers_eq PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog ) 

add_library(grace_grmhd OBJECT grmhd.cpp admbase.cpp )
target_include_directories(  grace_grmhd PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_grmhd PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )    

add_subdirectory(eos)