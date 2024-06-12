add_library(grace_variable_list OBJECT variables.cpp)

target_include_directories( grace_variable_list PRIVATE "${HEADER_DIR}" )
target_link_libraries(  grace_variable_list PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )


add_library(grace_variable_registration OBJECT variable_indices.cpp)

target_include_directories( grace_variable_registration PRIVATE "${HEADER_DIR}" )
target_link_libraries(  grace_variable_registration PRIVATE
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )


add_library(grace_variable_utils OBJECT variable_utils.cpp)

target_include_directories( grace_variable_utils PRIVATE "${HEADER_DIR}" )
target_link_libraries(  grace_variable_utils PRIVATE
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )

