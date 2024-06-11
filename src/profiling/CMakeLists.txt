add_library(grace_profiling OBJECT gpu_profiling.cpp)
target_include_directories(grace_profiling PRIVATE "${HEADER_DIR}")
target_link_libraries( grace_profiling PRIVATE
                       yaml_cpp::yaml 
                       MPI::MPI_CXX 
                       p4est::sc 
                       Kokkos::kokkos 
                       spdlog::spdlog )