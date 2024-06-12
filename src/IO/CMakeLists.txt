add_library(grace_vtk_output OBJECT vtk_output.cpp vtk_setup_grid.cpp vtk_volume_output.cpp vtk_surface_output.cpp scalar_output.cpp)
target_include_directories( grace_vtk_output PRIVATE "${HEADER_DIR}" )
target_link_libraries(  grace_vtk_output PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        VTK::VTK  
                        spdlog::spdlog )

vtk_module_autoinit(
    TARGETS grace_vtk_output
    MODULES ${VTK_LIBRARIES}
    )

add_library(grace_hdf5_output OBJECT hdf5_output.cpp xmf_utilities.cpp )
target_include_directories( grace_hdf5_output PRIVATE "${HEADER_DIR}" )
target_link_libraries(  grace_hdf5_output PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        VTK::VTK  
                        HDF5::HDF5 
                        spdlog::spdlog )
vtk_module_autoinit(
    TARGETS grace_hdf5_output
    MODULES ${VTK_LIBRARIES}
    )

add_library(grace_cell_output OBJECT cell_output.cpp)
target_include_directories( grace_cell_output PRIVATE "${HEADER_DIR}" )
target_link_libraries(  grace_cell_output PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc 
                        p4est::p4est 
                        Kokkos::kokkos 
                        VTK::VTK  
                        HDF5::HDF5 
                        OpenMP::OpenMP_CXX
                        spdlog::spdlog )
vtk_module_autoinit(
    TARGETS grace_cell_output
    MODULES ${VTK_LIBRARIES}
    )


add_library(grace_output_aux OBJECT vtk_output_auxiliaries.cpp )
target_include_directories( grace_output_aux PRIVATE "${HEADER_DIR}" )
target_link_libraries(  grace_output_aux PRIVATE
                        yaml_cpp::yaml
                        MPI::MPI_CXX 
                        p4est::sc
                        Kokkos::kokkos
                        spdlog::spdlog )