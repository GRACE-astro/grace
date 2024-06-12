add_library( grace_cartesian_coordinates OBJECT cartesian_coordinate_system.cpp )
target_include_directories( grace_cartesian_coordinates PRIVATE "${HEADER_DIR}" )
target_compile_options(grace_cartesian_coordinates PRIVATE -Wno-dangling-field)
target_link_libraries( grace_cartesian_coordinates PRIVATE 
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )

add_library(spherical_cell_volume_kernels OBJECT cell_volume_kernels.cpp)
target_include_directories( spherical_cell_volume_kernels PRIVATE "${HEADER_DIR}" )

add_library(spherical_jacobians_2D OBJECT jacobians/Jacobians_sph_2D.cpp)
add_library(spherical_jacobians_3D OBJECT jacobians/Jacobians_sph_3D.cpp)
target_include_directories( spherical_jacobians_2D PRIVATE "${HEADER_DIR}" )
target_include_directories( spherical_jacobians_3D PRIVATE "${HEADER_DIR}" )
add_library(spherical_jacobians_inv_2D OBJECT jacobians/Jacobians_inverse_sph_2D.cpp)
add_library(spherical_jacobians_inv_3D OBJECT jacobians/Jacobians_inverse_sph_3D.cpp)
target_include_directories( spherical_jacobians_inv_2D PRIVATE "${HEADER_DIR}" )
target_include_directories( spherical_jacobians_inv_3D PRIVATE "${HEADER_DIR}" )
add_library(spherical_jacobians_inv_phys_2D OBJECT jacobians/Jacobians_inverse_phys_sph_2D.cpp)
add_library(spherical_jacobians_inv_phys_3D OBJECT jacobians/Jacobians_inverse_phys_sph_3D.cpp)
target_include_directories( spherical_jacobians_inv_phys_3D PRIVATE "${HEADER_DIR}" )
target_include_directories( spherical_jacobians_inv_phys_2D PRIVATE "${HEADER_DIR}" )
add_library(spherical_surface_element_2D OBJECT surface_elements/cell_surfaces_2D.cpp)
add_library(spherical_surface_element_3D OBJECT surface_elements/cell_surfaces_3D.cpp)
target_include_directories( spherical_surface_element_2D PRIVATE "${HEADER_DIR}" )
target_include_directories( spherical_surface_element_3D PRIVATE "${HEADER_DIR}" )
add_library(spherical_volume_element_3D OBJECT volume_elements/vol_sph_3D.cpp)
add_library(spherical_volume_element_3D_log OBJECT volume_elements/vol_sph_3D_log.cpp)
target_include_directories( spherical_volume_element_3D PRIVATE "${HEADER_DIR}" )
target_include_directories( spherical_volume_element_3D_log PRIVATE "${HEADER_DIR}" )
add_library(spherical_volume_element_3D_log_exact OBJECT volume_elements/dVol_sph_3D_log_analytic.cpp)
target_include_directories( spherical_volume_element_3D_log_exact PRIVATE "${HEADER_DIR}" )

                                       
add_library( grace_spherical_coordinates OBJECT spherical_coordinate_system.cpp )
target_include_directories( grace_spherical_coordinates PRIVATE "${HEADER_DIR}" )
target_compile_options(grace_spherical_coordinates PRIVATE -Wno-dangling-field)
target_link_libraries( grace_spherical_coordinates PRIVATE 
                        yaml_cpp::yaml
                        MPI::MPI_CXX
                        p4est::sc
                        p4est::p4est 
                        Kokkos::kokkos 
                        spdlog::spdlog )

add_library( grace_coordinates OBJECT coordinates.cpp )
target_include_directories( grace_coordinates PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_coordinates PRIVATE 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       OpenMP::OpenMP_CXX
                       p4est::sc
                       p4est::p4est 
                       Kokkos::kokkos 
                       spdlog::spdlog )

add_library( grace_coordinate_systems OBJECT coordinate_systems.cpp )
target_include_directories( grace_coordinate_systems PRIVATE "${HEADER_DIR}" )
target_link_libraries( grace_coordinate_systems PRIVATE 
                       yaml_cpp::yaml
                       MPI::MPI_CXX
                       p4est::sc
                       p4est::p4est 
                       Kokkos::kokkos 
                       spdlog::spdlog )