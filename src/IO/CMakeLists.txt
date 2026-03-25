# ------------------------------------------------------------
# src/IO/CMakeLists.txt
# ------------------------------------------------------------

# Conditionally register VTK output targets
register_grace_object_conditional(
    grace_vtk_output
    GRACE_ENABLE_VTK
    vtk_output.cpp
    vtk_setup_grid.cpp
    vtk_volume_output.cpp
    vtk_surface_output.cpp
)

register_grace_object_conditional(
    grace_output_aux
    GRACE_ENABLE_VTK
    vtk_output_auxiliaries.cpp
)

# ----------------------------------------------------------------
# HDF5 Output
# ----------------------------------------------------------------

# Register the hdf5 output object
register_grace_object(grace_hdf5_output hdf5_output.cpp)

# Ensure MPI and HDF5 are found
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# Add include directories for MPI and HDF5
target_include_directories(grace_hdf5_output
    PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${MPI_C_INCLUDE_PATH}
)

# Link libraries for HDF5 + MPI
target_link_libraries(grace_hdf5_output
    PRIVATE
        ${HDF5_LIBRARIES}
        MPI::MPI_C
)

# Define macro so HDF5 knows MPI is used
target_compile_definitions(grace_hdf5_output
    PRIVATE
        H5_USE_MPI
)

# ----------------------------------------------------------------
# Other output utilities
# ----------------------------------------------------------------
register_grace_object(grace_hdf5_surface_output hdf5_surface_output.cpp octree_search_class.cpp)
register_grace_object(grace_hdf5_sphere_output hdf5_sphere_output.cpp )

register_grace_object(grace_cell_output cell_output.cpp)
register_grace_object(grace_scalar_output scalar_output.cpp)

register_grace_object(grace_spherical_surfaces spherical_surfaces.cpp)

register_grace_object(grace_diagnostics output_diagnostics.cpp diagnostics/black_hole_diagnostics.cpp)
register_grace_object_conditional(grace_gw_integrals GRACE_ENABLE_Z4C_METRIC diagnostics/gw_integrals.cpp)


register_grace_object(xmf_utilities xmf_utilities.cpp)
