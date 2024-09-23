add_subdirectory(regrid_helpers)
add_subdirectory(bc_helpers)

register_grace_object(grace_connectivity connectivity.cpp)
register_grace_object(grace_forest forest.cpp)
register_grace_object(grace_amr_functions amr_functions.cpp)
register_grace_object(grace_regrid regrid.cpp)
register_grace_object(grace_BC boundary_conditions.cpp)



