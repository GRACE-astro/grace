add_library( grace_regrid_helpers OBJECT evaluate_regrid_criterion.cpp)
target_include_directories( grace_regrid PRIVATE "${HEADER_DIR}" )
