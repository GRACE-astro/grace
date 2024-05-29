
add_library( grace_error_handlers OBJECT abort.cc error.cc )
target_include_directories( grace_error_handlers PRIVATE "${HEADER_DIR}")
target_link_libraries(grace_error_handlers PRIVATE spdlog::spdlog p4est::sc MPI::MPI_CXX $<TARGET_OBJECTS:grace_parallel>)