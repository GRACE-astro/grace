
add_library( thunder_error_handlers OBJECT abort.cc error.cc )
target_include_directories( thunder_error_handlers PRIVATE "${HEADER_DIR}")
target_link_libraries(thunder_error_handlers PRIVATE spdlog::spdlog p4est::sc MPI::MPI_CXX $<TARGET_OBJECTS:thunder_parallel>)