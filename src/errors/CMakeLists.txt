
add_library( thunder_error_handlers OBJECT abort.cc error.cc )
target_include_directories( thunder_error_handlers PRIVATE "${HEADER_DIR}")