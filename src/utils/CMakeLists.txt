add_library(grace_singleton OBJECT lifetime_tracker.cpp) 
target_include_directories(grace_singleton PRIVATE  "${HEADER_DIR}")
