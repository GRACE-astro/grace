add_library(thunder_singleton OBJECT lifetime_tracker.cpp) 

target_include_directories(thunder_singleton PRIVATE  "${HEADER_DIR}")