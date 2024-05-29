add_library(grace_singleton OBJECT lifetime_tracker.cpp) 
target_include_directories(grace_singleton PRIVATE  "${HEADER_DIR}")

add_library(grace_device_lowlevel OBJECT device.cpp)
target_include_directories(grace_device_lowlevel PRIVATE  "${HEADER_DIR}" )