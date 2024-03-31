add_library(thunder_singleton OBJECT lifetime_tracker.cpp) 
target_include_directories(thunder_singleton PRIVATE  "${HEADER_DIR}")

add_library(thunder_device_lowlevel OBJECT device.cpp)
target_include_directories(thunder_device_lowlevel PRIVATE  "${HEADER_DIR}" )