
if(NOT SPDLOG_ROOT )
set(SPDLOG_ROOT "")
set(SPDLOG_ROOT "$ENV{SPDLOG_ROOT}")
endif()

message(STATUS "Searching path ${SPDLOG_ROOT}")

find_package( spdlog REQUIRED PATHS "${SPDLOG_ROOT}" ) 

message(STATUS "spdlog libraries: ${SPDLOG_LIBRARIES}")
message(STATUS "spdlog includes: ${SPDLOG_INCLUDE_DIR}")

if( NOT TARGET spdlog::spdlog )
    add_library( spdlog::spdlog IMPORTED INTERFACE )

    set_property(TARGET spdlog::spdlog APPEND PROPERTY 
                 INTERFACE_INCLUDE_DIRECTORIES  "${SPDLOG_INCLUDE_DIRS}")
    set_property(TARGET spdlog::spdlog APPEND PROPERTY 
                 INTERFACE_LINK_LIBRARIES  "${SPDLOG_LIBRARIES}")
endif()