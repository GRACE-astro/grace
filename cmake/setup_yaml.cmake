
if(NOT YAML_ROOT )
set(YAML_ROOT "")
set(YAML_ROOT "$ENV{YAML_ROOT}")
endif()

message(STATUS "Searching path ${YAML_ROOT}")

find_package( yaml-cpp REQUIRED PATHS "${YAML_ROOT}" ) 

message(STATUS "yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")
message(STATUS "yaml-cpp includes: ${YAML_CPP_INCLUDE_DIR}")

if( NOT TARGET yaml_cpp::yaml )
    add_library( yaml_cpp::yaml IMPORTED INTERFACE )

    set_property(TARGET yaml_cpp::yaml APPEND PROPERTY 
                 INTERFACE_INCLUDE_DIRECTORIES  "${YAML_CPP_INCLUDE_DIRS}")
    set_property(TARGET yaml_cpp::yaml APPEND PROPERTY 
                 INTERFACE_LINK_LIBRARIES  "${YAML_CPP_LIBRARIES}")
endif()