
if(NOT YAML_ROOT )
set(YAML_ROOT "")
set(YAML_ROOT "$ENV{YAML_ROOT}")
endif()

message(STATUS "Searching path ${YAML_ROOT}")

find_package(yaml-cpp REQUIRED PATHS "${YAML_ROOT}")

if(TARGET yaml-cpp)
    message(STATUS "Using modern yaml-cpp target")
    add_library(yaml_cpp::yaml ALIAS yaml-cpp)
else()
    message(WARNING "Using legacy yaml-cpp variables")
    if(NOT DEFINED YAML_CPP_LIBRARIES OR NOT DEFINED YAML_CPP_INCLUDE_DIRS)
        message(FATAL_ERROR "Legacy yaml-cpp variables are not defined. Please check your installation.")
    endif()

    if(NOT TARGET yaml_cpp::yaml)
        add_library(yaml_cpp::yaml IMPORTED INTERFACE)
        set_property(TARGET yaml_cpp::yaml APPEND PROPERTY
                     INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIRS}")
        set_property(TARGET yaml_cpp::yaml APPEND PROPERTY
                     INTERFACE_LINK_LIBRARIES "${YAML_CPP_LIBRARIES}")
    endif()
endif()
