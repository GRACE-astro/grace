if( NOT VTK_ROOT) 
set(VTK_ROOT "")
set(VTK_ROOT "$ENV{VTK_ROOT}")
endif() 

find_package(VTK REQUIRED 
  PATHS 
  "${VTK_ROOT}"
  COMPONENTS 
  CommonCore
  CommonDataModel
  FiltersCore
  IOXML
  IOParallelXML
  ParallelMPI
)

message(STATUS "VTK libraries: ${VTK_LIBRARIES}")
message(STATUS "VTK includes: ${VTK_INCLUDE_DIRS}")

if( NOT TARGET VTK::VTK )
    add_library(VTK::VTK IMPORTED INTERFACE )
    set_property(TARGET VTK::VTK APPEND PROPERTY 
                INTERFACE_INCLUDE_DIRECTORIES "${VTK_INCLUDE_DIRS}")
    set_property(TARGET VTK::VTK APPEND PROPERTY 
                INTERFACE_LINK_LIBRARIES "${VTK_LIBRARIES}")
endif() 
