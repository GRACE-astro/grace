
# First, check in /opt/rocm
find_path(ROCPROFILER_INCLUDE_DIR NAMES rocprofiler.h PATHS /opt/rocm/include PATH_SUFFIXES rocprofiler)
find_library(ROCPROFILER_LIBRARY NAMES rocprofiler64v2 PATHS /opt/rocm/lib /opt/rocm/lib64)

find_path(AMD_COMGR_INCLUDE_DIR NAMES amd_comgr.h PATHS /opt/rocm/include PATH_SUFFIXES amd_comgr)
find_library(AMD_COMGR_LIBRARY NAMES amd_comgr PATHS /opt/rocm/lib /opt/rocm/lib64)

# If not found in /opt/rocm, check in standard system directories
if(NOT ROCPROFILER_INCLUDE_DIR OR NOT ROCPROFILER_LIBRARY)
    message(STATUS "rocprofiler not found in /opt/rocm, checking standard system directories")
    
    find_path(ROCPROFILER_INCLUDE_DIR NAMES rocprofiler.h PATH_SUFFIXES rocprofiler)
    find_library(ROCPROFILER_LIBRARY NAMES rocprofiler)
endif()

if(NOT AMD_COMGR_LIBRARY OR NOT AMD_COMGR_INCLUDE_DIR)
    message(STATUS "amd_comgr not found in /opt/rocm, checking standard system directories")
    find_path(AMD_COMGR_INCLUDE_DIR NAMES amd_comgr.h)
    find_library(AMD_COMGR_LIBRARY NAMES amd_comgr)
endif()

# If still not found, check for an environment variable
if(NOT ROCPROFILER_INCLUDE_DIR OR NOT ROCPROFILER_LIBRARY)
    message(STATUS "rocprofiler not found in standard system directories, checking environment variable ROCM_ROCPROFILER_PATH")
    
    if(DEFINED ENV{ROCM_ROCPROFILER_PATH})
        set(ROCM_ROCPROFILER_PATH $ENV{ROCM_ROCPROFILER_PATH})
        
        find_path(ROCPROFILER_INCLUDE_DIR NAMES rocprofiler.h PATHS ${ROCM_ROCPROFILER_PATH}/include PATH_SUFFIXES rocprofiler)
        find_library(ROCPROFILER_LIBRARY NAMES rocprofiler PATHS ${ROCM_ROCPROFILER_PATH}/lib ${ROCM_ROCPROFILER_PATH}/lib64)
    endif()
endif()
if(NOT AMD_COMGR_LIBRARY OR NOT AMD_COMGR_INCLUDE_DIR AND DEFINED ENV{ROCM_COMGR_PATH})
    message(STATUS "amd_comgr not found in standard system directories, checking environment variable ROCM_COMGR_PATH")
    
    set(ROCM_COMGR_PATH $ENV{ROCM_COMGR_PATH})
    find_path(AMD_COMGR_INCLUDE_DIR NAMES amd_comgr.h PATHS ${ROCM_COMGR_PATH}/include PATH_SUFFIXES amd_comgr)
    find_library(AMD_COMGR_LIBRARY NAMES amd_comgr PATHS ${ROCM_COMGR_PATH}/lib ${ROCM_COMGR_PATH}/lib64)
endif()

# Check if rocprofiler was found
if(ROCPROFILER_INCLUDE_DIR AND ROCPROFILER_LIBRARY)
    message(STATUS "rocprofiler found")
    
    add_library(ROCm::rocprofiler INTERFACE IMPORTED)
    set_target_properties(ROCm::rocprofiler PROPERTIES 
        INTERFACE_INCLUDE_DIRECTORIES "${ROCPROFILER_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${ROCPROFILER_LIBRARY}"
    )
else()
    message(FATAL_ERROR "rocprofiler not found. Please install rocprofiler or set ROCM_ROCPROFILER_PATH environment variable.")
endif()
# Check if rocprofiler was found
if(AMD_COMGR_INCLUDE_DIR AND AMD_COMGR_LIBRARY)
    message(STATUS "amd comgr found")
    
    add_library(ROCm::amd_comgr INTERFACE IMPORTED)
    set_target_properties(ROCm::amd_comgr PROPERTIES 
        INTERFACE_INCLUDE_DIRECTORIES "${AMD_COMGR_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${AMD_COMGR_LIBRARY}"
    )
else()
    message(FATAL_ERROR "amd comgr  not found. Please install rocprofiler or set ROCM_ROCPROFILER_PATH environment variable.")
endif()

add_library( GRACE_GPUProfiling INTERFACE  )
target_link_libraries(
    GRACE_GPUProfiling INTERFACE ROCm::rocprofiler ROCm::amd_comgr
)