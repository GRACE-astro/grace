################################################################################
#
# \file      cmake/FindSCICray.cmake
# \author    Adapted from J. Bakosi FindMKL.cmake
# \brief     Find Cray LibSci (BLAS/ScaLAPACK/LAPACK) on Cray systems
# \date      2025-10-05
#
################################################################################

# Find Cray LibSci
#
#  SCICRAY_FOUND - System has Cray LibSci
#  SCICRAY_INCLUDE_DIRS - Include directories (usually empty)
#  SCICRAY_LIBRARIES - Libraries to link (BLAS, LAPACK, ScaLAPACK, etc.)
#
#        Example usage:
#
#                    find_package(SCICRAY)
#                    if(SCICRAY_FOUND)
#                      target_link_libraries(TARGET ${SCICRAY_LIBRARIES})
#                    endif()

# Already in cache?
################################################################################
# FindSCICRAY.cmake
# Simple Cray LibSci finder for GRACE/Kadath
################################################################################

# Cray LibSci root
if(DEFINED ENV{CRAY_LIBSCI_BASE_DIR})
  set(SCICRAY_ROOT $ENV{CRAY_LIBSCI_BASE_DIR})
else()
  set(SCICRAY_ROOT "/opt/cray/pe/libsci")
endif()

# Detect first available crayclang version
file(GLOB CRAYCLANG_DIRS "${SCICRAY_ROOT}/crayclang/*")
list(SORT CRAYCLANG_DIRS)
list(GET CRAYCLANG_DIRS 0 SCICRAY_CRAYCLANG_DIR)

# Include and library directories
set(SCICRAY_INCLUDE_DIRS "${SCICRAY_CRAYCLANG_DIR}/x86_64/include")
set(SCICRAY_LIB_DIR "${SCICRAY_CRAYCLANG_DIR}/x86_64/lib")

message(STATUS "SCICRAY include dir: ${SCICRAY_INCLUDE_DIRS}")
file(GLOB DEBUG_LIBS "${SCICRAY_LIB_DIR}/*")
message(STATUS "SCICRAY lib dir contents: ${DEBUG_LIBS}")

# the multi-threaded MPI version
set(SCICRAY_LIBRARIES "-L${SCICRAY_LIB_DIR} -Wl,--start-group  -lsci_cray_mpi_mp -lsci_cray -Wl,--end-group -lgomp -lpthread -lm -ldl")
# set(SCICRAY_LIBRARIES "-L${SCICRAY_LIB_DIR} -Wl,--start-group -lsci_cray_mpi_mp -lsci_cray_mpi -Wl,--end-group -lgomp -lpthread -lm -ldl")

# the single-threaded MPI version
# set(SCICRAY_LIBRARIES "-L${SCICRAY_LIB_DIR} -lsci_cray -lsci_cray_mpi -lgomp -lpthread -lm -ldl")

# Mark advanced
MARK_AS_ADVANCED(SCICRAY_LIBRARIES SCICRAY_INCLUDE_DIRS)

# Standard FindPackage interface
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCICRAY
    REQUIRED_VARS SCICRAY_LIBRARIES
)

# Allow users to include headers
set(SCICRAY_INCLUDE_DIRS ${SCICRAY_INCLUDE_DIRS} CACHE PATH "Cray LibSci include directories")
