# setup_catch2.cmake

# Use environment variable if not already set
if (NOT CATCH2_ROOT)
    set(CATCH2_ROOT "$ENV{CATCH2_ROOT}")
endif()

# Try to find Catch2 package
find_package(Catch2 3 REQUIRED PATHS "${CATCH2_ROOT}" NO_DEFAULT_PATH)

# If found, set variables for later use
if (Catch2_FOUND)
    message(STATUS "Found Catch2 at ${Catch2_DIR}")
    set(catch2_INCLUDE_DIRS ${Catch2_INCLUDE_DIRS})
    set(catch2_LIBRARIES Catch2::Catch2)
else()
    message(FATAL_ERROR "Catch2 not found. Please set CATCH2_ROOT to your Catch2 installation")
endif()

# Optional: Append extras if needed (v3 may not have this)
# list(APPEND CMAKE_MODULE_PATH "${CATCH2_ROOT}/extras")

