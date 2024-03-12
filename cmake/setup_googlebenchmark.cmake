
find_package(GoogleBenchmark REQUIRED)

message(STATUS "Google Benchmark libraries: ${GoogleBenchmark_LIBRARIES}")
message(STATUS "Google Benchmark includes: ${GoogleBenchmark_INCLUDE_DIRS}")

if( NOT TARGET GoogleBenchmark::benchmark )
  add_library( GoogleBenchmark::benchmark IMPORTED INTERFACE)

  set_property( TARGET GoogleBenchmark::benchmark APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${GoogleBenchmark_INCLUDE_DIRS}")
  set_property( TARGET GoogleBenchmark::benchmark APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "${GoogleBenchmark_LIBRARIES}")

endif()
