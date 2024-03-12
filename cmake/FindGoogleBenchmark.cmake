
if(NOT GOOGLE_BENCHMARK_ROOT )
  set(GOOGLE_BENCHMARK_ROOT "")
  set(GOOGLE_BENCHMARK_ROOT "$ENV{GOOGLE_BENCHMARK_ROOT}")
endif()


find_path( GoogleBenchmark_INCLUDE_DIR
  NAMES benchmark/benchmark.h
  PATHS ${GOOGLE_BENCHMARK_ROOT}
  PATH_SUFFIXES include
  )

find_library( GoogleBenchmark_LIBRARY
  NAMES benchmark
  PATHS ${GOOGLE_BENCHMARK_ROOT}
  PATH_SUFFIXES build/src
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  GoogleBenchmark
  FOUND_VAR GoogleBenchmark_FOUND
  REQUIRED_VARS GoogleBenchmark_LIBRARY GoogleBenchmark_INCLUDE_DIR
  )

if( GoogleBenchmark_FOUND)
  set( GoogleBenchmark_INCLUDE_DIRS "${GoogleBenchmark_INCLUDE_DIR}")
  set( GoogleBenchmark_LIBRARIES "${GoogleBenchmark_LIBRARY}") 
endif()

mark_as_advanced(
  GoogleBenchmark_INCLUDE_DIR
  GoogleBenchmark_LIBRARY
  )
