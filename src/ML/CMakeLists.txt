# ML Inference Library
add_library(grace_ml OBJECT
    model_loader.cpp
    device_network.cpp
    ml_networks.cpp
    ml_runtime_stats.cpp
)

target_include_directories(grace_ml PRIVATE
    "${HEADER_DIR}"
)

target_compile_definitions(grace_ml PRIVATE
    GRACE_ENABLE_ML
)

# PUBLIC für Kokkos und KokkosKernels (weil in Headers verwendet)
target_link_libraries(grace_ml PUBLIC
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
)

# PRIVATE für alles andere (nur in .cpp verwendet)
target_link_libraries(grace_ml PRIVATE
    MPI::MPI_CXX
    HDF5::HDF5
    spdlog::spdlog
    yaml_cpp::yaml
    p4est::sc
    p4est::p4est
)