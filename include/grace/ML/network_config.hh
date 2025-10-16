#ifndef GRACE_ML_NETWORK_CONFIG_HH
#define GRACE_ML_NETWORK_CONFIG_HH

#include <Kokkos_Core.hpp>
#include <grace_config.h>

#ifdef GRACE_ENABLE_ML

#include <string>

namespace grace {

#if defined(GRACE_ENABLE_CUDA)
using default_space_ai = Kokkos::CudaSpace;
#elif defined(GRACE_ENABLE_HIP)
using default_space_ai = Kokkos::HIPSpace;
#elif defined(GRACE_ENABLE_OMP) || defined(GRACE_ENABLE_SERIAL)
using default_space_ai = Kokkos::HostSpace;
#endif

namespace ml {

using real_t = double;
using Matrix2D = Kokkos::View<real_t**, grace::default_space_ai>;
using Vector1D = Kokkos::View<real_t*, grace::default_space_ai>;
using HostMatrix2D = typename Matrix2D::HostMirror;
using HostVector1D = typename Vector1D::HostMirror;

struct NetworkConfig {
    int input_size = 0;
    int hidden_size = 0;
    int output_size = 0;
    bool has_normalization = false;
    bool is_physics_guided = false;
    real_t correction_scale = 0.1;
    
    HostVector1D input_min;
    HostVector1D input_max;
    HostVector1D output_min;
    HostVector1D output_max;
    
    std::string activation_function = "tanh";
    std::string model_type = "standard";
};

} // namespace ml
} // namespace grace

#endif // GRACE_ENABLE_ML
#endif // GRACE_ML_NETWORK_CONFIG_HH